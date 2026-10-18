import argparse
import csv
import gc
import json
import math
import statistics
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cs336_basics.transformer_lm import TransformerLM


SMALL = {"num_layers": 12, "d_model": 768, "d_ff": 2048, "num_heads": 12}
MIB = 1024**2
FIELDS = [
    "block_size", 
    "segments", 
    "status", 
    "start_allocated_mib",
    "forward_peak_allocated_mib", 
    "peak_allocated_mib", 
    "peak_reserved_mib",
    "peak_minus_start_mib", 
    "step_ms_mean", 
    "step_ms_std", 
    "loss", 
    "error",
]


def run_segment(layers, x, start, stop):
    for index in range(start, stop):
        x = layers[index](x)
    return x


def forward(model, tokens, block_size):
    x = model.token_embeddings(tokens)
    if block_size == 0:
        x = model.layers(x)
    else:
        for start in range(0, len(model.layers), block_size):
            # Bind the bounds now: checkpoint invokes this callable again in backward.
            segment = partial(
                run_segment, model.layers,
                start=start, stop=min(start + block_size, len(model.layers)),
            )
            x = checkpoint(segment, x, use_reentrant=False)
    if model.with_norm:
        x = model.norm(x)
    return model.lm_head(x)


def train_step(model, tokens, targets, block_size, device):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()

    logits = forward(model, tokens, block_size)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.cuda.synchronize(device)
    forward_peak = torch.cuda.max_memory_allocated(device)

    loss.backward()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    peak_allocated = torch.cuda.max_memory_allocated(device)

    return {
        "start_allocated_mib": start_allocated / MIB,
        "forward_peak_allocated_mib": forward_peak / MIB,
        "peak_allocated_mib": peak_allocated / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "peak_minus_start_mib": (peak_allocated - start_allocated) / MIB,
        "step_ms": elapsed_ms,
        "loss": loss.item(),
    }


def benchmark_case(args, block_size, device):
    # Recreate identical weights and inputs for each policy; never update weights.
    torch.manual_seed(args.seed)
    model = TransformerLM(
        **SMALL, vocab_size=args.vocab_size, context_length=args.context_length,
        device=device, dtype=torch.float32,
    ).train()
    if args.compile:
        for index, block in enumerate(model.layers):
            model.layers[index] = torch.compile(block, fullgraph=True)

    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, tokens.shape, device=device)

    # Warm up forward AND backward, including compilation when requested.
    for _ in range(args.warm_steps):
        train_step(model, tokens, targets, block_size, device)
    measurements = [
        train_step(model, tokens, targets, block_size, device)
        for _ in range(args.mark_steps)
    ]
    times = [step["step_ms"] for step in measurements]
    row = {
        "block_size": block_size,
        "segments": math.ceil(SMALL["num_layers"] / block_size) if block_size else 0,
        "status": "ok",
        "start_allocated_mib": measurements[0]["start_allocated_mib"],
        "step_ms_mean": statistics.mean(times),
        "step_ms_std": statistics.pstdev(times),
        "loss": measurements[0]["loss"],
        "error": "",
    }
    for key in (
        "forward_peak_allocated_mib", "peak_allocated_mib",
        "peak_reserved_mib", "peak_minus_start_mib",
    ):
        row[key] = max(step[key] for step in measurements)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--block_sizes", type=int, nargs="+", default=list(range(13)), help="0=baseline; 1..12=blocks per checkpoint")
    parser.add_argument("--warm_steps", type=int, default=5)
    parser.add_argument("--mark_steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="compile each Transformer block before applying checkpoint policies")
    parser.add_argument("--save_root", type=Path, help="default: outputs/gradient_checkpointing/small/<timestamp>")
    args = parser.parse_args()
    if min(args.batch_size, args.context_length, args.vocab_size, args.mark_steps) < 1 or args.warm_steps < 0:
        parser.error("batch_size, context_length, vocab_size and mark_steps must be positive; warm_steps must be nonnegative")
    if any(size < 0 or size > SMALL["num_layers"] for size in args.block_sizes):
        parser.error("block_sizes must be between 0 (baseline) and 12")
    args.block_sizes = sorted(set(args.block_sizes))
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required to measure GPU peak memory. Run this script in the project's CUDA/WSL environment.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    save_root = args.save_root or Path("outputs/gradient_checkpointing/small") / datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    save_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": "small", **SMALL,
        "batch_size": args.batch_size, "context_length": args.context_length,
        "vocab_size": args.vocab_size, "dtype": "float32", "seed": args.seed,
        "compile_blocks": args.compile, "warm_steps": args.warm_steps,
        "mark_steps": args.mark_steps, "block_sizes": args.block_sizes,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "scope": "full LM forward + cross_entropy + backward; no optimizer update",
        "peak_metric": "max_memory_allocated across measured steps, including live parameters and gradients",
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print("\nblock_size  status  forward_peak_MiB  total_peak_MiB  step_ms", flush=True)
    rows = []
    with (save_root / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        output.flush()
        for block_size in args.block_sizes:
            try:
                row = benchmark_case(args, block_size, device)
            except torch.cuda.OutOfMemoryError as error:
                row = {
                    "block_size": block_size,
                    "segments": math.ceil(SMALL["num_layers"] / block_size) if block_size else 0,
                    "status": "oom", "error": str(error).splitlines()[0],
                }
            finally:
                # No model or graph is kept between cases, even when a case OOMs.
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            writer.writerow(row)
            output.flush()
            if row["status"] == "ok":
                print(
                    f"{block_size:>10}  ok      {row['forward_peak_allocated_mib']:>12.2f}"
                    f"  {row['peak_allocated_mib']:>14.2f}  {row['step_ms_mean']:>7.2f}",
                    flush=True,
                )
            else:
                print(f"{block_size:>10}  OOM     {row['error']}", flush=True)

    candidates = [row for row in rows if row["block_size"] > 0 and row["status"] == "ok"]
    best = min(candidates, key=lambda row: row["peak_allocated_mib"]) if candidates else None
    neighbors = {}
    if best:
        for label, size in (("smaller", best["block_size"] - 1), ("larger", best["block_size"] + 1)):
            if not 1 <= size <= SMALL["num_layers"]:
                neighbors[label] = {"block_size": size, "status": "outside full-block search range"}
            else:
                neighbors[label] = next((row for row in rows if row["block_size"] == size), {"block_size": size, "status": "not measured"})
        print(f"\nBest measured checkpoint block size: {best['block_size']} ({best['peak_allocated_mib']:.2f} MiB)")
        print("Adjacent full-block sizes: " + json.dumps(neighbors, ensure_ascii=False))
    else:
        print("\nNo checkpoint candidate completed; try a smaller batch_size or context_length.")

    summary = {**metadata, "results": rows, "best_checkpoint": best, "adjacent_sizes": neighbors}
    (save_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {save_root.resolve()}", flush=True)
    if not candidates:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
