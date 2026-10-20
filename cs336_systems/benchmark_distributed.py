import argparse
from contextlib import nullcontext
from datetime import datetime
import logging
import os
import pprint
import sys
import timeit

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.functions import cross_entropy_loss
from cs336_basics.optim.adamw import AdamW
from cs336_basics.optim.functions import lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM
from cs336_systems.distributed import FlatDDP, FSDP, NaiveDDP, OverlapDDP, ShardedOptimizer


IMPL_REGISTRY = {
    "naive": NaiveDDP,
    "flat": FlatDDP,
    "overlap": OverlapDDP,
    "fsdp": FSDP,
}

WORLD_SIZE = 2
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29500


def get_autocast_context(args: argparse.Namespace, device: torch.device):
    if args.bf16 and args.impl != "fsdp":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def get_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def train(rank: int, args: argparse.Namespace, run_dir: str) -> None:
    world_size = WORLD_SIZE
    backend = "nccl" if args.device == "cuda" else "gloo"

    if args.device == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    log_path = os.path.join(run_dir, f"rank{rank}.txt")
    logger = get_logger(log_path)
    logger.info("[rank %d] Args:\n%s", rank, pprint.pformat(vars(args), sort_dicts=False))
    logger.info("[rank %d] Device: %s, world_size=%d, backend=%s", rank, device, world_size, backend)

    nvtx = bool(args.nvtx and device.type == "cuda")
    memory_profile = bool(args.memory_profile and device.type == "cuda")

    try:
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))

        model = TransformerLM(
            vocab_size=int(args.vocab_size),
            context_length=int(args.context_length),
            num_layers=int(args.num_layers),
            d_model=int(args.d_model),
            d_ff=int(args.d_ff),
            num_heads=int(args.num_heads),
            activation=str(args.activation),
            with_rope=(not bool(args.no_rope)),
            theta=float(args.theta),
            with_norm=(not bool(args.no_norm)),
            pre_norm=(not bool(args.post_norm)),
            eps=float(args.lm_eps),
        )
        model.to(device=device)

        if nvtx:
            import cs336_basics.modules.attention as attention_module
            from cs336_systems.annotated_functions import annotated_scaled_dot_product_attention

            attention_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
            for index, block in enumerate(model.layers):
                block.forward = torch.cuda.nvtx.range(f"TransformerBlock {index}")(block.forward)

        if args.impl == "fsdp":
            compute_dtype = torch.bfloat16 if args.bf16 else None
            model = FSDP(model, compute_dtype=compute_dtype)
        else:
            model = IMPL_REGISTRY[args.impl](model)
        model.train()

        optimizer_kwargs = {
            "lr": float(args.lr),
            "betas": (float(args.beta_low), float(args.beta_high)),
            "eps": float(args.optim_eps),
            "weight_decay": float(args.weight_decay),
        }
        if args.impl == "fsdp":
            optimizer = AdamW(params=model.parameters(), **optimizer_kwargs)
        elif args.oss:
            optimizer = ShardedOptimizer(model.parameters(), AdamW, **optimizer_kwargs)
        else:
            optimizer = AdamW(params=model.parameters(), **optimizer_kwargs)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.seed))

        local_batch_size = int(args.batch_size) // world_size
        logger.info(
            "[rank %d] local batch size=%d, data offset=%d",
            rank,
            local_batch_size,
            rank * local_batch_size,
        )

        total_costs: list[float] = []
        forward_costs: list[float] = []
        backward_costs: list[float] = []
        comm_costs: list[float] = []
        optim_costs: list[float] = []
        step = 0

        for phase, steps in (("warmup", args.warm_steps), ("measure", args.mark_steps)):
            if phase == "measure":
                dist.barrier()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if memory_profile:
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.memory._record_memory_history(max_entries=1000000)

            for _ in range(steps):
                global_x = torch.randint(
                    low=0,
                    high=int(args.vocab_size),
                    size=(int(args.batch_size), int(args.context_length)),
                    dtype=torch.long,
                    device=device,
                    generator=generator,
                )
                global_y = torch.randint(
                    low=0,
                    high=int(args.vocab_size),
                    size=(int(args.batch_size), int(args.context_length)),
                    dtype=torch.long,
                    device=device,
                    generator=generator,
                )

                offset = rank * local_batch_size
                x = global_x[offset: offset + local_batch_size]
                y = global_y[offset: offset + local_batch_size]

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                with (torch.cuda.nvtx.range("step") if nvtx else nullcontext()):
                    start = timeit.default_timer()

                    with (torch.cuda.nvtx.range("forward") if nvtx else nullcontext()):
                        with get_autocast_context(args, device):
                            logits = model(x)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        forward_end = timeit.default_timer()

                    logits = logits.reshape(-1, logits.size(-1))
                    y = y.reshape(-1)
                    with get_autocast_context(args, device):
                        loss = cross_entropy_loss(logits.float(), y)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                    with (torch.cuda.nvtx.range("backward") if nvtx else nullcontext()):
                        backward_start = timeit.default_timer()
                        loss.backward()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        backward_end = timeit.default_timer()

                    with (torch.cuda.nvtx.range("gradient_communication") if nvtx else nullcontext()):
                        comm_start = timeit.default_timer()
                        model.finish_gradient_synchronization()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        comm_end = timeit.default_timer()

                    with (torch.cuda.nvtx.range("optimizer") if nvtx else nullcontext()):
                        # Gradient clipping is omitted for comparability across implementations.
                        lr = lr_cosine_schedule(
                            t=step,
                            alpha_min=float(args.cosine_min),
                            alpha_max=float(args.lr),
                            warmup_iters=int(args.warmup_iters),
                            cosine_iters=int(args.cosine_iters),
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        if args.oss:
                            for group in optimizer.optimizer.param_groups:
                                group["lr"] = lr

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        total_end = timeit.default_timer()

                step += 1

                if phase == "measure":
                    total_costs.append(total_end - start)
                    forward_costs.append(forward_end - start)
                    backward_costs.append(backward_end - backward_start)
                    comm_costs.append(comm_end - comm_start)
                    optim_costs.append(total_end - comm_end)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

        if memory_profile:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            logger.info(
                "[rank %d] Peak allocated memory: %.2f MiB",
                rank,
                torch.cuda.max_memory_allocated(device) / 1024**2,
            )
            snapshot_path = os.path.join(run_dir, f"memory_snapshot_rank{rank}.pickle")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info("[rank %d] Memory snapshot saved to %s", rank, snapshot_path)

        dist.barrier()

        mean_total = np.mean(total_costs)
        logger.info("[rank %d] Forward: %.5f ± %.5f s", rank, np.mean(forward_costs), np.std(forward_costs))
        logger.info("[rank %d] Backward: %.5f ± %.5f s", rank, np.mean(backward_costs), np.std(backward_costs))
        logger.info("[rank %d] Gradient communication: %.5f ± %.5f s", rank, np.mean(comm_costs), np.std(comm_costs))
        logger.info("[rank %d] Optimizer: %.5f ± %.5f s", rank, np.mean(optim_costs), np.std(optim_costs))
        logger.info("[rank %d] Total step: %.5f ± %.5f s", rank, mean_total, np.std(total_costs))
        logger.info("[rank %d] Communication ratio: %.2f%%", rank, 100.0 * np.mean(comm_costs) / mean_total)

        gathered_stats = [None] * world_size
        stats = {
            "total": total_costs,
            "forward": forward_costs,
            "backward": backward_costs,
            "comm": comm_costs,
            "optim": optim_costs,
        }
        dist.all_gather_object(gathered_stats, stats)

        if rank == 0:
            aggregated_total = [value for rank_stats in gathered_stats for value in rank_stats["total"]]
            aggregated_forward = [value for rank_stats in gathered_stats for value in rank_stats["forward"]]
            aggregated_backward = [value for rank_stats in gathered_stats for value in rank_stats["backward"]]
            aggregated_comm = [value for rank_stats in gathered_stats for value in rank_stats["comm"]]
            aggregated_optim = [value for rank_stats in gathered_stats for value in rank_stats["optim"]]
            aggregated_mean_total = np.mean(aggregated_total)

            logger.info("[aggregated] Forward: %.5f ± %.5f s", np.mean(aggregated_forward), np.std(aggregated_forward))
            logger.info("[aggregated] Backward: %.5f ± %.5f s", np.mean(aggregated_backward), np.std(aggregated_backward))
            logger.info("[aggregated] Gradient communication: %.5f ± %.5f s", np.mean(aggregated_comm), np.std(aggregated_comm))
            logger.info("[aggregated] Optimizer: %.5f ± %.5f s", np.mean(aggregated_optim), np.std(aggregated_optim))
            logger.info("[aggregated] Total step: %.5f ± %.5f s", aggregated_mean_total, np.std(aggregated_total))
            logger.info("[aggregated] Communication ratio: %.2f%%", 100.0 * np.mean(aggregated_comm) / aggregated_mean_total)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser("cs336 assignment2 distributed benchmark")
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--save_root", type=str, required=False, default="./outputs")
    parser.add_argument("--device", type=str, choices=("cuda", "cpu"), required=False, default="cuda")
    parser.add_argument("--impl", type=str, choices=tuple(IMPL_REGISTRY), required=False, default="naive")
    parser.add_argument("--oss", action="store_true")
    parser.add_argument("--batch_size", type=int, required=False, default=4)
    parser.add_argument("--warm_steps", type=int, required=True)
    parser.add_argument("--mark_steps", type=int, required=True)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--memory_profile", action="store_true")

    parser.add_argument("--vocab_size", type=int, required=False, default=10000)
    parser.add_argument("--context_length", type=int, required=False, default=512)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--activation", type=str, required=False, default="swiglu")
    parser.add_argument("--no_rope", action="store_true", default=False)
    parser.add_argument("--theta", type=float, required=False, default=10000.0)
    parser.add_argument("--no_norm", action="store_true", default=False)
    parser.add_argument("--post_norm", action="store_true", default=False)
    parser.add_argument("--lm_eps", type=float, required=False, default=1e-5)

    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--beta_low", type=float, required=False, default=0.90)
    parser.add_argument("--beta_high", type=float, required=False, default=0.99)
    parser.add_argument("--optim_eps", type=float, required=False, default=1e-8)
    parser.add_argument("--weight_decay", type=float, required=False, default=1e-2)

    parser.add_argument("--cosine_min", type=float, required=False, default=1e-6)
    parser.add_argument("--warmup_iters", type=int, required=False, default=0)
    parser.add_argument("--cosine_iters", type=int, required=False, default=10000)
    args = parser.parse_args()

    if args.impl == "fsdp" and args.oss:
        raise ValueError("FSDP already shards optimizer state; --oss cannot be used with FSDP")

    os.makedirs(args.save_root, exist_ok=True)
    run_dir = os.path.join(
        args.save_root,
        f"benchmark_{args.impl}{'_oss' if args.oss else ''}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    )
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    os.environ.setdefault("MASTER_ADDR", MASTER_ADDR)
    os.environ.setdefault("MASTER_PORT", str(MASTER_PORT))

    print(f"Run directory: {run_dir}", flush=True)
    mp.spawn(train, args=(args, run_dir), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
