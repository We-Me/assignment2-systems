import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WARMUP_STEPS = 5
MEASURE_STEPS = 10


def benchmark_worker(rank: int, world_size: int, backend: str) -> None:
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(1)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        for label, size_bytes in (("1MiB", 2**20), ("10MiB", 10 * 2**20), ("100MiB", 100 * 2**20), ("1GiB", 2**30)):
            data = torch.ones(size_bytes // 4, dtype=torch.float32, device=device)
            for _ in range(WARMUP_STEPS):
                dist.all_reduce(data)

            timings = []
            for _ in range(MEASURE_STEPS):
                dist.barrier()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                dist.all_reduce(data)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append((time.perf_counter() - start) * 1000)

            rank_timings = [None] * world_size
            dist.all_gather_object(rank_timings, timings)
            if rank == 0:
                samples = [elapsed for worker_timings in rank_timings for elapsed in worker_timings]
                print(
                    f"{world_size:>9} {label:>6} {statistics.mean(samples):>12.3f} {statistics.pstdev(samples):>12.3f}",
                    flush=True,
                )
            del data
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark single-node all-reduce.")
    parser.add_argument("device", choices=("cpu", "gpu"))
    args = parser.parse_args()
    use_gpu = args.device == "gpu"
    backend = "nccl" if use_gpu else "gloo"

    gpu_count = 0
    if use_gpu:
        if not dist.is_nccl_available():
            parser.error("GPU benchmarks require PyTorch with NCCL support (normally on Linux).")
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            parser.error(f"At least 2 CUDA GPUs are required; found {gpu_count}.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    print(
        f"device={args.device}, backend={backend}, dtype=float32, "
        f"warmup={WARMUP_STEPS}, iterations={MEASURE_STEPS}",
        flush=True,
    )
    print(f"{'Processes':>9} {'Size':>6} {'Mean (ms)':>12} {'Std (ms)':>12}", flush=True)
    for world_size in (2, 4, 6):
        if use_gpu and world_size > gpu_count:
            print(f"Skipping {world_size} GPUs: only {gpu_count} available.", flush=True)
            continue
        mp.spawn(benchmark_worker, args=(world_size, backend), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
