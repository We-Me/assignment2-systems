import argparse
from contextlib import nullcontext
from datetime import datetime
import logging
import os
import pprint
import sys
import timeit

import torch
import numpy as np

from cs336_basics.functions import cross_entropy_loss
from cs336_basics.optim.adamw import AdamW
from cs336_basics.optim.functions import gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


def get_autocast_context(args: argparse.Namespace):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 else nullcontext()
    )


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


def inference_part(
    model: TransformerLM, 
    device: torch.device,
    logger: logging.Logger,
    args: argparse.Namespace
):
    model.eval()
    forward_costs = []
    with torch.no_grad():
        for phase, steps in (
            ("warmup", args.warm_steps),
            ("measure", args.mark_steps),
        ):
            # 每个阶段只有一个外层区间，measure 覆盖全部正式测量步骤。
            torch.cuda.synchronize(device)
            if args.memory_profile and phase == "measure":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            phase_context = (
                torch.cuda.nvtx.range(phase) if args.nvtx else nullcontext()
            )
            with phase_context:
                for _ in range(steps):
                    x = torch.randint(
                        low=0,
                        high=args.vocab_size,
                        size=(args.batch_size, args.context_length),
                        dtype=torch.long,
                        device=device,
                    )

                    torch.cuda.synchronize(device)
                    forward_context = (
                        torch.cuda.nvtx.range("forward")
                        if args.nvtx else nullcontext()
                    )
                    with forward_context:
                        # 标注的进入和退出位于原有计时区间之外。
                        start = timeit.default_timer()
                        with get_autocast_context(args):
                            model(x)
                        torch.cuda.synchronize(device)
                        forward_end = timeit.default_timer()

                    if phase == "measure":
                        forward_costs.append(forward_end - start)

            if args.memory_profile and phase == "measure":
                torch.cuda.synchronize(device)
                logger.info("Peak allocated memory: %.2f MiB", torch.cuda.max_memory_allocated(device) / 1024**2)
                snapshot_path = os.path.join(args.save_root, "memory_snapshot.pickle")
                torch.cuda.memory._dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                logger.info("Memory snapshot saved to %s", snapshot_path)

        logger.info("Inference: %.5f ± %.5f s", np.mean(forward_costs), np.std(forward_costs))


def forward_part(
    model: TransformerLM, 
    device: torch.device,
    logger: logging.Logger,
    args: argparse.Namespace
):
    model.train()
    forward_costs = []
    for phase, steps in (
        ("warmup", args.warm_steps),
        ("measure", args.mark_steps),
    ):
        torch.cuda.synchronize(device)
        if args.memory_profile and phase == "measure":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        phase_context = (
            torch.cuda.nvtx.range(phase) if args.nvtx else nullcontext()
        )

        with phase_context:
            for _ in range(steps):
                x = torch.randint(
                    low=0,
                    high=args.vocab_size,
                    size=(args.batch_size, args.context_length),
                    dtype=torch.long,
                    device=device,
                )

                torch.cuda.synchronize(device)
                forward_context = (
                    torch.cuda.nvtx.range("forward")
                    if args.nvtx else nullcontext()
                )
                with forward_context:
                    start = timeit.default_timer()
                    with get_autocast_context(args):
                        model(x)
                    torch.cuda.synchronize(device)
                    forward_end = timeit.default_timer()

                if phase == "measure":
                    forward_costs.append(forward_end - start)

        if args.memory_profile and phase == "measure":
            torch.cuda.synchronize(device)
            logger.info("Peak allocated memory: %.2f MiB", torch.cuda.max_memory_allocated(device) / 1024**2)
            snapshot_path = os.path.join(args.save_root, "memory_snapshot.pickle")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info("Memory snapshot saved to %s", snapshot_path)
        
    logger.info("Forward: %.5f ± %.5f s", np.mean(forward_costs), np.std(forward_costs))        


def backward_part(
    model: TransformerLM, 
    device: torch.device,
    logger: logging.Logger,
    args: argparse.Namespace
):
    model.train()
    forward_costs = []
    for_back_costs = []
    for phase, steps in (
        ("warmup", args.warm_steps),
        ("measure", args.mark_steps),
    ):
        torch.cuda.synchronize(device)
        if args.memory_profile and phase == "measure":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        phase_context = (
            torch.cuda.nvtx.range(phase) if args.nvtx else nullcontext()
        )

        with phase_context:
            for _ in range(steps):
                x = torch.randint(
                    low=0,
                    high=args.vocab_size,
                    size=(args.batch_size, args.context_length),
                    dtype=torch.long,
                    device=device,
                )

                y = torch.randint(
                    low=0,
                    high=args.vocab_size,
                    size=(args.batch_size, args.context_length),
                    dtype=torch.long,
                    device=device,
                )

                torch.cuda.synchronize(device)
                forback_context = (
                    torch.cuda.nvtx.range("forback_context")
                    if args.nvtx else nullcontext()
                )
                with forback_context:
                    forward_context = (
                        torch.cuda.nvtx.range("forward")
                        if args.nvtx else nullcontext()
                    )
                    with forward_context:
                        start = timeit.default_timer()
                        with get_autocast_context(args):
                            logits = model(x)
                        torch.cuda.synchronize(device)
                        forward_end = timeit.default_timer()

                    logits = logits.reshape(-1, logits.size(-1))
                    y = y.reshape(-1)
                    with get_autocast_context(args):
                        loss = cross_entropy_loss(logits.float(), y)
                    torch.cuda.synchronize(device)

                    backward_context = (
                        torch.cuda.nvtx.range("backward")
                        if args.nvtx else nullcontext()
                    )
                    with backward_context:
                        loss.backward()
                        torch.cuda.synchronize(device)
                        backward_end = timeit.default_timer()

                model.zero_grad(set_to_none=True)

                if phase == "measure":
                    forward_costs.append(forward_end - start)
                    for_back_costs.append(backward_end - start)

        if args.memory_profile and phase == "measure":
            torch.cuda.synchronize(device)
            logger.info("Peak allocated memory: %.2f MiB", torch.cuda.max_memory_allocated(device) / 1024**2)
            snapshot_path = os.path.join(args.save_root, "memory_snapshot.pickle")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info("Memory snapshot saved to %s", snapshot_path)

    logger.info("Forward: %.5f ± %.5f s", np.mean(forward_costs), np.std(forward_costs))
    logger.info("Forward + Backward: %.5f ± %.5f s", np.mean(for_back_costs), np.std(for_back_costs))


def optim_part(
    model: TransformerLM,
    optimizer: AdamW,
    device: torch.device,
    logger: logging.Logger,
    args: argparse.Namespace
):
    model.train()
    forward_costs = []
    for_back_costs = []
    total_costs = []
    step = 0
    for phase, steps in (
        ("warmup", args.warm_steps),
        ("measure", args.mark_steps),
    ):
        torch.cuda.synchronize(device)
        if args.memory_profile and phase == "measure":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        phase_context = (
            torch.cuda.nvtx.range(phase) if args.nvtx else nullcontext()
        )

        with phase_context:
            for _ in range(steps):
                x = torch.randint(
                    low=0,
                    high=args.vocab_size,
                    size=(args.batch_size, args.context_length),
                    dtype=torch.long,
                    device=device,
                )

                y = torch.randint(
                    low=0,
                    high=args.vocab_size,
                    size=(args.batch_size, args.context_length),
                    dtype=torch.long,
                    device=device,
                )

                torch.cuda.synchronize(device)
                total_context = (
                    torch.cuda.nvtx.range("total_context")
                    if args.nvtx else nullcontext()
                )
                with total_context:
                    forward_context = (
                        torch.cuda.nvtx.range("forward")
                        if args.nvtx else nullcontext()
                    )
                    with forward_context:
                        start = timeit.default_timer()
                        with get_autocast_context(args):
                            logits = model(x)
                        torch.cuda.synchronize(device)
                        forward_end = timeit.default_timer()

                    logits = logits.reshape(-1, logits.size(-1))
                    y = y.reshape(-1)
                    with get_autocast_context(args):
                        loss = cross_entropy_loss(logits.float(), y)
                    torch.cuda.synchronize(device)

                    backward_context = (
                        torch.cuda.nvtx.range("backward")
                        if args.nvtx else nullcontext()
                    )
                    with backward_context:
                        loss.backward()
                        torch.cuda.synchronize(device)
                        backward_end = timeit.default_timer()

                    optim_context = (
                        torch.cuda.nvtx.range("optim")
                        if args.nvtx else nullcontext()
                    )
                    with optim_context:
                        gradient_clipping(
                            parameters = model.parameters(),
                            max_l2_norm=float(args.max_l2_norm),
                        )
                
                        lr = lr_cosine_schedule(
                            t=step,
                            alpha_min=float(args.cosine_min),
                            alpha_max=float(args.lr),
                            warmup_iters=int(args.warmup_iters),
                            cosine_iters=int(args.cosine_iters),
                        )
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                
                        torch.cuda.synchronize(device)
                        total_end = timeit.default_timer()
                        
                step = step + 1

                if phase == "measure":
                    forward_costs.append(forward_end - start)
                    for_back_costs.append(backward_end - start)
                    total_costs.append(total_end - start)

        if args.memory_profile and phase == "measure":
            torch.cuda.synchronize(device)
            logger.info("Peak allocated memory: %.2f MiB", torch.cuda.max_memory_allocated(device) / 1024**2)
            snapshot_path = os.path.join(args.save_root, "memory_snapshot.pickle")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            logger.info("Memory snapshot saved to %s", snapshot_path)

    logger.info("Forward: %.5f ± %.5f s", np.mean(forward_costs), np.std(forward_costs))
    logger.info("Forward + Backward: %.5f ± %.5f s", np.mean(for_back_costs), np.std(for_back_costs))
    logger.info("Total step: %.5f ± %.5f s", np.mean(total_costs), np.std(total_costs))


def benchmark(args: argparse.Namespace):
    seed=int(args.seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(args.save_root, exist_ok=True)
    log_path = os.path.join(args.save_root, f"benchmark_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt")
    logger = get_logger(log_path)
    logger.info(f'The logging will save at {log_path}')

    logger.info("Args:\n%s", pprint.pformat(vars(args), sort_dicts=False))

    device = torch.device(args.device)
    logger.info(f'Device: {device}')

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

    if args.nvtx:
        import cs336_basics.modules.attention as attention_module
        from cs336_systems.annotated_functions import annotated_scaled_dot_product_attention

        attention_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        for index, block in enumerate(model.layers):
            block.forward = torch.cuda.nvtx.range(f"TransformerBlock {index}")(block.forward)

    if args.mode == "inference":
        inference_part(model, device, logger, args)
    elif args.mode == "forward":
        forward_part(model, device, logger, args)
    elif args.mode == "backward":
        backward_part(model, device, logger, args)
    elif args.mode == "total":
        optimizer = AdamW(
            params=model.parameters(),
            lr=float(args.lr),
            betas=(float(args.beta_low), float(args.beta_high)),
            eps=float(args.optim_eps),
            weight_decay=float(args.weight_decay),
        )
        optim_part(model, optimizer, device, logger, args)
    else:
        raise ValueError("args.mode should in [inference, forward, backward, total]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cs336 assignment1 basics training')
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('--save_root', type=str, required=False, default="./outputs")
    parser.add_argument('--device', type=str, required=False, default="cuda")
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--warm_steps', type=int, required=True)
    parser.add_argument('--mark_steps', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--nvtx', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--memory_profile', action='store_true')

    parser.add_argument('--vocab_size', type=int, required=False, default=10000)
    parser.add_argument('--context_length', type=int, required=False, default=512)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--num_heads', type=int, required=True)
    parser.add_argument('--activation', type=str, required=False, default="swiglu")
    parser.add_argument('--no_rope', action='store_true', default=False)
    parser.add_argument('--theta', type=float, required=False, default=10000.0)
    parser.add_argument('--no_norm', action='store_true', default=False)
    parser.add_argument('--post_norm', action='store_true', default=False)
    parser.add_argument('--lm_eps', type=float, required=False, default=1e-5)

    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--beta_low', type=float, required=False, default=0.90)
    parser.add_argument('--beta_high', type=float, required=False, default=0.99)
    parser.add_argument('--optim_eps', type=float, required=False, default=1e-8)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-2)
    
    parser.add_argument('--max_l2_norm', type=float, required=False, default=1e2)
    parser.add_argument('--cosine_min', type=float, required=False, default=1e-6)
    parser.add_argument('--warmup_iters', type=int, required=False, default=0)
    parser.add_argument('--cosine_iters', type=int, required=False, default=10000)
    args = parser.parse_args()

    benchmark(args)
