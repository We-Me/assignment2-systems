import argparse
from datetime import datetime
import logging
import os
import pprint
import sys

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from cs336_basics.dataset import StreamingDataset
from cs336_basics.functions import cross_entropy_loss, load_checkpoint, save_checkpoint
from cs336_basics.optim.adamw import AdamW
from cs336_basics.optim.functions import gradient_clipping, lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


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


def train(args: argparse.Namespace):
    seed=int(args.seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)

    experiment_name = f"{args.experiment_base}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    save_path = os.path.join(args.save_root, experiment_name)
    os.makedirs(save_path, exist_ok=True)

    log_path = os.path.join(save_path, 'log.txt')
    logger = get_logger(log_path)
    logger.info(f'The logging will save at {log_path}')

    logger.info("Args:\n%s", pprint.pformat(vars(args), sort_dicts=False))

    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'), flush_secs=60)

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
    model.train()

    optimizer = AdamW(
        params=model.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta_low), float(args.beta_high)),
        eps=float(args.optim_eps),
        weight_decay=float(args.weight_decay),
    )

    train_dataset = StreamingDataset(
        data_dir=str(args.train_data_dir),
        batch_size=int(args.batch_size),
        context_length=int(args.context_length),
        max_samples_per_iter=int(args.max_train_samples_per_iter),
        seed=int(args.seed),
    )
    val_dataset = StreamingDataset(
        data_dir=str(args.val_data_dir),
        batch_size=int(args.batch_size),
        context_length=int(args.context_length),
        max_samples_per_iter=int(args.max_val_samples_per_iter),
        seed=int(args.seed),
    )

    num_batches_per_epoch = train_dataset.num_batches_per_epoch
    total_steps = int(args.max_steps)

    start_steps = 1
    if args.resume:
        logger.info(f"Loading checkpoint from {args.checkpoint_file}")
        start_steps = load_checkpoint(args.checkpoint_file, model, optimizer)
        logger.info(f"Resumed trainer from iteration {start_steps}")

    start_epoch = (start_steps - 1) // num_batches_per_epoch
    skip_batches = (start_steps - 1) % num_batches_per_epoch
    train_dataset.set_epoch(start_epoch)
    data_iter = iter(train_dataset)
    for _ in range(skip_batches):
        next(data_iter)

    pbar = tqdm(
        range(start_steps, total_steps + 1),
        desc="Train",
        unit="step",
        total=total_steps,          # 总步数
        initial=start_steps - 1,    # 已完成的 step 数
    )
    last_val_loss = None
    for step in pbar:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataset)
            x, y = next(data_iter)

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        logits = model(x)
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()

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

        train_loss = loss.item()
        if step % int(args.write_steps) == 0:
            writer.add_scalar('metrics/train_loss', train_loss, step)
            writer.add_scalar('metrics/lr', lr, step)

        if step % int(args.val_steps) == 0:
            model.eval()

            val_loss = 0.0
            val_step = 0
            with torch.no_grad():
                for x, y in val_dataset:
                    x = torch.from_numpy(x).to(device)
                    y = torch.from_numpy(y).to(device)
                    logits = model(x)
                    logits = logits.reshape(-1, logits.size(-1))
                    y = y.reshape(-1)
                    loss = cross_entropy_loss(logits, y)

                    val_step += 1
                    val_loss += (loss.item() - val_loss) / val_step

                val_dataset.set_epoch(0)

            writer.add_scalar('metrics/val_loss', val_loss, step)

            last_val_loss = val_loss

            model.train()

        if step % int(args.checkpoint_save_steps) == 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_step_{step:08}.pt")
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            logger.info(f"Saved checkpoint step {step} to {checkpoint_path}")

        postfix = {"train_loss": train_loss, "lr": lr}
        if last_val_loss is not None:
            postfix["val_loss"] = last_val_loss
        pbar.set_postfix(**postfix)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cs336 assignment1 basics training')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--experiment_base', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--max_steps', type=int, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--context_length', type=int, required=True)
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
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--beta_low', type=float, required=False, default=0.90)
    parser.add_argument('--beta_high', type=float, required=False, default=0.99)
    parser.add_argument('--optim_eps', type=float, required=False, default=1e-8)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-2)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--checkpoint_file', type=str, required=False)
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--val_data_dir', type=str, required=True)
    parser.add_argument('--max_train_samples_per_iter', type=int, required=True)
    parser.add_argument('--max_val_samples_per_iter', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_l2_norm', type=float, required=False, default=1e2)
    parser.add_argument('--cosine_min', type=float, required=False, default=1e-6)
    parser.add_argument('--warmup_iters', type=int, required=True)
    parser.add_argument('--cosine_iters', type=int, required=True)
    parser.add_argument('--write_steps', type=int, required=True)
    parser.add_argument('--val_steps', type=int, required=True)
    parser.add_argument('--checkpoint_save_steps', type=int, required=True)
    args = parser.parse_args()

    train(args)
