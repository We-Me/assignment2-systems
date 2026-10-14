#!/usr/bin/env bash
set -euo pipefail

# 脚本位于 <项目根>/sh/，项目根目录是上一级
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

RUN_ID="$(date +%Y_%m_%d_%H_%M_%S)_$$"
RUN_ROOT="${PROJECT_DIR}/outputs/2_1_3_benchmark_standard/${RUN_ID}"
mkdir -p "$RUN_ROOT"

for mode in inference forward backward total; do
  for warm in 5 0 1 2; do
    # small
    uv run python cs336_systems/benchmark.py \
      --save_root "${RUN_ROOT}/small_${mode}_warm_${warm}" \
      --mode $mode \
      --warm_steps $warm \
      --batch_size 4 \
      --context_length 512 \
      --mark_steps 10 \
      --num_layers 12 \
      --d_model 768 \
      --d_ff 2048 \
      --num_heads 12 \

  done
done

for mode in inference forward backward total; do
  for warm in 5 0 1 2; do
    # medium
    uv run python cs336_systems/benchmark.py \
      --save_root "${RUN_ROOT}/medium_${mode}_warm_${warm}" \
      --mode $mode \
      --warm_steps $warm \
      --batch_size 4 \
      --context_length 512 \
      --mark_steps 10 \
      --num_layers 24 \
      --d_model 1024 \
      --d_ff 2752 \
      --num_heads 16 \

  done
done

for mode in inference forward backward total; do
  for warm in 5 0 1 2; do
    # large
    uv run python cs336_systems/benchmark.py \
      --save_root "${RUN_ROOT}/large_${mode}_warm_${warm}" \
      --mode $mode \
      --warm_steps $warm \
      --batch_size 4 \
      --context_length 512 \
      --mark_steps 10 \
      --num_layers 36 \
      --d_model 1280 \
      --d_ff 3392 \
      --num_heads 20 \

  done
done

for mode in inference forward backward total; do
  for warm in 5 0 1 2; do
    # xl
    uv run python cs336_systems/benchmark.py \
      --save_root "${RUN_ROOT}/xl_${mode}_warm_${warm}" \
      --mode $mode \
      --warm_steps $warm \
      --batch_size 4 \
      --context_length 512 \
      --mark_steps 10 \
      --num_layers 32 \
      --d_model 2560 \
      --d_ff 6848 \
      --num_heads 32 \

  done
done

printf 'Results saved to %s\n' "$RUN_ROOT"
