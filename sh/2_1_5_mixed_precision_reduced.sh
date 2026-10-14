#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

RUN_ID="$(date +%Y_%m_%d_%H_%M_%S)_$$"
RUN_ROOT="${PROJECT_DIR}/outputs/2_1_5_mixed_precision_reduced/${RUN_ID}"

mkdir -p "$RUN_ROOT"

for config in \
  "small 12 768 2048 12" \
  "medium 24 1024 2752 16"; do
  read -r model_size num_layers d_model d_ff num_heads <<< "$config"
  for precision in fp32 bf16; do
    precision_args=()
    if [[ "$precision" == bf16 ]]; then
      precision_args+=(--bf16)
    fi

    uv run python -m cs336_systems.benchmark \
      --save_root "${RUN_ROOT}/${model_size}_${precision}" \
      --mode backward \
      --seed 0 \
      --device cuda \
      --batch_size 2 \
      --context_length 256 \
      --warm_steps 5 \
      --mark_steps 10 \
      --num_layers "$num_layers" \
      --d_model "$d_model" \
      --d_ff "$d_ff" \
      --num_heads "$num_heads" \
      "${precision_args[@]}"
  done
done

printf 'Results saved to %s\n' "$RUN_ROOT"
