#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

command -v uv >/dev/null 2>&1 || { echo "uv is required in WSL." >&2; exit 1; }
command -v nsys >/dev/null 2>&1 || { echo "nsys is required in WSL." >&2; exit 1; }

BATCH_SIZE=2
CONTEXT_LENGTH=256
WARM_STEPS=5
MARK_STEPS=10
RUN_ID="$(date +%Y_%m_%d_%H_%M_%S)_$$"
RUN_ROOT="${PROJECT_DIR}/outputs/2_1_4_nsys_reduced/${RUN_ID}"
mkdir -p "$RUN_ROOT"
nsys --version > "${RUN_ROOT}/nsys_version.txt"

profile_model() {
  local model_size="$1"
  local num_layers="$2"
  local d_model="$3"
  local d_ff="$4"
  local num_heads="$5"
  local mode case_dir

  for mode in inference forward backward total; do
    case_dir="${RUN_ROOT}/${model_size}_${mode}"
    mkdir -p "$case_dir"
    printf 'Profiling %s / %s -> %s\n' "$model_size" "$mode" "$case_dir"

    uv run -- nsys profile \
      --trace=cuda,nvtx \
      --pytorch=autograd-nvtx \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=nvtx \
      --nvtx-capture=measure \
      --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
      --capture-range-end=stop \
      --kill=none \
      --output="${case_dir}/profile" \
      python cs336_systems/benchmark.py \
        --save_root "$case_dir" \
        --mode "$mode" \
        --warm_steps "$WARM_STEPS" \
        --mark_steps "$MARK_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --context_length "$CONTEXT_LENGTH" \
        --num_layers "$num_layers" \
        --d_model "$d_model" \
        --d_ff "$d_ff" \
        --num_heads "$num_heads" \
        --nvtx \
      2>&1 | tee "${case_dir}/nsys.log"
  done
}

profile_model small 12 768 2048 12
profile_model medium 24 1024 2752 16
