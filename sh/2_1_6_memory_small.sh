#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

RUN_ID="$(date +%Y_%m_%d_%H_%M_%S)_$$"
RUN_ROOT="${PROJECT_DIR}/outputs/2_1_6_memory_small/${RUN_ID}"
mkdir -p "$RUN_ROOT"

for context_length in 128 256 512 1024; do
  for precision in fp32 bf16; do
    precision_args=()
    if [[ "$precision" == bf16 ]]; then
      precision_args+=(--bf16)
    fi

    for mode in inference total; do
      uv run python -m cs336_systems.benchmark \
        --save_root "${RUN_ROOT}/context_${context_length}_${precision}_${mode}" \
        --mode "$mode" \
        --seed 0 \
        --device cuda \
        --batch_size 2 \
        --context_length "$context_length" \
        --warm_steps 5 \
        --mark_steps 1 \
        --num_layers 12 \
        --d_model 768 \
        --d_ff 2048 \
        --num_heads 12 \
        --memory_profile \
        "${precision_args[@]}"
    done
  done

  # Nsight data for (f): FP32 training, without the CUDA memory cache.
  nsys_dir="${RUN_ROOT}/context_${context_length}_fp32_nsys"
  mkdir -p "$nsys_dir"
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 uv run -- nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --pytorch=autograd-shapes-nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=nvtx \
    --nvtx-capture=measure \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    --capture-range-end=stop \
    --kill=none \
    --export=sqlite \
    --output="${nsys_dir}/profile" \
    python -m cs336_systems.benchmark \
      --save_root "$nsys_dir" \
      --mode total \
      --seed 0 \
      --device cuda \
      --batch_size 2 \
      --context_length "$context_length" \
      --warm_steps 5 \
      --mark_steps 1 \
      --num_layers 12 \
      --d_model 768 \
      --d_ff 2048 \
      --num_heads 12 \
      --nvtx \
    2>&1 | tee "${nsys_dir}/nsys.log"
done

printf 'Results saved to %s\n' "$RUN_ROOT"
