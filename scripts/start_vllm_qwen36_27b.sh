#!/usr/bin/env bash
# Start vLLM for Qwen3.6-27B-FP8 (dense) at 128K context — smaller / faster
# alternative to the 35B-A3B anchor.
#
# Requirements: vLLM >= 0.19.1, two GPUs with >=48 GB total VRAM.

set -euo pipefail

MODEL="${PASCAL_MODEL_PATH:-Qwen/Qwen3.6-27B-FP8}"
GPUS="${PASCAL_GPUS:-0,1}"
PORT="${PASCAL_PORT:-8000}"
MAX_MODEL_LEN="${PASCAL_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${PASCAL_MAX_NUM_SEQS:-48}"
GPU_MEM_UTIL="${PASCAL_GPU_MEM_UTIL:-0.92}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_NO_USAGE_STATS=1

echo "[start_vllm_qwen36_27b] model=$MODEL gpus=$GPUS port=$PORT max_len=$MAX_MODEL_LEN"

if command -v ss >/dev/null && ss -tln | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "[FATAL] port $PORT already in use." >&2
  exit 1
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3.6-27b \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 32768 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3
