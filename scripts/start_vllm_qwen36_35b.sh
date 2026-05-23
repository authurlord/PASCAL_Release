#!/usr/bin/env bash
# Start vLLM for Qwen3.6-35B-A3B-FP8 at 128K context with MTP=3 speculation.
#
# Requirements: vLLM >= 0.19.1, two GPUs with >=40 GB total VRAM.
#
# Override defaults via env vars:
#   PASCAL_MODEL_PATH       Path or HF id of the model (default: Qwen/Qwen3.6-35B-A3B-FP8)
#   PASCAL_GPUS             CUDA device IDs (default: 0,1)
#   PASCAL_PORT             vLLM HTTP port (default: 8000)
#   PASCAL_MAX_MODEL_LEN    Max context length (default: 131072)
#   PASCAL_MAX_NUM_SEQS     Max concurrent sequences (default: 48)
#   PASCAL_GPU_MEM_UTIL     gpu-memory-utilization (default: 0.92)
#   PASCAL_NUM_SPEC_TOKENS  MTP speculative tokens (default: 3; set 0 to disable MTP)

set -euo pipefail

MODEL="${PASCAL_MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"
GPUS="${PASCAL_GPUS:-0,1}"
PORT="${PASCAL_PORT:-8000}"
MAX_MODEL_LEN="${PASCAL_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${PASCAL_MAX_NUM_SEQS:-48}"
GPU_MEM_UTIL="${PASCAL_GPU_MEM_UTIL:-0.92}"
NUM_SPEC_TOKENS="${PASCAL_NUM_SPEC_TOKENS:-3}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_NO_USAGE_STATS=1

echo "[start_vllm_qwen36_35b] model=$MODEL gpus=$GPUS port=$PORT max_len=$MAX_MODEL_LEN mtp=$NUM_SPEC_TOKENS"

if command -v ss >/dev/null && ss -tln | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "[FATAL] port $PORT already in use." >&2
  exit 1
fi

SPEC_FLAG=()
if [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
  SPEC_FLAG=(--speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}")
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3.6-35b \
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
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${SPEC_FLAG[@]}"
