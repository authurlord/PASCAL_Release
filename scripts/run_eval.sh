#!/usr/bin/env bash
# Single-entry runner for the PASCAL anchor and the official ReACT baseline.
#
# Usage:
#   bash scripts/run_eval.sh anchor [extra args …]   # PASCAL anchor
#   bash scripts/run_eval.sh react  [extra args …]   # official ReACT baseline
#
# Default args evaluate the full BIRD-Interact-lite split at concurrency 48
# and write to results/eval_<mode>.json. Pass extra flags after the mode to
# override (e.g. --data data/hard_60.jsonl --limit 12 --concurrency 8).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

MODE="${1:-anchor}"
shift || true

DATA="${DATA:-data/bird-interact-lite-hf-meta/bird_interact_data.jsonl}"
OUTPUT="${OUTPUT:-results/eval_${MODE}.json}"
CONCURRENCY="${CONCURRENCY:-48}"

# Common env
export LITELLM_API_BASE="${LITELLM_API_BASE:-http://127.0.0.1:8000/v1}"
export LITELLM_API_KEY="${LITELLM_API_KEY:-dummy-vllm-key}"
export SYSTEM_AGENT_MODEL="${SYSTEM_AGENT_MODEL:-openai/qwen3.6-35b}"
export USER_SIM_MODEL="${USER_SIM_MODEL:-gemini/gemini-2.5-flash-lite}"
export SYSTEM_AGENT_PORT=6000
export USER_SIM_PORT=6001
export DB_ENV_PORT=6002

case "$MODE" in
  anchor)
    # PASCAL anchor — applies to all benchmarks (BIRD-Interact lite /
    # full, PRACTIQ).  PASCAL prompt + streamlined tools
    # + schema pre-injection.  The agent retrieves KB on demand via
    # `get_all_external_knowledge_names` + `get_knowledge_definition`;
    # no pre-injection.  Oracle row-cell value-diff feedback disabled
    # for clean audit (`PASCAL_NO_VALUE_DIFF=1`).
    export PASCAL_NO_VALUE_DIFF=1
    unset PASCAL_NO_PROTOCOL PASCAL_KB_INJECTION || true
    ;;
  react)
    # Official ReACT baseline: minimal prompt, original 9-tool surface
    # minus KB tools. Matches the upstream BIRD-Interact agentic scaffold.
    export PASCAL_NO_PROTOCOL=1
    unset PASCAL_NO_VALUE_DIFF PASCAL_KB_INJECTION || true
    ;;
  *)
    echo "Usage: bash scripts/run_eval.sh {anchor|react} [extra args ...]" >&2
    exit 1
    ;;
esac

# Ensure services are up (idempotent — start_services.sh kills stale ones).
bash scripts/start_services.sh

mkdir -p results
python -m orchestrator.runner \
  --mode a-interact \
  --data "$DATA" \
  --output "$OUTPUT" \
  --concurrency "$CONCURRENCY" "$@"
