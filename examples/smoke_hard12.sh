#!/usr/bin/env bash
# End-to-end smoke test: PASCAL anchor on the first 12 tasks of the
# BIRD-Interact-lite hard_60 subset. ~10-20 minutes on a single 35B vLLM.
#
# Prerequisites
# -------------
# * vLLM listening on http://127.0.0.1:8000/v1 with qwen3.6-35b loaded
#   (run scripts/start_vllm_qwen36_35b.sh in a separate shell).
# * PostgreSQL running with the lite databases restored (dumps/README.md).
# * GOOGLE_API_KEY exported for the Gemini user simulator
#   (see docs/MODEL_CARDS.md).
# * The hard_60 JSONL with ground truth merged in.  The committed
#   data/hard_60.jsonl ships WITHOUT sol_sql/test_cases — you must
#   merge the GT JSONL obtained from the upstream maintainers using
#   data/combine_public_with_gt.py before running this script.  See
#   data/README.md for the email + merge recipe.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export DATA="${DATA:-data/hard_60.jsonl}"
export OUTPUT="${OUTPUT:-results/smoke_hard12_anchor.json}"
export CONCURRENCY="${CONCURRENCY:-4}"

if ! python -c "import json,sys; r=json.loads(open('$DATA').readline()); sys.exit(0 if r.get('sol_sql') else 1)" 2>/dev/null; then
  echo "[ABORT] $DATA does not have ground truth merged in." >&2
  echo "        Run data/combine_public_with_gt.py first — see data/README.md." >&2
  exit 1
fi

bash scripts/run_eval.sh anchor --limit 12
echo
echo "Smoke test complete. Inspect: $OUTPUT"
echo "Expect P1 in the 30-40% range on hard_60 with the 35B anchor."
