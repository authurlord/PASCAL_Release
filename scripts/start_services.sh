#!/usr/bin/env bash
# Start the three PASCAL microservices: system_agent (6000),
# user_simulator (6001), db_environment (6002).
#
# Reads PASCAL_* env vars to control the agent's behavior. Override the
# defaults via the calling shell (see README.md, "Reproduce the anchor").
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${SERVICE_HOST:-127.0.0.1}"

# Make sure no stray uvicorn from a previous run is hogging the ports.
pkill -f "uvicorn (system_agent|user_simulator|db_environment)" 2>/dev/null || true
sleep 1

"$PYTHON_BIN" -m uvicorn system_agent.server:app   --host "$HOST" --port 6000 --log-level warning &
"$PYTHON_BIN" -m uvicorn user_simulator.server:app --host "$HOST" --port 6001 --log-level warning &
"$PYTHON_BIN" -m uvicorn db_environment.server:app --host "$HOST" --port 6002 --log-level warning &

for i in $(seq 1 30); do
  if curl --noproxy '*' -s "http://127.0.0.1:6000/health" >/dev/null 2>&1 && \
     curl --noproxy '*' -s "http://127.0.0.1:6001/health" >/dev/null 2>&1 && \
     curl --noproxy '*' -s "http://127.0.0.1:6002/health" >/dev/null 2>&1; then
    echo "ALL_SERVICES_READY (ports 6000, 6001, 6002)"
    exit 0
  fi
  sleep 1
done
echo "SERVICES_FAILED"
exit 1
