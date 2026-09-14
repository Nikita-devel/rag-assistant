#!/usr/bin/env bash
# Start the API on the port this project owns.
#
# The port lives in .env (APP_PORT) so the dev run, the Docker container and the
# Cloudflare tunnel all read the same number from one place. Picking it on the
# command line each time is how you end up tunnelling to the wrong service.
#
#   bash scripts/serve.sh            production-ish: no reload, warm start
#   bash scripts/serve.sh --dev      reload on file changes
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="$(grep -E '^APP_PORT=' .env 2>/dev/null | cut -d= -f2 || true)"
PORT="${PORT:-8077}"
HOST="$(grep -E '^APP_HOST=' .env 2>/dev/null | cut -d= -f2 || true)"
HOST="${HOST:-127.0.0.1}"

if ss -ltn "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  echo "Port ${PORT} is already in use:"
  ss -ltnp "sport = :${PORT}" 2>/dev/null | tail -n +2
  echo
  echo "Stop it, or change APP_PORT in .env."
  exit 1
fi

# --reload restarts the process on every file change, and each restart reloads
# the embedder, the reranker and the BM25 index — about 10 seconds. Fine while
# editing, unacceptable while showing the demo to someone.
if [[ "${1:-}" == "--dev" ]]; then
  exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
fi
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
