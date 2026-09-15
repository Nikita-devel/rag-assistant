#!/usr/bin/env bash
# Build the corpus and the index on first boot, then hand over to the server.
#
# Neither is baked into the image: the corpus is reproducible from open data,
# and the index depends on which embedding model is configured. Both live in a
# volume, so rebuilding the image does not throw them away, and switching
# EMBEDDING_MODEL is a matter of deleting the volume.
set -euo pipefail

DOCUMENTS_PATH="${DOCUMENTS_PATH:-/data/documents}"
CHROMA_PATH="${CHROMA_PATH:-/data/chroma_db}"
mkdir -p "$DOCUMENTS_PATH" "$CHROMA_PATH"

if ! compgen -G "$DOCUMENTS_PATH/*.md" > /dev/null; then
  echo "[entrypoint] no corpus found — building it (needs network, ~2 min)"
  python scripts/build_corpus.py
fi

# "Does the directory exist and contain something" is NOT a test for a usable
# index: a failed ingestion leaves an empty Chroma directory behind, and that
# earlier version then skipped indexing and served 0 chunks quite happily. Ask
# the collection how many chunks it actually holds.
indexed="$(python - <<'PY'
try:
    from app.ingestion import get_collection
    print(get_collection().count())
except Exception:
    print(0)
PY
)"

if [ "${indexed:-0}" -lt 1 ]; then
  echo "[entrypoint] index is empty — ingesting (~4 min on CPU, downloads the embedder)"
  python -m app.ingestion --reset
  indexed="$(python -c 'from app.ingestion import get_collection; print(get_collection().count())')"
fi

if [ "${indexed:-0}" -lt 1 ]; then
  echo "[entrypoint] FATAL: the index is still empty after ingestion." >&2
  echo "[entrypoint] corpus directory: $DOCUMENTS_PATH" >&2
  ls -la "$DOCUMENTS_PATH" >&2 || true
  exit 1
fi

echo "[entrypoint] $indexed chunks indexed — starting: $*"
exec "$@"
