# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Layer order is deliberate: the things that change least often go first, so a
# code edit rebuilds in seconds instead of re-downloading 200 MB of torch.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model weights live in a named volume, not in the image: ~600 MB that
    # would otherwise be re-baked into every rebuild.
    HF_HOME=/models \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# curl for the container healthcheck; nothing else is needed at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# torch FIRST, from the CPU index. Installed from PyPI it pulls the CUDA build
# plus ~3 GB of nvidia-* wheels that are dead weight on a CPU host — the
# difference between a 1.5 GB and a 6 GB image.
RUN pip install --no-cache-dir torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Inside the container the app must listen on all interfaces; the port is still
# published only to localhost on the host (see compose).
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8077 \
    CHROMA_PATH=/data/chroma_db \
    DOCUMENTS_PATH=/data/documents

EXPOSE 8077

# start-period is generous on purpose: the first boot downloads the embedder and
# the reranker, and an impatient healthcheck would kill the container mid-pull.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8077/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]
