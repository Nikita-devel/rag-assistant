# syntax=docker/dockerfile:1
# Image for Google Cloud Run. It differs from the root Dockerfile in four ways,
# each forced by how Cloud Run actually works.
#
# 1. EVERYTHING IS BAKED IN — corpus, index, embedder, reranker. A Cloud Run
#    container has no persistent disk and is destroyed when idle: every cold
#    start begins from the image. Building the index at boot would mean a
#    ten-minute wait for whoever arrives first, which on a demo link sent to a
#    prospect is the same as being offline. The ten minutes are paid once, at
#    build time.
# 2. THE PORT COMES FROM $PORT. Cloud Run picks it and tells the container;
#    a hardcoded port means the health check never passes and the revision is
#    rolled back with an unhelpful message.
# 3. PROXY HEADERS ARE TRUSTED. Every request arrives via Google's front end, so
#    without --proxy-headers the app sees the proxy's address as the client and
#    slowapi's per-IP rate limit becomes one shared bucket for the whole
#    internet — one curious visitor then locks out everyone else.
# 4. IT RUNS AS A NON-ROOT USER. Not required by Cloud Run, but this container
#    is reachable by anyone with the URL and nothing in it needs root.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False

# torch FIRST, from the CPU index: from PyPI it drags in the CUDA build and
# ~3 GB of nvidia-* wheels that do nothing here and would slow every cold start.
RUN pip install --no-cache-dir torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The directories are created AND chowned here, while we are still root.
# WORKDIR creates a missing directory as root even when a USER is already in
# effect, so `USER app` followed by `WORKDIR /home/app/srv` yields a root-owned
# working directory the app cannot write to — and the corpus build then dies
# with "Permission denied: /home/app/srv/data".
RUN useradd -m -u 1000 app \
 && mkdir -p /home/app/srv/data/documents /home/app/srv/chroma_db \
             /home/app/.cache/huggingface \
 && chown -R app:app /home/app

USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    HF_HOME=/home/app/.cache/huggingface
WORKDIR /home/app/srv

COPY --chown=app:app app/ ./app/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app frontend/ ./frontend/

# Settings normally supplied by .env, which is never committed and never
# shipped. MIN_RERANK_SCORE above all: it is the calibrated refusal threshold,
# and leaving it at the code default of 0.0 makes the assistant refuse
# perfectly valid questions — a demo that looks broken for a reason nobody can
# see. GROQ_API_KEY is NOT here; it is injected from Secret Manager at deploy.
ENV CHROMA_PATH=/home/app/srv/chroma_db \
    DOCUMENTS_PATH=/home/app/srv/data/documents \
    EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    EMBEDDING_DEVICE=cpu \
    COLLECTION_NAME=sme_hr_corpus \
    CHUNK_SIZE=1400 \
    CHUNK_OVERLAP=200 \
    TOP_K=5 \
    MIN_SIMILARITY=0.72 \
    MIN_RERANK_SCORE=-3.2 \
    MAX_QUESTION_LENGTH=500 \
    LLM_PROVIDER=groq \
    GROQ_BASE_URL=https://api.groq.com/openai/v1 \
    GROQ_MODEL=openai/gpt-oss-120b \
    RATE_LIMIT=6/minute \
    PORT=8080

# Corpus and index, built into the image. The raw DILA dumps are deleted in the
# SAME layer — they are ~90 MB of JSON that nothing reads once the markdown
# exists, and a file removed in a later layer still ships inside the image.
RUN python scripts/build_corpus.py \
 && python -m app.ingestion --reset \
 && python -c "from app.ingestion import get_collection; n=get_collection().count(); print(f'indexed {n} chunks'); assert n > 2000, f'index too small: {n}'" \
 && rm -f /home/app/srv/data/*.json

# The reranker is lazy and would otherwise be fetched on the first question —
# ~470 MB while a visitor waits. Pull it now so it ships in the image.
RUN python -c "from app.retriever import get_reranker; get_reranker().predict([('question test', 'texte de test')]); print('reranker cached')"

EXPOSE 8080

# Shell form on purpose: $PORT has to be expanded at run time, and exec keeps
# uvicorn as PID 1 so Cloud Run's shutdown signal reaches it.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} \
      --proxy-headers --forwarded-allow-ips='*'
