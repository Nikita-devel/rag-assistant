"""Embedding backend — shared by ingestion and retrieval.

Why this module exists (it is not in the original spec): documents and queries
MUST be embedded by the same model with the same prefixes. Duplicating that
logic in ingestion.py and retriever.py is the classic way to silently break a
RAG system. One module, one source of truth.

Model: intfloat/multilingual-e5-small
  - multilingual (the corpus is French, the demo answers FR + EN)
  - 384 dims, ~120 MB, CPU-friendly
  - REQUIRES asymmetric prefixes: "passage: " for documents, "query: " for
    questions. Omitting them costs ~10 points of retrieval accuracy.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:                       # heavy import, loaded only when used
    from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: "SentenceTransformer | None" = None
_lock = threading.Lock()


def get_model() -> "SentenceTransformer":
    """Lazily load the model once per process (thread-safe).

    The torch import costs seconds and ~300 MB; keeping it inside this function
    means a module that only needs the pure helpers does not pay for it.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", settings.embedding_model)
                _model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device,
                )
    return _model


def _is_e5() -> bool:
    return "e5" in settings.embedding_model.lower()


def embed_passages(texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks (indexing side)."""
    texts = list(texts)
    if not texts:
        return []
    if _is_e5():
        texts = [f"passage: {t}" for t in texts]
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine == dot product, and distances stay in [0, 2]
        show_progress_bar=len(texts) > 64,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user question (search side)."""
    payload = f"query: {text}" if _is_e5() else text
    vector = get_model().encode(payload, normalize_embeddings=True)
    return vector.tolist()
