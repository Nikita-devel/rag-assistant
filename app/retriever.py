"""Retrieval: hybrid (dense + BM25) fusion with optional cross-encoder reranking.

WHY NOT PLAIN DENSE SEARCH
--------------------------
The first smoke test on this corpus gave a separation gap of +0.019 between the
worst in-corpus question and the best out-of-corpus question, and several
in-corpus questions returned the wrong article entirely (e.g. "à partir de
combien de salariés faut-il un CSE ?" did not retrieve L2311-2, which is in the
index). Two independent problems:

1. RANKING. Legal text is dense in near-identical boilerplate, so a bi-encoder
   maps most of it into a narrow cone. Queries also carry exact legal terms
   ("astreinte", "habillage", "CSE") that lexical search matches precisely and
   embeddings blur. BM25 and dense retrieval fail on *different* queries, so
   fusing them recovers most of the misses.

2. CONFIDENCE. Bi-encoder cosine is a similarity, not a calibrated probability.
   On this corpus everything lands in 0.77-0.90, so no threshold can separate
   "found it" from "nothing relevant". A cross-encoder reads the query and the
   chunk together and outputs a genuinely separated relevance score — that is
   what the guardrail should threshold, not cosine.

This module is deliberately dataset-agnostic: it knows about chunks, metadata
and scores, and nothing about French labour law. It is the piece to reuse on
client projects.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from app.config import settings
from app.embeddings import embed_query
from app.ingestion import get_collection
from app.query_expansion import prepare

logger = logging.getLogger(__name__)

Strategy = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float                      # final score, meaning depends on strategy
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        meta = self.metadata
        article = meta.get("article") or ""
        source = meta.get("source", "").replace(".md", "")
        return f"{article} ({source})" if article else source


@dataclass
class RetrievalResult:
    hits: list[Hit]
    confident: bool                   # False -> the generator must refuse to answer
    strategy: str
    top_score: float = 0.0
    reason: str = ""
    language: str = "fr"          # detected question language; the generator
                                  # answers in it


# --------------------------------------------------------------------------- #
# Lexical index (BM25)
# --------------------------------------------------------------------------- #
# Minimal French + English stopword list. Deliberately short: aggressive
# stopword removal hurts legal queries, where words like "sans", "moins" and
# "au-delà" carry real meaning.
STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou",
    "a", "à", "au", "aux", "en", "dans", "par", "pour", "sur", "est", "sont",
    "ce", "cette", "ces", "qui", "que", "quel", "quelle", "quels", "quelles",
    "il", "elle", "on", "se", "s", "son", "sa", "ses", "leur", "leurs",
    "the", "a", "an", "of", "to", "in", "is", "are", "for", "on", "what", "how",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Accent-folded lowercase tokens. Keeps 'L3141-3' as one token."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in TOKEN_RE.findall(text) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Okapi BM25. ~60 lines instead of a dependency, and it lets us index the
    article number alongside the body so 'L3141-3' as a query matches directly.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d) for d in documents]
        self.n = len(self.corpus)
        self.lengths = [len(d) for d in self.corpus]
        self.avg_len = sum(self.lengths) / self.n if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.corpus]

        df: Counter = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        # BM25+ style idf floor: never let a common term go negative.
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.postings: dict[str, list[int]] = {}
        for i, doc in enumerate(self.corpus):
            for term in set(doc):
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        candidates: set[int] = set()
        for term in terms:
            candidates.update(self.postings.get(term, ()))
        if not candidates:
            return []

        scores: list[tuple[int, float]] = []
        for i in sorted(candidates):
            tf_i, len_i, total = self.tf[i], self.lengths[i], 0.0
            for term in terms:
                freq = tf_i.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * len_i / self.avg_len)
                total += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            if total > 0:
                scores.append((i, total))
        # Score first, document index second: float ties must never be broken by
        # iteration order.
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]


# --------------------------------------------------------------------------- #
# Reranker
# --------------------------------------------------------------------------- #
_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    """Cross-encoder, loaded lazily. Multilingual and small enough for CPU:
    ~120 M params, ~0.4 s for 20 candidates on a modern laptop core.
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker %s", settings.reranker_model)
                _reranker = CrossEncoder(
                    settings.reranker_model,
                    max_length=512,
                    device=settings.embedding_device,
                )
    return _reranker


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
@dataclass
class Retriever:
    strategy: Strategy = "hybrid_rerank"
    _collection: object = field(default=None, repr=False)
    _bm25: BM25 | None = field(default=None, repr=False)
    _docs: list[str] = field(default_factory=list, repr=False)
    _metas: list[dict] = field(default_factory=list, repr=False)
    _ids: list[str] = field(default_factory=list, repr=False)
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._collection = get_collection()
        self._load_lexical_index()

    def _load_lexical_index(self) -> None:
        """Pull every chunk once and build the BM25 index in memory.

        2.6k chunks is ~2 MB of text — trivial. For a corpus 100x larger this
        would move to a real search engine, but premature here.
        """
        data = self._collection.get(include=["documents", "metadatas"])

        # DETERMINISM. Chroma.get() does not promise an order, and the order it
        # happens to return changes between processes. That order becomes the
        # BM25 document index, which becomes the tie-break order everywhere
        # downstream — so two runs over an identical index produced different
        # top-1 chunks and different guardrail scores. Measured: one
        # out-of-scope question scored -3.862 in one run and -2.972 in the next,
        # and the calibrated threshold moved with it (-3.2 vs -2.8).
        #
        # Sorting by chunk id pins the index. Chunk ids are content hashes, so
        # this order is stable across re-ingestion too, not just across runs.
        order = sorted(range(len(data["ids"])), key=lambda i: data["ids"][i])
        self._ids = [data["ids"][i] for i in order]
        self._docs = [data["documents"][i] for i in order]
        self._metas = [data["metadatas"][i] for i in order]
        self._index = {uid: i for i, uid in enumerate(self._ids)}

        # Index "<article> <section> <body>" so a query naming an article number
        # or a chapter matches it lexically.
        indexed = [
            f"{m.get('article', '')} {m.get('section', '')} {d}"
            for m, d in zip(self._metas, self._docs)
        ]
        self._bm25 = BM25(indexed)
        logger.info("Lexical index: %d chunks, %d terms",
                    len(self._docs), len(self._bm25.idf))

    # -- individual strategies ---------------------------------------------- #
    def _dense(self, question: str, top_k: int) -> list[tuple[int, float]]:
        res = self._collection.query(
            query_embeddings=[embed_query(question)],
            n_results=top_k,
            include=["distances"],
        )
        out = []
        for uid, dist in zip(res["ids"][0], res["distances"][0]):
            i = self._index.get(uid)
            if i is not None:
                out.append((i, 1.0 - dist))
        return out

    def _lexical(self, question: str, top_k: int) -> list[tuple[int, float]]:
        return self._bm25.search(question, top_k)

    @staticmethod
    def _rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion. Rank-based, so it needs no score calibration
        between BM25 (unbounded) and cosine (0-1) — which is exactly why it is
        the right fusion here.
        """
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        # Tie-break on the document index so equal fused scores order stably.
        return sorted(fused.items(), key=lambda x: (-x[1], x[0]))

    # -- public API ---------------------------------------------------------- #
    def search(
        self,
        question: str,
        top_k: int | None = None,
        strategy: Strategy | None = None,
        candidates: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.top_k
        strategy = strategy or self.strategy
        candidates = candidates or settings.rerank_candidates

        question = question.strip()
        if not question:
            return RetrievalResult([], False, strategy, reason="empty query")

        # Two forms of the same question: a clean sentence for the embedder,
        # a synonym-padded one for BM25. See app/query_expansion.py for why.
        semantic_q, lexical_q, language = prepare(question)

        dense_scores: dict[int, float] = {}
        bm25_scores: dict[int, float] = {}

        if strategy == "dense":
            ranked = self._dense(semantic_q, top_k)
            dense_scores = dict(ranked)
        elif strategy == "bm25":
            ranked = self._lexical(lexical_q, top_k)
            bm25_scores = dict(ranked)
        else:
            dense = self._dense(semantic_q, candidates)
            lexical = self._lexical(lexical_q, candidates)
            dense_scores, bm25_scores = dict(dense), dict(lexical)
            ranked = self._rrf([dense, lexical])[:candidates]

        hits = [
            Hit(
                text=self._docs[i],
                # chunk_uid rides along so downstream sorts have a stable
                # tie-breaker that does not depend on load order.
                metadata={**self._metas[i], "chunk_uid": self._ids[i]},
                score=score,
                dense_score=dense_scores.get(i),
                bm25_score=bm25_scores.get(i),
            )
            for i, score in ranked
        ]

        if strategy == "hybrid_rerank" and hits:
            # The cross-encoder sees the real question (bridged to French if it
            # arrived in English), never the padded lexical form — it reads the
            # sentence, and padding it with synonyms only adds noise.
            pairs = [(semantic_q, h.text) for h in hits]
            scores = get_reranker().predict(pairs, show_progress_bar=False)
            for hit, score in zip(hits, scores):
                hit.rerank_score = float(score)

            # MEASURED, not assumed: an earlier version fused the reranker's
            # order back with the fusion order (RRF again), on the theory that
            # the cross-encoder was unfairly demoting short definitional
            # articles. It scored WORSE — R@1 0.54 -> 0.46, R@5 0.85 -> 0.77.
            # The reranker really is the best single ordering signal here; the
            # one question it got wrong cost less than the blending cost
            # everywhere else. Left as a comment so the next person does not
            # retry it.
            for hit, score in zip(hits, scores):
                hit.score = float(score)
            hits.sort(key=lambda h: (-h.score, h.metadata.get("chunk_uid", "")))

        hits = hits[:top_k]
        result = self._apply_guardrail(hits, strategy)
        result.language = language
        return result

    def _apply_guardrail(self, hits: list[Hit], strategy: str) -> RetrievalResult:
        """Decide whether the generator is allowed to answer at all.

        The threshold depends on the strategy, because the scores mean different
        things: a reranker logit is not a cosine.
        """
        if not hits:
            return RetrievalResult([], False, strategy, reason="no candidates")

        top = hits[0].score
        if strategy == "hybrid_rerank":
            threshold = settings.min_rerank_score
        elif strategy in ("dense",):
            threshold = settings.min_similarity
        else:
            threshold = float("-inf")     # RRF/BM25 scores are not calibrated

        confident = top >= threshold
        reason = "" if confident else f"top score {top:.3f} < threshold {threshold:.3f}"
        return RetrievalResult(hits, confident, strategy, top_score=top, reason=reason)


_default: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide singleton — building the BM25 index on every request would
    be absurd, and the model loads are expensive too."""
    global _default
    if _default is None:
        _default = Retriever()
    return _default
