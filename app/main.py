"""FastAPI application.

Endpoints:
    GET  /health        — is the service alive, and with what loaded
    POST /query         — question in, cited answer out
    POST /query/stream  — the same, as Server-Sent Events
    POST /ingest        — add a document to the index

Two decisions worth stating.

WARM-UP AT STARTUP. The embedder, the BM25 index and the cross-encoder together
take ~10 s to load. Doing that lazily means the first visitor to the public demo
waits 10 s longer than everyone else — and the first visitor is usually the
prospect you sent the link to. The lifespan hook pays that cost before the port
opens.

SOURCES BEFORE TOKENS. The streaming endpoint sends the retrieved sources as its
first event, before the model has produced a word. The page can render the
citations immediately, which is the part a B2B visitor actually studies, while
the prose is still arriving.
"""
# NOTE: no `from __future__ import annotations` here, deliberately.
# It turns every annotation into a string, and FastAPI then cannot resolve the
# non-pydantic types in route signatures — `UploadFile` and `Request` arrive as
# ForwardRef and the app refuses to start. The rest of the package keeps the
# future import (it is what lets torch and chromadb load lazily); this module
# is the exception.

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.generator import (
    LANGUAGE_NAMES, LLM_ERROR, REFUSAL, SYSTEM_PROMPT, _strip_dangling_citations,
    answer_question, build_context, mark_cited,
)
from app.llm import LLMError, describe, stream
from app.retriever import get_retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("warming up models …")
    t0 = time.perf_counter()
    retriever = await asyncio.to_thread(get_retriever)
    # A throwaway query forces the cross-encoder to load too — it is lazy, and
    # loading it on the first real request would cost that visitor ~5 s.
    await asyncio.to_thread(retriever.search, "congés payés")
    logger.info("ready in %.1fs — %d chunks, provider %s",
                time.perf_counter() - t0, len(retriever._docs), describe())
    yield


app = FastAPI(
    title="Assistant droit du travail",
    description="RAG over the Code du travail and the metallurgy collective "
                "agreement (IDCC 3248). Every answer cites its article.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# The demo page is served from this same origin; a permissive policy exists only
# so the API can be tried from elsewhere. Tighten allow_origins before putting a
# real client's data behind it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=settings.max_question_length)


class SourceOut(BaseModel):
    n: int
    document: str
    article: str
    section: str
    excerpt: str
    score: float
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    answered: bool
    language: str
    model: str
    sources: list[SourceOut]
    elapsed_seconds: float


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """The demo page, served from the same origin as the API.

    Same origin on purpose: no CORS preflight on every question, and one URL to
    put behind the tunnel instead of two things to keep in sync.
    """
    if not FRONTEND.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html is missing")
    # The page is one file and changes with the code; letting a browser cache it
    # is how you demo yesterday's build without noticing.
    return FileResponse(FRONTEND, media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health() -> dict:
    """Cheap enough to hit every 30 s from a monitor.

    Returns 503 when the index is empty. An earlier version answered 200 with
    zero chunks, so Docker's healthcheck reported a healthy container that could
    not answer a single question — the worst kind of green light.
    """
    retriever = get_retriever()
    chunks = len(retriever._docs)
    if chunks == 0:
        raise HTTPException(
            status_code=503,
            detail="index is empty — run `python -m app.ingestion --reset`")
    return {
        "status": "ok",
        "chunks": chunks,
        "provider": settings.llm_provider,
        "model": describe(),
        "embedding_model": settings.embedding_model,
        "guardrail_threshold": settings.min_rerank_score,
    }


@app.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    # answer_question is CPU- and network-bound and fully synchronous; off the
    # event loop it goes, or one slow question blocks every other visitor.
    answer = await asyncio.to_thread(answer_question, body.question)
    payload = answer.to_dict()
    return QueryResponse(**payload, elapsed_seconds=round(time.perf_counter() - t0, 2))


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
@limiter.limit(settings.rate_limit)
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        question = body.question.strip()

        result = await asyncio.to_thread(get_retriever().search, question)
        language = result.language if result.language in LANGUAGE_NAMES else "fr"

        if not result.confident or not result.hits:
            yield _sse("refused", {"answer": REFUSAL[language], "language": language,
                                   "reason": result.reason})
            yield _sse("done", {"elapsed_seconds": round(time.perf_counter() - t0, 2)})
            return

        context, sources = build_context(result.hits)
        # Citations first: the page renders them while the model is still
        # reading the prompt, which on CPU is most of the wait.
        yield _sse("sources", {
            "language": language,
            "model": describe(),
            "sources": [{"n": s.index, "document": s.document, "article": s.article,
                         "section": s.section, "excerpt": s.excerpt,
                         "score": round(s.score, 3)} for s in sources],
        })

        system = SYSTEM_PROMPT.format(language_name=LANGUAGE_NAMES[language])
        user = f"Extraits disponibles :\n\n{context}\n\n---\nQuestion : {question}"

        pieces: list[str] = []
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def pump() -> None:
            try:
                for piece in stream(system, user):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", piece))
            except LLMError as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

        asyncio.get_running_loop().run_in_executor(None, pump)

        failed = ""
        while True:
            kind, payload = await queue.get()
            if kind == "eof":
                break
            if kind == "error":
                failed = payload
                break
            pieces.append(payload)
            yield _sse("token", {"text": payload})

        if failed:
            logger.error("stream failed: %s", failed)
            yield _sse("error", {"answer": LLM_ERROR[language], "reason": failed})
            yield _sse("done", {"elapsed_seconds": round(time.perf_counter() - t0, 2)})
            return

        # Tokens went out raw; the cleaned text and the citation flags can only
        # be computed once the answer is complete, so they come at the end and
        # the page swaps them in.
        text = _strip_dangling_citations("".join(pieces))
        mark_cited(text, sources)
        yield _sse("final", {
            "answer": text,
            "cited": [s.index for s in sources if s.cited],
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
        })

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest")
@limiter.limit("3/minute")
async def ingest(request: Request, file: UploadFile = File(...)) -> dict:
    from app.ingestion import SUPPORTED_SUFFIXES, ingest_file

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type '{suffix}'; expected one of "
                   f"{sorted(SUPPORTED_SUFFIXES)}")

    target = settings.documents_path / Path(file.filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())

    try:
        result = await asyncio.to_thread(ingest_file, target)
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The lexical index lives in memory and knows nothing about the new chunks
    # until it is rebuilt. Skipping this is how a freshly uploaded document
    # becomes findable by vector search but invisible to BM25.
    await asyncio.to_thread(get_retriever()._load_lexical_index)

    return {"status": "success", **result}
