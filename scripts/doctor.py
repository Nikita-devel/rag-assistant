"""Check every moving part before a demo, and say exactly what is broken.

    python scripts/doctor.py

Runs in dependency order and stops being useful the moment something fails, so
the first FAIL is the thing to fix. Written because "why is the demo not
answering" was costing more time than the bugs themselves.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, FAIL, WARN = "\033[32mOK  \033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
problems: list[str] = []


def line(status: str, name: str, detail: str = "") -> None:
    print(f"  {status}  {name}{'  — ' + detail if detail else ''}")


def fail(name: str, detail: str, fix: str) -> None:
    line(FAIL, name, detail)
    problems.append(f"{name}: {fix}")


def main() -> None:
    print("\nconfiguration")
    from app.config import settings

    line(OK, "config loaded", f"provider={settings.llm_provider}")
    line(OK, "embedding model", settings.embedding_model)
    line(OK, "reranker", settings.reranker_model)
    line(OK, "guardrail threshold", f"MIN_RERANK_SCORE={settings.min_rerank_score}")

    print("\ncorpus")
    docs = list(settings.documents_path.glob("*.md"))
    if docs:
        size = sum(d.stat().st_size for d in docs) / 1024
        line(OK, "documents", f"{len(docs)} files, {size:.0f} KB")
    else:
        fail("documents", f"none in {settings.documents_path}",
             "python scripts/build_corpus.py")

    print("\nvector store")
    try:
        from app.ingestion import get_collection
        count = get_collection().count()
        if count:
            line(OK, "chroma collection", f"{count} chunks")
        else:
            fail("chroma collection", "empty", "python -m app.ingestion --reset")
    except Exception as exc:
        fail("chroma collection", str(exc)[:90], "python -m app.ingestion --reset")
        count = 0

    print("\nretrieval")
    if count:
        try:
            from app.retriever import get_retriever
            t0 = time.perf_counter()
            result = get_retriever().search("Combien de jours de congés payés par mois ?")
            elapsed = time.perf_counter() - t0
            if result.hits:
                top = result.hits[0].metadata.get("article", "?")
                line(OK, "end-to-end search", f"{elapsed:.1f}s, top={top}")
            else:
                fail("end-to-end search", "no hits", "check the ingestion ran")
            if result.confident:
                line(OK, "guardrail accepts a valid question",
                     f"score {result.top_score:.2f}")
            else:
                fail("guardrail accepts a valid question",
                     f"refused at {result.top_score:.2f} < {result.reason}",
                     "python scripts/calibrate_guardrail.py, then lower "
                     "MIN_RERANK_SCORE in .env")

            junk = get_retriever().search("Quelle est la recette de la tarte tatin ?")
            if junk.confident:
                line(WARN, "guardrail rejects nonsense",
                     f"accepted at {junk.top_score:.2f} — the prompt is the only "
                     f"defence left")
            else:
                line(OK, "guardrail rejects nonsense", f"score {junk.top_score:.2f}")
        except Exception as exc:
            fail("end-to-end search", str(exc)[:90], "see the traceback with -v")
    else:
        line(WARN, "retrieval", "skipped, no chunks indexed")

    print(f"\nLLM provider — {settings.llm_provider}")
    try:
        from app.llm import complete, describe
        t0 = time.perf_counter()
        reply = complete("Answer with exactly one word.", "Say OK.")
        line(OK, describe(), f"{time.perf_counter() - t0:.1f}s, replied "
                             f"{reply[:30]!r}")
    except Exception as exc:
        message = str(exc)
        if settings.llm_provider == "ollama":
            if "cannot reach" in message:
                fix = "start it:  ollama serve"
            elif "not pulled" in message:
                fix = f"ollama pull {settings.ollama_model}"
            else:
                fix = "check `ollama list` and OLLAMA_MODEL in .env"
        else:
            fix = ("check quota at console.mistral.ai, or switch to a local "
                   "model: LLM_PROVIDER=ollama in .env")
        fail(f"{settings.llm_provider} reachable", message[:90], fix)

    print()
    if problems:
        print(f"{len(problems)} problem(s) to fix, in order:\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        sys.exit(1)
    print("everything green — python scripts/ask.py --demo")


if __name__ == "__main__":
    main()
