"""LLM backend — one interface, swappable providers.

Why this module exists: the project was wired directly to Mistral, and when that
account returned 429 on every request the whole demo was dead. A provider the
demo cannot run without is a single point of failure, and for a portfolio piece
shown to prospects on a schedule you do not control, that is unacceptable.

Providers:

  mistral — hosted, fast, costs money (or needs an activated free tier).
  ollama  — a model running on your own machine. No key, no quota, no rate
            limit, and nothing leaves the network. For an assistant sold to
            French industrial SMEs on exactly that promise, this is not the
            fallback — it is arguably the better demo: "your HR documents never
            leave your building" is a sentence the hosted version cannot say.

Pick with LLM_PROVIDER in .env. Everything above this module is unchanged.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Iterator

from app.config import settings

logger = logging.getLogger(__name__)

TEMPERATURE = 0.1        # this is extraction, not creative writing
MAX_TOKENS = 700         # hosted providers; Ollama uses ollama_num_predict


class LLMError(RuntimeError):
    """Any provider failure, normalised so callers need not know the SDK."""


class LLMUnavailable(LLMError):
    """Rate limited, out of quota, or the local server is not running."""


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def _mistral(system: str, user: str) -> str:
    from mistralai import Mistral

    if not settings.mistral_api_key:
        raise LLMError("MISTRAL_API_KEY is not set — check your .env")

    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.complete(
        model=settings.llm_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def _ollama(system: str, user: str) -> str:
    import requests

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    try:
        response = requests.post(
            url,
            json={
                "model": settings.ollama_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "stream": False,
                "keep_alive": settings.ollama_keep_alive,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": settings.ollama_num_predict,
                    "num_ctx": settings.ollama_num_ctx,
                },
            },
            timeout=settings.ollama_timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnavailable(
            f"cannot reach Ollama at {settings.ollama_base_url} — is `ollama serve` "
            f"running?") from exc
    except requests.exceptions.Timeout as exc:
        raise LLMUnavailable(
            f"Ollama did not respond within {settings.ollama_timeout}s — a large "
            f"model on CPU may need a longer OLLAMA_TIMEOUT") from exc

    if response.status_code == 404:
        raise LLMError(
            f"model '{settings.ollama_model}' is not pulled — run: "
            f"ollama pull {settings.ollama_model}")
    if response.status_code != 200:
        raise LLMError(f"Ollama returned {response.status_code}: {response.text[:200]}")

    return response.json()["message"]["content"].strip()


PROVIDERS = {"mistral": _mistral, "ollama": _ollama}


# --------------------------------------------------------------------------- #
def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, LLMUnavailable):
        return False          # a dead local server will still be dead in 3s
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "timeout" in text or (
        "50" == str(getattr(exc, "status_code", ""))[:2])


def complete(system: str, user: str, attempts: int = 3) -> str:
    """Send one prompt, return the text. Retries transient failures only.

    Retrying a hard failure (no quota, model not pulled, server down) just makes
    the user wait longer for the same error, so those raise immediately.
    """
    provider = PROVIDERS.get(settings.llm_provider)
    if provider is None:
        raise LLMError(
            f"unknown LLM_PROVIDER '{settings.llm_provider}' — "
            f"expected one of: {', '.join(PROVIDERS)}")

    delay = 1.5
    for attempt in range(1, attempts + 1):
        try:
            return provider(system, user)
        except Exception as exc:
            if not _is_transient(exc) or attempt == attempts:
                raise LLMError(str(exc)) from exc
            wait = delay + random.uniform(0, 0.5)   # jitter, so parallel callers
            logger.warning("transient LLM failure, retry %d/%d in %.1fs: %s",
                           attempt, attempts, wait, exc)
            time.sleep(wait)
            delay *= 2

    raise LLMError("unreachable")


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def stream(system: str, user: str) -> Iterator[str]:
    """Yield the answer in pieces as the model produces them.

    A caveat worth knowing before you count on this: on CPU the wait is
    dominated by PREFILL — the model reading the ~1800-token prompt — and the
    first token cannot appear until that finishes. Streaming removes the second
    half of the wait, not the first. It is still worth having (a page that shows
    text arriving reads as working rather than hung), but it is not a fix for a
    slow machine.

    Only Ollama streams here; the hosted path falls back to one blocking call
    and yields the whole answer at once, which keeps callers uniform.
    """
    if settings.llm_provider != "ollama":
        yield complete(system, user)
        return

    import requests

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    try:
        with requests.post(
            url,
            json={
                "model": settings.ollama_model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "stream": True,
                "keep_alive": settings.ollama_keep_alive,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": settings.ollama_num_predict,
                    "num_ctx": settings.ollama_num_ctx,
                },
            },
            timeout=settings.ollama_timeout,
            stream=True,
        ) as response:
            if response.status_code == 404:
                raise LLMError(
                    f"model '{settings.ollama_model}' is not pulled — run: "
                    f"ollama pull {settings.ollama_model}")
            if response.status_code != 200:
                raise LLMError(f"Ollama returned {response.status_code}")

            for raw in response.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue            # a partial line; the next one completes it
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnavailable(
            f"cannot reach Ollama at {settings.ollama_base_url} — is `ollama serve` "
            f"running?") from exc
    except requests.exceptions.Timeout as exc:
        raise LLMUnavailable(
            f"Ollama stopped responding after {settings.ollama_timeout}s") from exc


def describe() -> str:
    """Short label for logs and the UI footer."""
    if settings.llm_provider == "ollama":
        return f"ollama/{settings.ollama_model}"
    return f"mistral/{settings.llm_model}"
