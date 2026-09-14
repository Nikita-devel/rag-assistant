"""Generation: retrieved chunks -> a grounded, cited answer.

Two rules decide almost everything here.

1. NO CONTEXT, NO LLM CALL. When the retriever's guardrail says it is not
   confident, the model is never invoked. Not "invoked with a careful prompt
   telling it to refuse" — not invoked at all. A prompt instruction is a request;
   skipping the call is a guarantee. It also means an out-of-scope question costs
   nothing, which matters on a public demo with a shared API budget.

2. EVERY CLAIM CARRIES ITS ARTICLE. The chunks arrive numbered, the model is
   required to cite by number, and we resolve those numbers back to real article
   references afterwards. A B2B prospect does not need to trust the model — they
   click the source and read the text of L3141-3 themselves. That verifiability
   is the entire product.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.config import settings
from app.llm import LLMError, complete, describe
from app.retriever import Hit, RetrievalResult, get_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an assistant answering questions about French labour \
law for a small industrial company (metalworking / sheet metal). You answer \
ONLY from the numbered extracts supplied below, which come from the Code du \
travail and the convention collective de la métallurgie (IDCC 3248).

Rules, in order of importance:

1. Use only the supplied extracts. Never add legal knowledge from memory, even \
if you are certain it is correct — the user is relying on being able to verify \
every statement against the cited text.
2. Cite the extract number in square brackets after each statement that rests \
on it, like [2]. Every factual sentence carries at least one citation.
3. If the extracts do not answer the question, say so plainly and stop. Do not \
reason your way to a plausible answer. A wrong answer about working time costs \
this company real money.
4. When the Code du travail and the convention collective both apply, say so and \
give both — the convention often grants more than the law, and that difference \
is usually the thing the reader actually needs.
5. Answer in {language_name}. Keep it short: three to six sentences unless the \
question genuinely needs more. Write for a manager, not a lawyer — plain \
sentences, no preamble, no "according to the extracts provided"."""

LANGUAGE_NAMES = {"fr": "French", "en": "English"}

# Shown when retrieval worked but the model did not answer. Deliberately does
# not name a cause — rate limit, missing quota, local server down all reach the
# user the same way, and the real reason is in Answer.reason and the log.
LLM_ERROR = {
    "fr": ("Le service de génération est momentanément indisponible. Les sources "
           "ci-dessous ont bien été trouvées."),
    "en": ("The generation service is temporarily unavailable. The sources below "
           "were still retrieved successfully."),
}

REFUSAL = {
    "fr": ("Je n'ai pas d'information sur ce point dans les documents dont je "
           "dispose (Code du travail et convention collective de la métallurgie). "
           "Je préfère vous le dire plutôt que de risquer une réponse inexacte."),
    "en": ("I don't have information on this in the documents available to me "
           "(French Labour Code and the metallurgy collective agreement). I would "
           "rather say so than risk an inaccurate answer."),
}

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    index: int
    article: str
    document: str
    section: str
    excerpt: str
    score: float
    cited: bool = False


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    answered: bool = True          # False = guardrail refused, no LLM call made
    language: str = "fr"
    model: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "answered": self.answered,
            "language": self.language,
            "model": self.model,
            "sources": [
                {
                    "n": s.index,
                    "document": s.document,
                    "article": s.article,
                    "section": s.section,
                    "excerpt": s.excerpt,
                    "score": round(s.score, 3),
                    "cited": s.cited,
                }
                for s in self.sources
            ],
        }


# --------------------------------------------------------------------------- #
def build_context(hits: list[Hit], max_chars: int = 4500) -> tuple[str, list[Source]]:
    """Number the extracts and keep the context inside a sane budget.

    The cap serves two purposes. Focus: padding the prompt with marginal
    extracts measurably increases the odds the model cites the wrong one. And
    fit: 4500 characters of French is roughly 1250 tokens, which leaves room
    inside a 4096-token context for the instructions, the question and the
    answer. Raising this without raising OLLAMA_NUM_CTX silently truncates the
    extracts, and the model then cites only the ones it actually saw.
    """
    blocks, sources, used = [], [], 0
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        article = (meta.get("article") or "").strip()
        document = (meta.get("source") or "").replace(".md", "")
        section = (meta.get("section") or "").strip()
        text = hit.text.strip()

        if used + len(text) > max_chars and blocks:
            break
        used += len(text)

        header = f"[{i}] {article}" if article else f"[{i}]"
        if section:
            header += f" — {section}"
        header += f" (source : {document})"
        blocks.append(f"{header}\n{text}")
        sources.append(Source(
            index=i, article=article, document=document, section=section,
            excerpt=text, score=hit.score,
        ))
    return "\n\n".join(blocks), sources


def mark_cited(answer_text: str, sources: list[Source]) -> None:
    cited = {int(n) for n in CITATION_RE.findall(answer_text)}
    for source in sources:
        source.cited = source.index in cited


DANGLING_CITATION_RE = re.compile(r"(?:^[ \t]*(?:\[\d+\][ \t,;]*)+$\n?)+\Z", re.MULTILINE)
LEADING_CITATION_RE = re.compile(r"\A(?:\[\d+\][ \t,;]*)+")


def _strip_dangling_citations(text: str) -> str:
    """Tidy the citation markers small models scatter around the answer.

    Two habits, both cosmetic and both bad in a demo:

      trailing — a bare reference list after an answer that already cites
                 inline ("…un jour franc.\n\n[1]\n[2]")
      leading  — a marker before the first word ("[1] Le délai minimum est…"),
                 which reads like a numbered list item rather than a sentence

    Markers inside the prose are the whole point of the product and are left
    exactly where the model put them.
    """
    text = DANGLING_CITATION_RE.sub("", text)
    text = LEADING_CITATION_RE.sub("", text.lstrip())
    return text.strip()


def answer_question(question: str, result: RetrievalResult | None = None) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer("", answered=False, reason="empty question")
    if len(question) > settings.max_question_length:
        return Answer("", answered=False,
                      reason=f"question exceeds {settings.max_question_length} characters")

    result = result or get_retriever().search(question)
    language = result.language if result.language in LANGUAGE_NAMES else "fr"

    # Rule 1: the guardrail refused, so no call is made at all.
    if not result.confident or not result.hits:
        logger.info("refused: %s", result.reason or "no hits")
        return Answer(REFUSAL[language], sources=[], answered=False,
                      language=language, reason=result.reason)

    context, sources = build_context(result.hits)
    system = SYSTEM_PROMPT.format(language_name=LANGUAGE_NAMES[language])
    user = (f"Extraits disponibles :\n\n{context}\n\n"
            f"---\nQuestion : {question}")

    try:
        text = complete(system, user)
    except LLMError as exc:
        logger.error("LLM call failed: %s", exc)
        return Answer(LLM_ERROR[language], sources=sources, answered=False,
                      language=language, model=describe(),
                      reason=f"LLM error: {exc}")

    text = _strip_dangling_citations(text)
    mark_cited(text, sources)
    if not any(s.cited for s in sources):
        # The model answered without citing anything. The answer may well be
        # right, but the product promise is verifiability — flag it rather than
        # present an uncitable answer as if it were sourced.
        logger.warning("answer contained no citations")

    return Answer(text=text, sources=sources, answered=True,
                  language=language, model=describe())
