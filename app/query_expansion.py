"""Query preprocessing: acronym expansion, synonyms, cross-lingual handling.

The evaluation showed three failure modes that are all the same problem — the
question and the article use different words for the same thing:

  "CSE"          vs  "comité social et économique"   (acronym never spelled in the query)
  "hebdomadaire" vs  "par semaine"                   (synonym)
  English query  vs  a corpus that is entirely French

No embedding model fixes an acronym that appears nowhere in the indexed text.
A 40-line domain map does, deterministically and for free.

Design: the EXPANDED query feeds BM25 only. Dense retrieval gets the original
(or the translated) query, because padding a sentence with synonyms shifts its
embedding away from the thing it was actually asking about. Lexical search wants
more surface forms; semantic search wants a clean sentence.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Domain acronyms. Both directions: a query saying "CSE" gets the long form,
# a query saying the long form gets "CSE" — chunks use both.
ACRONYMS: dict[str, str] = {
    "cse": "comité social et économique",
    "chsct": "comité hygiène sécurité conditions de travail",
    "cssct": "commission santé sécurité et conditions de travail",
    "epi": "équipement de protection individuelle",
    "cdd": "contrat à durée déterminée",
    "cdi": "contrat à durée indéterminée",
    "smic": "salaire minimum interprofessionnel de croissance",
    "rtt": "réduction du temps de travail",
    "duerp": "document unique d'évaluation des risques professionnels",
    "at": "accident du travail",
    "mp": "maladie professionnelle",
    "ccn": "convention collective nationale",
    "dp": "délégué du personnel",
    "ds": "délégué syndical",
    "idcc": "identifiant de convention collective",
    "aft": "aménagement du temps de travail",
}

# Synonym groups. Any member in the query pulls in the others.
SYNONYMS: list[set[str]] = [
    {"hebdomadaire", "semaine", "semaines"},
    {"mensuel", "mensuelle", "mois"},
    {"annuel", "annuelle", "année", "an"},
    {"quotidien", "quotidienne", "jour", "journalier"},
    {"salarié", "salariés", "employé", "employés", "travailleur", "travailleurs"},
    {"employeur", "entreprise", "établissement"},
    {"rémunération", "salaire", "paie", "paye"},
    {"licenciement", "rupture", "congédiement"},
    {"préavis", "prévenance", "délai"},
    {"indemnité", "indemnisation", "compensation"},
    {"ancienneté", "durée de présence"},
    {"congé", "congés", "absence", "repos"},
    {"sécurité", "protection", "prévention"},
    {"obligation", "obligations", "devoir", "tenu"},
    {"seuil", "effectif", "nombre"},
    {"habillage", "déshabillage", "tenue de travail"},
    {"astreinte", "astreintes", "permanence"},
    {"télétravail", "travail à distance"},
    {"harcèlement", "agissements hostiles"},
]

# Minimal EN -> FR bridge for the demo's bilingual promise. This is a fallback:
# it covers the vocabulary a prospect is likely to type in English, and costs
# nothing. `translate_query` (LLM) is the general solution when a key is set.
EN_FR: dict[str, str] = {
    "paid leave": "congés payés",
    "paid holiday": "congés payés",
    "holiday": "congés",
    "leave": "congé",
    "working time": "durée du travail",
    "working hours": "durée du travail",
    "overtime": "heures supplémentaires",
    "notice period": "préavis",
    "dismissal": "licenciement",
    "severance": "indemnité de licenciement",
    "safety": "sécurité",
    "safety obligation": "obligation de sécurité",
    "employer": "employeur",
    "employee": "salarié",
    "seniority": "ancienneté",
    "wage": "salaire",
    "salary": "salaire",
    "minimum wage": "salaire minimum",
    "protective equipment": "équipement de protection individuelle",
    "harassment": "harcèlement",
    "remote work": "télétravail",
    "on-call": "astreinte",
    "works council": "comité social et économique",
    "trial period": "période d'essai",
    "night work": "travail de nuit",
    "accrue": "acquérir",
    "entitled": "droit",
}

_WORD = re.compile(r"[\w'-]+", re.UNICODE)

ACCENTED_CHARS = re.compile(r"[àâäçéèêëîïôöùûüÿœæ]", re.IGNORECASE)


def restore_accents(question: str) -> str:
    """Put the diacritics back, word by word, from the corpus-derived map.

    Applied ONLY when the question contains no accented character at all. If the
    user typed even one, they have a French keyboard and their spelling is
    better evidence than our map — leaving it alone avoids "corrections" that
    are really corruptions.
    """
    if ACCENTED_CHARS.search(question):
        return question

    from app.accents_fr import ACCENTS

    def swap(match: re.Match) -> str:
        word = match.group(0)
        accented = ACCENTS.get(word.lower())
        if accented is None:
            return word
        return accented.capitalize() if word[0].isupper() else accented

    return _WORD.sub(swap, question)


def detect_language(text: str) -> str:
    """Cheap and dependency-free: count French vs English function words.

    langdetect is in requirements for the generator, but it is unreliable on
    5-word queries, which is exactly what we get here.
    """
    words = {w.lower() for w in _WORD.findall(text)}
    fr = len(words & {"le", "la", "les", "de", "du", "des", "est", "quel",
                      "quelle", "combien", "pour", "dans", "un", "une", "au",
                      "aux", "sur", "que", "qui", "doit", "faut", "il", "en"})
    en = len(words & {"the", "is", "are", "what", "how", "many", "much", "does",
                      "do", "of", "to", "in", "for", "a", "an", "must", "should",
                      "per", "can", "and"})
    return "en" if en > fr else "fr"


def bridge_english(question: str) -> str:
    """Append French equivalents of recognised English terms.

    Longest phrases first, so "paid leave" wins over "leave".
    """
    lowered = question.lower()
    added: list[str] = []
    for term in sorted(EN_FR, key=len, reverse=True):
        if term in lowered:
            french = EN_FR[term]
            if french not in added:
                added.append(french)
    return f"{question} {' '.join(added)}".strip() if added else question


def expand(question: str) -> str:
    """Acronyms + synonyms. Output is for BM25, never for the embedder."""
    tokens = [t.lower() for t in _WORD.findall(question)]
    token_set = set(tokens)
    extra: list[str] = []

    for short, long in ACRONYMS.items():
        if short in token_set:
            extra.append(long)
        elif all(w in token_set for w in long.split() if len(w) > 3):
            extra.append(short)

    for group in SYNONYMS:
        single = {t for t in group if " " not in t}
        phrases = {t for t in group if " " in t}
        hit = bool(token_set & single) or any(p in question.lower() for p in phrases)
        if hit:
            extra.extend(sorted(group - token_set))

    if not extra:
        return question
    return f"{question} {' '.join(dict.fromkeys(extra))}"


def prepare(question: str) -> tuple[str, str, str]:
    """Return (semantic_query, lexical_query, detected_language).

    semantic_query — clean sentence for the embedder: accents restored if the
                     French question was typed without them, or bridged to
                     French if the question was English.
    lexical_query  — same, plus every acronym and synonym we can add.
    """
    language = detect_language(question)
    if language == "en":
        semantic = bridge_english(question)
    else:
        semantic = restore_accents(question)
    lexical = expand(semantic)
    if lexical != question:
        logger.debug("query expanded: %r -> %r", question, lexical)
    return semantic, lexical, language
