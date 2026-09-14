"""Tests for the pure logic — no model downloads, no vector store, no API key.

    python tests/test_core.py

Everything heavy (torch, chromadb, the Mistral client) is imported lazily inside
the functions that need it, so this file exercises the real modules rather than
copies of them, and runs in under a second.

What is covered is what has actually broken in this project:
  - article-label matching (silently scored every strategy at 0.00)
  - BM25 determinism (moved the calibrated guardrail threshold between runs)
  - query expansion (the fix for the CSE / hebdomadaire / English misses)
  - citation parsing (the product promise is that answers are verifiable)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.generator import CITATION_RE, Source, build_context, mark_cited  # noqa: E402
from app.ingestion import _recursive_split, split_by_structure  # noqa: E402
from app.query_expansion import (  # noqa: E402
    bridge_english, detect_language, expand, prepare, restore_accents,
)
from app.retriever import BM25, Hit, tokenize  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{' — ' + detail if detail else ''}")
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
def test_tokenize() -> None:
    print("\ntokenize")
    check("folds accents", tokenize("Congés Payés") == ["conges", "payes"])
    check("keeps article numbers whole", "l3141-3" in tokenize("Article L3141-3"))
    check("drops stopwords", "le" not in tokenize("le salarié"))
    check("drops single chars", tokenize("a b salarié") == ["salarie"])


def test_bm25() -> None:
    print("\nBM25")
    docs = [
        "Le salarié a droit à un congé de deux jours et demi ouvrables par mois",
        "La durée légale de travail effectif est fixée à trente-cinq heures par semaine",
        "Un comité social et économique est mis en place dans les entreprises "
        "d'au moins onze salariés",
        "Les équipements de protection individuelle sont fournis gratuitement",
    ]
    bm25 = BM25(docs)

    hits = bm25.search("congé par mois", top_k=3)
    check("finds the right document", bool(hits) and hits[0][0] == 0,
          f"got {hits[:1]}")

    check("empty query returns nothing", bm25.search("zzzz qqqq", top_k=3) == [])

    # The bug this guards: iteration order of a set decided ties, so identical
    # inputs gave different rankings in different processes.
    repeated = [BM25(docs).search("salarié entreprise", top_k=4) for _ in range(5)]
    check("deterministic across rebuilds", all(r == repeated[0] for r in repeated),
          f"{repeated}")

    tied = BM25(["alpha beta", "alpha beta", "alpha beta"])
    order = tied.search("alpha beta", top_k=3)
    check("ties break by document index", [i for i, _ in order] == [0, 1, 2],
          f"{order}")


def test_language_detection() -> None:
    print("\nlanguage detection")
    check("french question", detect_language("Quelle est la durée du travail ?") == "fr")
    check("english question", detect_language("What is the working time?") == "en")
    check("english short", detect_language("How many days per month?") == "en")


def test_expansion() -> None:
    print("\nquery expansion")
    expanded = expand("faut-il un CSE ?")
    check("expands acronym", "comité social et économique" in expanded, expanded)

    expanded = expand("durée légale hebdomadaire")
    check("expands synonym", "semaine" in expanded, expanded)

    bridged = bridge_english("How many paid holiday days per month?")
    check("bridges EN to FR", "congés payés" in bridged, bridged)

    bridged = bridge_english("What is the employer's safety obligation?")
    check("longest phrase wins", "obligation de sécurité" in bridged, bridged)

    semantic, lexical, language = prepare("À partir de combien de salariés faut-il un CSE ?")
    check("semantic form stays clean", "comité social" not in semantic, semantic)
    check("lexical form is padded", "comité social et économique" in lexical)
    check("language reported", language == "fr")

    untouched = "Quel est le prix du bitcoin ?"
    check("unknown terms pass through", expand(untouched).startswith(untouched))


def test_accent_restoration() -> None:
    """The failure this fixes: asked without diacritics, the correct article
    dropped from rank 1 to rank 5 because the embedder saw different words."""
    print("\naccent restoration")
    restored = restore_accents("Quel delai pour prevenir un salarie ?")
    check("restores domain words", restored == "Quel délai pour prévenir un salarié ?",
          restored)

    already = "Le délai est de 15 jours."
    check("leaves an accented question alone", restore_accents(already) == already)

    check("keeps capitalisation",
          restore_accents("Delai de prevenance") == "Délai de prévenance",
          restore_accents("Delai de prevenance"))

    # The whole point of the corpus-derived safety filter: these words exist
    # unaccented too, so rewriting them would corrupt more queries than it fixes.
    from app.accents_fr import ACCENTS
    for word in ("sur", "cote", "mur", "jeune", "marche", "entre", "pale"):
        check(f"{word!r} not rewritten", word not in ACCENTS,
              f"maps to {ACCENTS.get(word)}")

    check("map is non-trivial", len(ACCENTS) > 300, f"{len(ACCENTS)} entries")

    semantic, _, _ = prepare("Combien de jours de conges payes par mois ?")
    check("prepare() restores before embedding", "congés payés" in semantic, semantic)


def test_structure_split() -> None:
    print("\nstructure-aware splitting")
    text = (
        "TITRE IV : CONGÉS PAYÉS\n\n"
        "Article L3141-3\nLe salarié a droit à un congé.\n\n"
        "Article L3141-5\nSont assimilées à un mois de travail.\n\n"
        "Article 96.2.1.2\nL'employeur informe le salarié.\n"
    )
    blocks = split_by_structure(text)
    labels = [label for _, label in blocks if label]
    check("splits Code du travail articles", "Article L3141-3" in labels, str(labels))
    check("splits convention sub-articles", "Article 96.2.1.2" in labels, str(labels))
    check("one block per article", len(labels) == 3, str(labels))

    plain = "No articles here, just prose that goes on for a while."
    check("prose stays one block", len(split_by_structure(plain)) == 1)


def test_recursive_split() -> None:
    print("\nrecursive splitting")
    long_text = "Le salarié bénéficie de dispositions particulières. " * 120
    parts = _recursive_split(long_text, 1400, 200)
    check("splits long text", len(parts) > 1, f"{len(parts)} parts")
    check("respects size + overlap budget",
          all(len(p) <= 1400 + 200 + 60 for p in parts),
          f"max {max(len(p) for p in parts)}")
    check("short text is untouched",
          _recursive_split("court", 1400, 200) == ["court"])


def test_citations() -> None:
    print("\ncitations")
    check("parses bracketed numbers",
          [int(n) for n in CITATION_RE.findall("Oui [1] et aussi [3].")] == [1, 3])

    sources = [Source(i, f"Article {i}", "doc.md", "", "texte", 1.0) for i in (1, 2, 3)]
    mark_cited("La réponse est oui [2].", sources)
    check("marks only cited sources",
          [s.cited for s in sources] == [False, True, False])

    hits = [
        Hit(text="Le salarié a droit à un congé.",
            metadata={"article": "Article L3141-3", "source": "conges_payes.md",
                      "section": "TITRE IV"},
            score=5.0),
        Hit(text="La durée légale est de 35 heures.",
            metadata={"article": "Article L3121-27", "source": "duree.md",
                      "section": ""},
            score=4.0),
    ]
    context, built = build_context(hits)
    check("numbers extracts from 1", context.startswith("[1] Article L3141-3"), context[:40])
    check("carries the section", "TITRE IV" in context)
    check("names the source document", "conges_payes" in context)
    check("one Source per chunk", [s.index for s in built] == [1, 2])

    # A single chunk larger than the budget must still be returned, or a long
    # article would silently produce an empty context.
    big = [Hit(text="x" * 9000, metadata={"article": "A", "source": "d.md"}, score=1.0)]
    ctx, srcs = build_context(big, max_chars=6000)
    check("oversized single chunk is kept", len(srcs) == 1 and bool(ctx))


def test_eval_matching() -> None:
    print("\neval article matching")
    spec = importlib.util.spec_from_file_location(
        "ev", ROOT / "scripts" / "eval_retrieval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    cases = [
        ("Article L3141-3", {"L3141-3"}, True, "strips the Article prefix"),
        ("article l4121-1", {"L4121-1"}, True, "case insensitive"),
        ("Article 75.3.3", {"75.3"}, True, "sub-article counts"),
        ("Article 75.3", {"75.3"}, True, "exact match"),
        ("Article L3141-30", {"L3141-3"}, False, "no prefix false positive"),
        ("Article 68", {"75.3"}, False, "unrelated article"),
        ("", {"L3141-3"}, False, "empty label"),
        ("Convention collective — Titre VIII", {"96.1"}, False, "heading is not an article"),
    ]
    for found, accepted, expected, name in cases:
        check(name, ev.matches(found, accepted) is expected, f"{found!r}")

    check("labels are non-empty", all(a for _, a in ev.LABELLED))
    check("no duplicate questions",
          len({q for q, _ in ev.LABELLED}) == len(ev.LABELLED))


def test_api_module_annotations() -> None:
    """Guard the bug that stopped the app from starting at all.

    `from __future__ import annotations` turns every annotation into a string.
    FastAPI resolves route signatures at import time and cannot handle the
    non-pydantic ones as ForwardRef — `UploadFile` and `Request` both break, and
    uvicorn dies before the port opens. This is an AST check rather than an
    import, because FastAPI is not needed to see the mistake.
    """
    import ast

    print("\napi module")
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    future = [n for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom) and n.module == "__future__"
              and any(a.name == "annotations" for a in n.names)]
    check("no postponed annotations in the route module", not future,
          "remove `from __future__ import annotations` from app/main.py")

    imported = {a.asname or a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}

    def annotation_names(node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def is_route(node: ast.AST) -> bool:
        for dec in getattr(node, "decorator_list", []):
            call = dec.func if isinstance(dec, ast.Call) else dec
            if (isinstance(call, ast.Attribute)
                    and isinstance(call.value, ast.Name)
                    and call.value.id == "app"
                    and call.attr in {"get", "post", "put", "delete", "patch"}):
                return True
        return False

    routes = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_route(n)]
    check("routes were found to check", len(routes) >= 4, f"{len(routes)} found")

    for route in routes:
        names: set[str] = set()
        for arg in route.args.args:
            if arg.annotation is not None:
                names |= annotation_names(arg.annotation)
        if route.returns is not None:
            names |= annotation_names(route.returns)
        builtins_ok = {"dict", "list", "str", "int", "float", "bool", "None"}
        missing = names - imported - builtins_ok - {"QueryRequest", "QueryResponse"}
        check(f"{route.name}: every annotation name is importable", not missing,
              f"missing {sorted(missing)}")

# --------------------------------------------------------------------------- #
def main() -> None:
    for test in (test_tokenize, test_bm25, test_language_detection, test_expansion, test_accent_restoration,
                 test_structure_split, test_recursive_split, test_citations,
                 test_eval_matching, test_api_module_annotations):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
