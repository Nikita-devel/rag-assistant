"""Regenerate app/accents_fr.py from the ingested corpus.

Run it after changing the corpus. The map is derived from the documents rather
than written by hand so it stays in the vocabulary the retriever actually
indexes — and so the safety filter is applied consistently instead of by memory.

    python scripts/build_accent_map.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "accents_fr.py"
WORD = re.compile(r"[A-Za-zÀ-ÿ'-]{3,}")

# Thresholds. Each one exists to keep a specific kind of mistake out.
DOMINANCE = 0.90     # the accented form must own its group; near-ties are skipped
BARE_RATIO = 0.15    # if the unaccented spelling is itself common, do not rewrite
MIN_COUNT = 12       # rare words are not worth the risk of a wrong rewrite


def fold(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", word)
                   if not unicodedata.combining(c))


def main() -> None:
    counts: Counter = Counter()
    files = sorted(settings.documents_path.glob("*.md"))
    if not files:
        raise SystemExit(f"no documents in {settings.documents_path}")
    for path in files:
        for word in WORD.findall(path.read_text(encoding="utf-8")):
            counts[word.lower()] += 1

    groups: dict[str, Counter] = {}
    for word, n in counts.items():
        folded = fold(word)
        if folded != word:
            groups.setdefault(folded, Counter())[word] += n

    rows, rejected = [], 0
    for folded, variants in groups.items():
        best, best_n = variants.most_common(1)[0]
        if best_n / sum(variants.values()) < DOMINANCE:
            rejected += 1
            continue
        if counts.get(folded, 0) > best_n * BARE_RATIO:
            rejected += 1           # "sur"/"sûr", "cote"/"côte", "mur"/"mûr" …
            continue
        if best_n < MIN_COUNT:
            continue
        rows.append((folded, best, best_n))

    rows.sort(key=lambda r: -r[2])

    header = '''"""Accent restoration for French queries typed without diacritics.

GENERATED, not hand-written: built from the ingested corpus itself by
scripts/build_accent_map.py. Every entry is a folded spelling that maps to one
accented word which dominates its group in the corpus, and whose bare spelling
is not itself a common word there. That filter is what keeps "sur", "cote",
"mur", "jeune" and "marche" out — rewriting those would break more queries than
it fixes.

Why it exists: asked "Quel delai pour prevenir un salarie d'une astreinte ?"
(no accents, as people type on a phone or a foreign keyboard) the correct
article fell from rank 1 to rank 5. BM25 folds accents on both sides so it did
not care, but the embedder and the cross-encoder see "delai" and "délai" as
different words.
"""
'''
    OUT.write_text(
        header + "\nACCENTS = {\n"
        + "".join(f'    "{f}": "{a}",\n' for f, a, _ in rows) + "}\n",
        encoding="utf-8")

    print(f"{len(files)} files, {len(groups)} accented groups, "
          f"{rejected} rejected as unsafe, {len(rows)} written")
    print("top 15:", ", ".join(f"{f}->{a}" for f, a, _ in rows[:15]))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
