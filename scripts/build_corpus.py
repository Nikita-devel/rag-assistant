"""Build the demo corpus from the official Code du travail dataset.

Source: https://github.com/SocialGouv/legi-data (French government open data,
daily extraction of the LEGI database). One 54 MB JSON holds the whole code as
a tree: sections -> subsections -> articles.

This avoids clicking through Légifrance by hand: we select thematic blocks by
article-number prefix and write one Markdown file per theme, already shaped the
way app/ingestion.py wants (TITRE/CHAPITRE headings + "Article LXXXX-Y").

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --themes conges,securite
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

# The output directory comes from configuration, NOT from the repository layout.
# Hardcoding BASE_DIR/"data"/"documents" here wrote the corpus to /app/data in
# the container while ingestion read DOCUMENTS_PATH=/data — the two never met,
# and the container crash-looped building a corpus nobody would ever read.
OUT_DIR = settings.documents_path
# Downloaded JSON dumps sit beside the corpus so a volume keeps them too; they
# are 67 MB and re-downloading them on every container restart is rude.
CACHE_DIR = OUT_DIR.parent
CACHE = CACHE_DIR / "LEGITEXT000006072050.json"
URL = ("https://raw.githubusercontent.com/SocialGouv/legi-data/"
       "master/data/LEGITEXT000006072050.json")

# Convention collective nationale de la métallurgie (IDCC 3248, in force since
# 2024-01-01). Source: https://github.com/SocialGouv/kali-data — JSON dump of
# the DILA/KALI database. Different schema from LEGI: articles carry HTML in
# `content`, and the hierarchy title lives on the parent section.
CCN_ID = "KALICONT000046993250"
CCN_CACHE = CACHE_DIR / f"{CCN_ID}.json"
CCN_URL = (f"https://raw.githubusercontent.com/SocialGouv/kali-data/"
           f"master/data/{CCN_ID}.json")
# Only the base text. "Textes Attachés" / "Textes Salaires" are regional and
# sectoral annexes — 5000 extra articles of noise for a demo corpus.
CCN_ROOT_SECTION = "Texte de base"

# Thematic slices relevant to an industrial SME (tôlerie / chaudronnerie).
# Keys are file names; values are article-number prefixes.
THEMES: dict[str, dict] = {
    "conges_payes": {
        "title": "Congés payés et absences",
        "prefixes": ["L3141", "L3142", "L3143", "R3141", "D3141"],
    },
    "duree_du_travail": {
        "title": "Durée du travail, heures supplémentaires et repos",
        "prefixes": ["L3121", "L3122", "L3131", "L3132", "L3133", "R3121"],
    },
    "contrat_de_travail": {
        "title": "Formation et exécution du contrat de travail",
        "prefixes": ["L1221", "L1222", "L1223", "L1224", "L1225", "L1226"],
    },
    "cdd_interim": {
        "title": "Contrat à durée déterminée et travail temporaire",
        "prefixes": ["L1242", "L1243", "L1244", "L1245", "L1246", "L1251"],
    },
    "rupture_contrat": {
        "title": "Rupture du contrat de travail et licenciement",
        "prefixes": ["L1231", "L1232", "L1233", "L1234", "L1235", "L1237"],
    },
    "sante_securite": {
        "title": "Santé et sécurité au travail — obligations de l'employeur",
        "prefixes": ["L4121", "L4122", "L4131", "L4132", "L4141", "L4154",
                     "R4121", "R4141"],
    },
    "equipements_protection": {
        "title": "Équipements de travail et moyens de protection",
        "prefixes": ["L4321", "L4322", "L4323", "R4321", "R4323"],
    },
    "risques_chimiques_bruit": {
        "title": "Risques chimiques, bruit et vibrations",
        "prefixes": ["R4412", "R4431", "R4432", "R4441", "R4443"],
    },
    "cse_representation": {
        "title": "Comité social et économique",
        "prefixes": ["L2311", "L2312", "L2313", "L2314", "L2315", "L2316"],
    },
    "salaire_remuneration": {
        "title": "Salaire, SMIC et paiement de la rémunération",
        "prefixes": ["L3231", "L3241", "L3242", "L3243", "L3245", "L3251"],
    },
    "apprentissage": {
        "title": "Apprentissage et contrat de professionnalisation",
        "prefixes": ["L6221", "L6222", "L6223", "L6224", "L6225", "L6226"],
    },
    "travail_de_nuit": {
        "title": "Travail de nuit et travail posté",
        "prefixes": ["L3122", "R3122"],
    },
    "egalite_discrimination": {
        "title": "Égalité professionnelle et non-discrimination",
        "prefixes": ["L1132", "L1133", "L1134", "L1142", "L1143", "L1144"],
    },
    "harcelement": {
        "title": "Harcèlement moral et sexuel",
        "prefixes": ["L1152", "L1153", "L1154", "L1155"],
    },
    "teletravail_deplacement": {
        "title": "Télétravail et déplacements professionnels",
        "prefixes": ["L1222-9", "L1222-10", "L1222-11", "L3261"],
    },
}

NUM_RE = re.compile(r"^([LRD])\.?\s?(\d+)(?:-(\d+))*", re.IGNORECASE)


def download() -> dict:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        print(f"Downloading Code du travail dataset (~54 MB) …")
        urllib.request.urlretrieve(URL, CACHE)
    return json.loads(CACHE.read_text(encoding="utf-8"))


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def walk(node: dict, path: tuple[str, ...] = ()):
    """Yield (article_data, section_path) for every in-force article."""
    data = node.get("data", {})
    ntype = node.get("type")
    if ntype == "article":
        if data.get("etat") == "VIGUEUR" and data.get("texte"):
            yield data, path
        return
    title = clean(data.get("title", ""))
    new_path = path + (title,) if title else path
    for child in node.get("children", []):
        yield from walk(child, new_path)


def article_prefix(num: str) -> str:
    """'L3141-3' -> 'L3141'"""
    return num.split("-")[0].replace(".", "").replace(" ", "").upper()


def build(themes: list[str]) -> None:
    root = download()
    print("Indexing articles …")
    collected: list[tuple[dict, tuple[str, ...]]] = list(walk(root))
    print(f"{len(collected)} in-force articles in the code")

    by_prefix: dict[str, list] = {}
    for data, path in collected:
        by_prefix.setdefault(article_prefix(data.get("num", "")), []).append((data, path))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_articles = 0

    for name in themes:
        spec = THEMES[name]
        seen: set[str] = set()
        entries: list[tuple[dict, tuple[str, ...]]] = []
        for prefix in spec["prefixes"]:
            key = article_prefix(prefix)
            for data, path in by_prefix.get(key, []):
                num = data.get("num", "")
                # exact-article prefixes like "L1222-9" must match exactly
                if "-" in prefix and num.upper() != prefix.upper():
                    continue
                if num in seen:
                    continue
                seen.add(num)
                entries.append((data, path))

        if not entries:
            print(f"  !! {name}: no articles matched, skipped")
            continue

        entries.sort(key=lambda e: _sort_key(e[0].get("num", "")))

        lines = [f"# Code du travail — {spec['title']}", ""]
        last_section = None
        for data, path in entries:
            section = path[-1] if path else ""
            if section and section != last_section:
                lines += ["", f"CHAPITRE : {section}", ""]
                last_section = section
            lines.append(f"Article {data['num']}")
            lines.append(clean(data["texte"]))
            if data.get("nota"):
                lines.append(f"NOTA : {clean(data['nota'])}")
            lines.append("")

        out = OUT_DIR / f"{name}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        size_kb = out.stat().st_size / 1024
        print(f"  {out.name:<34} {len(entries):>4} articles  {size_kb:>7.1f} KB")
        total_articles += len(entries)

    print(f"\n{len(themes)} files, {total_articles} articles -> {OUT_DIR}")


def _sort_key(num: str):
    parts = re.split(r"[-.]", num)
    key = []
    for p in parts:
        m = re.match(r"^([A-Za-z]*)(\d*)$", p)
        key.append((m.group(1) if m else p, int(m.group(2)) if m and m.group(2) else 0))
    return key


# --------------------------------------------------------------------------- #
# Convention collective de la métallurgie (IDCC 3248)
# --------------------------------------------------------------------------- #
def html_to_text(raw: str) -> str:
    """KALI stores article bodies as HTML fragments (tables included)."""
    if not raw:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", raw)
    s = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n\n", s)
    s = re.sub(r"</t[dh]>", " | ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s).replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)[:48]


def ccn_articles(node: dict, path: tuple[str, ...] = ()):
    data = node.get("data", {})
    if node.get("type") == "article":
        if str(data.get("etat", "")).startswith("VIGUEUR") and data.get("content"):
            yield data, path
        return
    title = clean(data.get("title", ""))
    new_path = path + (title,) if title else path
    for child in node.get("children", []):
        yield from ccn_articles(child, new_path)


def build_convention() -> None:
    CCN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CCN_CACHE.exists():
        print("Downloading convention collective métallurgie (~13 MB) …")
        urllib.request.urlretrieve(CCN_URL, CCN_CACHE)
    root = json.loads(CCN_CACHE.read_text(encoding="utf-8"))

    base = next(
        (c for c in root.get("children", [])
         if CCN_ROOT_SECTION.lower() in clean(c.get("data", {}).get("title", "")).lower()),
        None,
    )
    if base is None:
        raise SystemExit("Base text section not found in the convention dump")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {}
    for data, path in ccn_articles(base):
        # path[0] is "Texte de base : ...", path[1] is the Titre
        titre = path[1] if len(path) > 1 else "dispositions_generales"
        groups.setdefault(titre, []).append((data, path))

    total = 0
    for titre, entries in groups.items():
        lines = [f"# Convention collective de la métallurgie (IDCC 3248) — {titre}", ""]
        last_chapter = None
        for data, path in entries:
            chapter = path[2] if len(path) > 2 else ""
            if chapter and chapter != last_chapter:
                lines += ["", f"CHAPITRE : {chapter}", ""]
                last_chapter = chapter
            num = clean(str(data.get("num") or "")) or data.get("cid", "")
            surtitre = clean(str(data.get("surtitre") or ""))
            lines.append(f"Article {num}")
            if surtitre:
                lines.append(surtitre)
            lines.append(html_to_text(data["content"]))
            lines.append("")

        out = OUT_DIR / f"ccn_metallurgie_{slugify(titre)}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  {out.name:<46} {len(entries):>4} articles  "
              f"{out.stat().st_size / 1024:>7.1f} KB")
        total += len(entries)

    print(f"\n{len(groups)} convention files, {total} articles -> {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--themes", help="comma-separated subset of Code du travail themes")
    parser.add_argument("--list", action="store_true", help="list available themes")
    parser.add_argument("--skip-code", action="store_true", help="skip the Code du travail")
    parser.add_argument("--skip-convention", action="store_true",
                        help="skip the convention collective métallurgie")
    args = parser.parse_args()

    if args.list:
        for k, v in THEMES.items():
            print(f"{k:<28} {v['title']}")
        return

    if not args.skip_code:
        themes = args.themes.split(",") if args.themes else list(THEMES)
        unknown = [t for t in themes if t not in THEMES]
        if unknown:
            raise SystemExit(f"Unknown themes: {unknown}. Use --list.")
        build(themes)

    if not args.skip_convention:
        print()
        build_convention()

    print("\nNext: python -m app.ingestion --reset")


if __name__ == "__main__":
    main()
