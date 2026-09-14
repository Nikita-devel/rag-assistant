"""Ingestion pipeline: raw documents -> clean text -> structured chunks -> Chroma.

Design notes (French legal / HR corpus: Code du travail + convention collective
de la métallurgie):

1. Legal text is NOT flat prose. It is a tree: Titre > Chapitre > Article.
   Blind character splitting cuts articles in half and destroys the single most
   valuable metadata field for a B2B demo — the article number the answer cites.
   So we split on structure FIRST, and only fall back to character splitting for
   sections that are still too long.

2. Chunk sizes are expressed in CHARACTERS, not tokens. For French text
   1 token ~= 3.6 characters, so the spec's "300-500 tokens" maps to roughly
   1100-1800 characters. Default: 1400 / 200 overlap.

3. Chunk IDs are deterministic (sha1 of source + normalised text). Re-running
   ingestion on an unchanged corpus is a no-op instead of creating duplicates.

Usage:
    python -m app.ingestion --reset
    python -m app.ingestion --path data/documents/convention_metallurgie.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.embeddings import embed_passages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingestion")

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class RawDocument:
    text: str
    source: str          # file name, shown to the end user as the citation
    doc_type: str        # pdf | markdown | text | html
    path: str
    page_map: list[tuple[int, int]] = field(default_factory=list)
    # page_map: [(char_offset_of_page_start, page_number)], PDFs only


@dataclass
class Chunk:
    text: str
    metadata: dict


# --------------------------------------------------------------------------- #
# 1. Loaders
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    """Fix the usual PDF/HTML garbage before anything else touches the text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                 # soft hyphen
    text = text.replace(" ", " ")                # non-breaking space
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # word split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Page footers/headers seen in Légifrance exports
    text = re.sub(r"\n\s*Page \d+( sur \d+)?\s*\n", "\n", text)
    return text.strip()


def load_pdf(path: Path) -> RawDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    page_map: list[tuple[int, int]] = []
    offset = 0
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        page_map.append((offset, page_no))
        parts.append(page_text)
        offset += len(page_text) + 1
    text = _normalise("\n".join(parts))
    if not text:
        raise ValueError(
            f"{path.name}: no extractable text. It is probably a scanned PDF — "
            f"run OCR (ocrmypdf) before ingesting."
        )
    return RawDocument(text=text, source=path.name, doc_type="pdf",
                       path=str(path), page_map=page_map)


def load_html(path: Path) -> RawDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    # Keep block structure so the splitter still sees paragraph boundaries
    text = soup.get_text(separator="\n")
    return RawDocument(text=_normalise(text), source=path.name,
                       doc_type="html", path=str(path))


def load_plain(path: Path) -> RawDocument:
    doc_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "text"
    text = _normalise(path.read_text(encoding="utf-8", errors="ignore"))
    return RawDocument(text=text, source=path.name, doc_type=doc_type, path=str(path))


def load_document(path: Path) -> RawDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".html", ".htm"}:
        return load_html(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return load_plain(path)
    raise ValueError(f"Unsupported file type: {path.name}")


# --------------------------------------------------------------------------- #
# 2. Structure-aware splitting
# --------------------------------------------------------------------------- #
# Matches: "Article L3141-3", "Article R1234-5", "Article 12.4", "Art. 3",
# and markdown headings used in the synthetic parts of the corpus.
ARTICLE_RE = re.compile(
    r"^\s*(?:"
    r"(?P<legal>Art(?:icle|\.)\s+(?:[LRD]\.?\s?)?\d+(?:[-.]\d+)*(?:\s*(?:bis|ter|quater))?)"
    r"|(?P<md>#{1,4}\s+.+)"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)

SECTION_RE = re.compile(
    r"^\s*((?:TITRE|CHAPITRE|SECTION|PARTIE|LIVRE)\s+[IVXLC\d]+.*)$",
    re.MULTILINE | re.IGNORECASE,
)


def _find_current_section(text: str, upto: int) -> str | None:
    """Last TITRE/CHAPITRE heading seen before position `upto`."""
    last = None
    for m in SECTION_RE.finditer(text, 0, upto):
        last = m.group(1).strip()
    return last


def split_by_structure(text: str) -> list[tuple[str, str | None]]:
    """Split into (block_text, article_label) on article boundaries.

    Anything before the first article (preamble) is returned with label None.
    """
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [(text, None)]

    blocks: list[tuple[str, str | None]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if len(preamble) > 100:
            blocks.append((preamble, None))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start():end].strip()
        label = (m.group("legal") or m.group("md") or "").lstrip("# ").strip()
        if block:
            blocks.append((block, label or None))
    return blocks


def _recursive_split(text: str, size: int, overlap: int) -> list[str]:
    """Character splitter with a separator hierarchy, no LangChain dependency.

    We keep this in-house on purpose: LangChain's splitter pulls a large
    dependency tree for ~40 lines of logic, and controlling the separators
    matters for legal text (we want to break on paragraph markers, not on
    the abbreviation dots inside 'art. L. 3141-1').
    """
    if len(text) <= size:
        return [text]

    separators = ["\n\n", "\n", " ; ", ". ", " "]

    def _split(chunk: str, seps: list[str]) -> list[str]:
        if len(chunk) <= size:
            return [chunk]
        if not seps:
            return [chunk[i:i + size] for i in range(0, len(chunk), size - overlap)]
        sep, rest = seps[0], seps[1:]
        pieces = chunk.split(sep)
        out: list[str] = []
        buf = ""
        for piece in pieces:
            candidate = f"{buf}{sep}{piece}" if buf else piece
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                if len(piece) > size:
                    out.extend(_split(piece, rest))
                    buf = ""
                else:
                    buf = piece
        if buf:
            out.append(buf)
        return out

    pieces = _split(text, separators)

    # Re-apply overlap between adjacent pieces so context is not lost at seams.
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    overlapped = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        overlapped.append(f"{tail} {cur}".strip())
    return overlapped


def chunk_document(doc: RawDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    index = 0

    for block, article in split_by_structure(doc.text):
        position = doc.text.find(block, cursor)
        if position == -1:
            position = cursor
        cursor = position + len(block)
        section = _find_current_section(doc.text, position)
        page = _page_for_offset(doc, position)

        for part in _recursive_split(block, settings.chunk_size, settings.chunk_overlap):
            part = part.strip()
            if len(part) < 80:          # drop headings-only / noise fragments
                continue
            chunks.append(
                Chunk(
                    text=part,
                    metadata={
                        "source": doc.source,
                        "doc_type": doc.doc_type,
                        "chunk_id": index,
                        "article": article or "",
                        "section": section or "",
                        "page": page if page is not None else -1,
                        "char_count": len(part),
                    },
                )
            )
            index += 1

    logger.info("%s -> %d chunks", doc.source, len(chunks))
    return chunks


def _page_for_offset(doc: RawDocument, offset: int) -> int | None:
    if not doc.page_map:
        return None
    page = doc.page_map[0][1]
    for start, page_no in doc.page_map:
        if start <= offset:
            page = page_no
        else:
            break
    return page


# --------------------------------------------------------------------------- #
# 3. Vector store
# --------------------------------------------------------------------------- #
def get_client():
    # Chroma 0.5.x calls posthog with a signature newer posthog releases no
    # longer accept, so every operation logs
    #   "Failed to send telemetry event ...: capture() takes 1 positional
    #    argument but 3 were given"
    # Harmless — nothing is sent either way — but it buries the real log lines
    # and looks like a broken service in a screenshot. The env var stops the
    # attempt; the logger level catches anything that still slips through.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        # We supply embeddings ourselves, so no embedding_function here.
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_uid(source: str, text: str) -> str:
    digest = hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def index_chunks(chunks: list[Chunk], collection) -> int:
    if not chunks:
        return 0
    ids = [_chunk_uid(c.metadata["source"], c.text) for c in chunks]
    # Deduplicate within the batch (identical boilerplate paragraphs happen)
    seen: set[str] = set()
    keep = [i for i, uid in enumerate(ids) if not (uid in seen or seen.add(uid))]
    ids = [ids[i] for i in keep]
    chunks = [chunks[i] for i in keep]

    texts = [c.text for c in chunks]
    vectors = embed_passages(texts)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    return len(ids)


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def ingest_file(path: Path, collection=None) -> dict:
    collection = collection or get_collection()
    doc = load_document(path)
    chunks = chunk_document(doc)
    written = index_chunks(chunks, collection)
    return {"source": doc.source, "chunks_created": written}


def ingest_directory(directory: Path | None = None, reset: bool = False) -> dict:
    directory = directory or settings.documents_path
    client = get_client()

    if reset:
        try:
            client.delete_collection(settings.collection_name)
            logger.info("Collection %s dropped", settings.collection_name)
        except Exception:
            pass
    collection = get_collection(client)

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise SystemExit(f"No supported documents found in {directory}")

    results, total = [], 0
    for path in files:
        try:
            res = ingest_file(path, collection)
        except Exception as exc:                      # one bad file must not kill the run
            logger.error("FAILED %s: %s", path.name, exc)
            results.append({"source": path.name, "error": str(exc)})
            continue
        results.append(res)
        total += res["chunks_created"]

    logger.info("Done. %d files, %d chunks, collection size = %d",
                len(files), total, collection.count())
    return {"files": results, "chunks_created": total, "collection_size": collection.count()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument("--path", type=Path, help="single file or directory")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    target = args.path or settings.documents_path
    if target.is_file():
        print(ingest_file(target))
    else:
        ingest_directory(target, reset=args.reset)


if __name__ == "__main__":
    main()
