"""Load approved reference documents from knowledge/ and split them into chunks.

Every document must start with a front-matter block giving its provenance:

    ---
    title: ...
    publisher: ...
    url: https://...
    published: 2024-05-01      (or reviewed: ...)
    version: ...
    ---

Documents missing required provenance are skipped and reported, never silently used.
"""
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("title", "publisher", "url", "version")
DATE_FIELDS = ("published", "reviewed")
SUPPORTED_SUFFIXES = {".md", ".txt"}
SKIP_NAMES = {"readme.md", "readme.txt"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    publisher: str
    url: str
    published: str
    reviewed: str
    version: str
    text: str

    def citation(self):
        dates = []
        if self.published:
            dates.append(f"published {self.published}")
        if self.reviewed:
            dates.append(f"reviewed {self.reviewed}")
        return f"{self.title}, {self.publisher} (version {self.version}; {', '.join(dates)})"


@dataclass(frozen=True)
class Chunk:
    doc: Document
    index: int
    text: str


def parse_front_matter(raw):
    """Split '---' front matter from the body. Raises ValueError if absent or invalid."""
    lines = raw.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing front matter (file must start with '---')")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("front matter is not closed with '---'")
    meta = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, "\n".join(lines[end + 1:]).strip()


def _build_document(path, raw):
    meta, body = parse_front_matter(raw)
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    if not any(meta.get(f) for f in DATE_FIELDS):
        raise ValueError("needs a 'published' or 'reviewed' date")
    if not re.match(r"https?://", meta["url"]):
        raise ValueError("url must start with http:// or https://")
    if not body:
        raise ValueError("document body is empty")
    return Document(
        doc_id=path.stem,
        title=meta["title"],
        publisher=meta["publisher"],
        url=meta["url"],
        published=meta.get("published", ""),
        reviewed=meta.get("reviewed", ""),
        version=meta["version"],
        text=body,
    )


def load_documents(folder):
    """Return (documents, problems). problems is a list of 'filename: reason' strings."""
    docs, problems = [], []
    folder = Path(folder)
    if not folder.is_dir():
        return docs, [f"{folder}: knowledge folder not found"]
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.name.lower() in SKIP_NAMES:
            continue
        try:
            docs.append(_build_document(path, path.read_text(encoding="utf-8")))
        except (ValueError, OSError, UnicodeDecodeError) as error:
            problems.append(f"{path.name}: {error}")
    return docs, problems


def chunk_document(doc, max_words=110):
    """Split a document into paragraph-aligned chunks of roughly max_words words."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    pieces = []
    for p in paragraphs:
        words = p.split()
        for i in range(0, len(words), max_words):
            pieces.append(" ".join(words[i:i + max_words]))

    chunks, current, count = [], [], 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        chunks.append(" ".join(current))
    return [Chunk(doc, i, text) for i, text in enumerate(chunks)]


def load_chunks(folder, max_words=110):
    """Return (chunks, documents, problems) for everything approved in the folder."""
    docs, problems = load_documents(folder)
    chunks = [c for d in docs for c in chunk_document(d, max_words)]
    return chunks, docs, problems
