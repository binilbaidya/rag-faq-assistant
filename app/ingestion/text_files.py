from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.schemas.faq import FAQDocument

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE)
_FENCE_PATTERN = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def documents_from_text_file(
    *,
    filename: str,
    content: bytes,
    store_id: str,
    source: str | None = None,
    doc_type: str = "faq",
    metadata: dict[str, Any] | None = None,
    chunk_words: int = 800,
    overlap_words: int = 100,
) -> list[FAQDocument]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TEXT_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    text = content.decode("utf-8", errors="replace").strip()
    chunks = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words)
    base_metadata = metadata or {}
    file_source = source or filename
    document_id = str(uuid4())
    ingested_on = datetime.now(UTC).isoformat()

    return [
        FAQDocument(
            text=chunk,
            store_id=store_id,
            source=file_source,
            doc_type=doc_type,
            metadata={
                **base_metadata,
                "document_id": document_id,
                "filename": filename,
                "ingested_on": ingested_on,
                "chunk_index": index,
                "chunk_count": len(chunks),
            },
        )
        for index, chunk in enumerate(chunks)
    ]

def chunk_text(text: str, *, chunk_words: int = 800, overlap_words: int = 100) -> list[str]:
    if not text.strip():
        return []
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than 0")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be between 0 and chunk_words - 1")

    chunks: list[str] = []
    for section in _split_sections(text):
        chunks.extend(_chunk_section(section, chunk_words=chunk_words, overlap_words=overlap_words))
    return chunks


def _split_sections(text: str) -> list[str]:
    """Each section keeps its full heading path, so a subsection still names its parent topic."""
    # A `# comment` inside a fenced code block is code, not a heading.
    fences = [match.span() for match in _FENCE_PATTERN.finditer(text)]
    matches = [
        match for match in _HEADING_PATTERN.finditer(text)
        if not any(start <= match.start() < end for start, end in fences)
    ]
    if not matches:
        return [text.strip()] if text.strip() else []

    sections: list[str] = []
    leading = text[: matches[0].start()].strip()
    if leading:
        sections.append(leading)

    trail: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(0).strip()
        title = heading.lstrip("#")
        level = len(heading) - len(title)
        trail = [entry for entry in trail if entry[0] < level] + [(level, title.strip())]
        path = " > ".join(entry[1] for entry in trail)

        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections.append(f"{'#' * level} {path}\n\n{body}")
    return sections


def _chunk_section(section: str, *, chunk_words: int, overlap_words: int) -> list[str]:
    if len(section.split()) <= chunk_words:
        return [section]

    heading, body = _split_heading(section)
    body_words = body.split()
    step = chunk_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(body_words), step):
        piece = " ".join(body_words[start:start + chunk_words]).strip()
        if piece:
            chunks.append(f"{heading}\n\n{piece}" if heading else piece)
        if start + chunk_words >= len(body_words):
            break
    return chunks


def _split_heading(section: str) -> tuple[str, str]:
    match = _HEADING_PATTERN.match(section)
    if not match:
        return "", section
    return match.group(0).strip(), section[match.end():].strip()
