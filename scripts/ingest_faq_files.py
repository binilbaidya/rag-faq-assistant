from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.database.vector_store import FAQVectorStore
from app.ingestion.text_files import SUPPORTED_TEXT_SUFFIXES, documents_from_text_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk, embed and upsert .md/.txt files into Qdrant.")
    parser.add_argument("path")
    parser.add_argument("--store-id", default=get_settings().store_id)
    parser.add_argument("--doc-type", default="faq")
    parser.add_argument("--chunk-words", type=int, default=800)
    parser.add_argument("--overlap-words", type=int, default=100)
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first (full re-index)")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    files = _files(root)
    if not files:
        raise SystemExit(f"No ingestible files found under {root}.")
    vector_store = FAQVectorStore()
    vector_store.initialize_collection(recreate=args.recreate)
    total = 0

    for file_path in files:
        documents = documents_from_text_file(
            filename=file_path.name,
            content=file_path.read_bytes(),
            store_id=args.store_id,
            doc_type=args.doc_type,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        total += vector_store.replace_source(documents)
        print(f"ingested {len(documents)} chunks from {file_path}")

    print(f"done: {total} chunks ingested")


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES else []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_TEXT_SUFFIXES
    )


if __name__ == "__main__":
    main()
