from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import require_superadmin_user
from app.config import get_settings
from app.dependencies import get_ai_settings, get_vector_store
from app.ingestion.text_files import documents_from_text_file
from app.ingestion.text_files import SUPPORTED_TEXT_SUFFIXES
from app.schemas.faq import (
    DeleteFAQResponse,
    FAQSourceListResponse,
    FAQSourceSummary,
    IngestFAQRequest,
    IngestFAQResponse,
    IngestFileResult,
)
from app.security import check_request


logger = logging.getLogger(__name__)

MAX_INGEST_FILES = 10
MAX_INGEST_BYTES = 5 * 1024 * 1024
KNOWN_STORE_IDS: tuple[str, ...] = (get_settings().store_id,)

router = APIRouter(
    prefix="/v1/admin/faqs",
    tags=["faqs"],
    dependencies=[Depends(require_superadmin_user)],
)


@router.post("", response_model=IngestFAQResponse, status_code=status.HTTP_201_CREATED)
async def ingest_faqs(request: Request) -> IngestFAQResponse:
    _ensure_ai_enabled_for_ingestion()
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        return await _ingest_from_multipart(request)

    try:
        payload = IngestFAQRequest.model_validate(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid FAQ ingest body: {exc}",
        ) from exc
    check_request(payload)
    inserted = get_vector_store().upsert_documents(payload.documents)
    return IngestFAQResponse(inserted=inserted)


@router.get("", response_model=FAQSourceListResponse)
async def list_ingested_faqs() -> FAQSourceListResponse:
    files_by_store = {
        store_id: _fetch_source_summaries(store_id)
        for store_id in KNOWN_STORE_IDS
    }
    return FAQSourceListResponse(files=files_by_store)


@router.delete("", response_model=DeleteFAQResponse)
async def delete_ingested_faq(
    document_id: str,
    store_id: str | None = None,
    doc_type: str = "faq",
) -> DeleteFAQResponse:
    normalized_store_id, normalized_doc_type = _parse_store_scope(store_id, doc_type)
    normalized_document_id = document_id.strip()
    if not normalized_document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id is required.",
        )

    vector_store = get_vector_store()
    entry = vector_store.find_document(
        store_id=normalized_store_id,
        doc_type=normalized_doc_type,
        document_id=normalized_document_id,
    )
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingested FAQ file found for document_id '{normalized_document_id}'.",
        )

    deleted = vector_store.delete_by_document_id(
        store_id=normalized_store_id,
        document_id=normalized_document_id,
        doc_type=normalized_doc_type,
    )
    return DeleteFAQResponse(
        store_id=normalized_store_id,
        document_id=normalized_document_id,
        source=str(entry["source"]),
        filename=str(entry["filename"]),
        deleted_chunks=deleted,
    )


def _parse_store_scope(store_id: str | None, doc_type: str) -> tuple[str, str]:
    return _require_known_store(store_id), doc_type.strip() or "faq"


def _require_known_store(store_id: str | None) -> str:
    normalized = (store_id or "").strip() or KNOWN_STORE_IDS[0]
    if normalized not in KNOWN_STORE_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown store_id {normalized!r}. Expected one of: {', '.join(KNOWN_STORE_IDS)}.",
        )
    return normalized


def _fetch_source_summaries(store_id: str) -> list[FAQSourceSummary]:
    rows = get_vector_store().list_sources(store_id=store_id)
    return [FAQSourceSummary.model_validate(row) for row in rows]


def _ensure_ai_enabled_for_ingestion() -> None:
    if get_ai_settings().is_ai_enabled():
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="AI features are disabled. FAQ ingestion is unavailable.",
    )


async def _ingest_from_multipart(request: Request) -> IngestFAQResponse:
    form = await request.form()
    uploads = _collect_uploads(form)
    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing text file(s). Use the files form field (one or more).",
        )
    if len(uploads) > MAX_INGEST_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {MAX_INGEST_FILES} files per request.",
        )

    store_id = _require_known_store(_form_str(form.get("store_id")) or "")

    shared_source = _form_str(form.get("source")) if len(uploads) == 1 else None
    doc_type = _form_str(form.get("doc_type")) or "faq"
    metadata = _metadata(form.get("metadata"))
    chunk_words = _form_int(form.get("chunk_words"), 800)
    overlap_words = _form_int(form.get("overlap_words"), 100)

    file_results: list[IngestFileResult] = []
    total_inserted = 0

    for upload in uploads:
        filename = getattr(upload, "filename", "upload.txt") or "upload.txt"
        try:
            if Path(filename).suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
                raise ValueError(f"Unsupported file type: {Path(filename).suffix or 'unknown'}")
            content = await upload.read()
            if len(content) > MAX_INGEST_BYTES:
                raise ValueError(
                    f"File is larger than the {MAX_INGEST_BYTES // (1024 * 1024)}MB ingest limit."
                )
            documents = documents_from_text_file(
                filename=filename,
                content=content,
                store_id=store_id,
                source=shared_source or filename,
                doc_type=doc_type,
                metadata=metadata,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
            if not documents:
                file_results.append(
                    IngestFileResult(filename=filename, error="Text file has no ingestible content.")
                )
                continue
            inserted = get_vector_store().replace_source(documents)
            total_inserted += inserted
            ingest_document_id = str(documents[0].metadata.get("document_id") or "")
            ingest_timestamp = str(documents[0].metadata.get("ingested_on") or "")
            file_results.append(
                IngestFileResult(
                    filename=filename,
                    document_id=ingest_document_id or None,
                    ingested_on=ingest_timestamp or None,
                    chunks=inserted,
                )
            )
        except ValueError as exc:
            file_results.append(IngestFileResult(filename=filename, error=str(exc)))
        except Exception as exc:
            logger.exception("Failed to ingest %s", filename)
            file_results.append(IngestFileResult(filename=filename, error=f"Ingest failed: {exc}"))

    if total_inserted == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "No files were ingested.",
                "files": [result.model_dump() for result in file_results],
            },
        )

    return IngestFAQResponse(
        inserted=total_inserted,
        files_processed=len(file_results),
        files=file_results,
    )


def _collect_uploads(form: Any) -> list[Any]:
    return [item for item in form.getlist("files") if hasattr(item, "read")]


def _form_str(value: Any) -> str | None:
    if value is None or hasattr(value, "read"):
        return None
    text = str(value).strip()
    return text or None


def _form_int(value: Any, default: int) -> int:
    text = _form_str(value)
    if text is None:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid integer: {text}") from exc


def _metadata(value: Any) -> dict[str, Any]:
    text = _form_str(value)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="metadata must be valid JSON.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="metadata must be a JSON object.")
    return data
