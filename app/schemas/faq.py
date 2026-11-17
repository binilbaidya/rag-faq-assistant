from typing import Any

from pydantic import BaseModel, Field


class FAQDocument(BaseModel):
    text: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    source: str = Field(default="manual", min_length=1)
    doc_type: str = Field(default="faq", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    point_id: str | None = Field(
        default=None,
        description="Qdrant point id. Unrelated to metadata.document_id, which groups one upload.",
    )


class IngestFAQRequest(BaseModel):
    documents: list[FAQDocument] = Field(min_length=1, max_length=500)


class IngestFileResult(BaseModel):
    filename: str
    document_id: str | None = Field(
        default=None,
        description="Id grouping this upload's chunks; pass it to DELETE to remove the file.",
    )
    ingested_on: str | None = None
    chunks: int = 0
    error: str | None = None


class IngestFAQResponse(BaseModel):
    inserted: int
    files_processed: int = 0
    files: list[IngestFileResult] = Field(default_factory=list)


class FAQSourceSummary(BaseModel):
    document_id: str
    source: str
    filename: str
    doc_type: str = "faq"
    ingested_on: str
    chunks: int


class FAQSourceListResponse(BaseModel):
    files: dict[str, list[FAQSourceSummary]] = Field(default_factory=dict)


class DeleteFAQResponse(BaseModel):
    store_id: str
    document_id: str
    source: str
    filename: str
    deleted_chunks: int
