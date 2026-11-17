from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.schemas.faq import FAQDocument


logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")

_STOPWORDS = frozenset(
    """a an the and or but if is are was were be been being do does did doing have has had
    i me my we our you your it its this that these those they them their he she his her
    to for of in on at by with from as about into over after before between out up down
    can could will would shall should may might must
    what when where which who whom why how there here any some all each both more most other
    such no nor not only own same so than too very just also"""
    .split()
)


class SearchResult(BaseModel):
    text: str
    score: float
    vector_score: float
    lexical_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class FAQVectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = self.settings.qdrant_collection
        self.client = client or QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
        )
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            # Send raw text, not tiktoken ids, so non-OpenAI compatible endpoints work.
            check_embedding_ctx_length=False,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=1,
        )
        self._initialized = False

    def initialize_collection(self, recreate: bool = False) -> None:
        # Checked once per process: otherwise every search pays 5 extra Qdrant round trips.
        if self._initialized and not recreate:
            return
        exists = self._collection_exists()
        if exists and recreate:
            logger.warning("Recreating Qdrant collection %s", self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.settings.vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

        self._ensure_payload_indexes()
        self._initialized = True

    def upsert_documents(self, documents: Iterable[FAQDocument]) -> int:
        docs = list(documents)
        if not docs:
            return 0

        self.initialize_collection()
        texts = [doc.text.strip() for doc in docs]
        vectors = self.embeddings.embed_documents(texts)

        # One upsert call is one deletable upload; without this the admin list and delete
        # endpoints skip the chunks entirely, since both group on document_id.
        batch_id = str(uuid4())
        ingested_on = datetime.now(UTC).isoformat()

        points: list[models.PointStruct] = []
        for doc, vector in zip(docs, vectors, strict=True):
            metadata = dict(doc.metadata)
            metadata.setdefault("document_id", batch_id)
            metadata.setdefault("ingested_on", ingested_on)
            payload = {
                **metadata,
                "text": doc.text.strip(),
                "store_id": doc.store_id,
                "source": doc.source,
                "doc_type": doc.doc_type,
            }
            points.append(
                models.PointStruct(
                    id=doc.point_id or str(uuid4()),
                    vector={"dense": vector},
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def replace_source(self, documents: list[FAQDocument]) -> int:
        """Upsert one file's chunks, then drop earlier uploads of the same file.

        New chunks go in first, so a failed embed never leaves the file missing from the store.
        """
        inserted = self.upsert_documents(documents)
        if not inserted:
            return 0
        first = documents[0]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="store_id", match=models.MatchValue(value=first.store_id)),
                        models.FieldCondition(key="source", match=models.MatchValue(value=first.source)),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=str(first.metadata["document_id"]))
                        )
                    ],
                )
            ),
        )
        return inserted

    def list_sources(
        self,
        *,
        store_id: str,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Distinct ingested files for a store, grouped by document_id (one row per upload)."""
        self.initialize_collection()
        qdrant_filter = self._build_filter(store_id=store_id, doc_type=doc_type)
        summaries: dict[str, dict[str, Any]] = {}
        offset: Any = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=256,
                offset=offset,
                with_payload=["document_id", "source", "filename", "doc_type", "ingested_on"],
                with_vectors=False,
            )
            for point in records:
                payload = dict(point.payload or {})
                document_id = str(payload.get("document_id") or "").strip()
                if not document_id:
                    continue

                source = str(payload.get("source") or "").strip()
                if document_id not in summaries:
                    summaries[document_id] = {
                        "document_id": document_id,
                        "source": source,
                        "filename": str(payload.get("filename") or source),
                        "doc_type": str(payload.get("doc_type") or "faq"),
                        "ingested_on": str(payload.get("ingested_on") or ""),
                        "chunks": 0,
                    }
                summaries[document_id]["chunks"] += 1
            if offset is None:
                break

        return sorted(
            summaries.values(),
            key=lambda item: str(item.get("ingested_on") or ""),
            reverse=True,
        )

    def find_document(
        self,
        *,
        store_id: str,
        document_id: str,
        doc_type: str = "faq",
    ) -> dict[str, Any] | None:
        normalized_id = document_id.strip()
        if not normalized_id:
            return None
        for row in self.list_sources(store_id=store_id, doc_type=doc_type):
            if row["document_id"] == normalized_id:
                return row
        return None

    def delete_by_document_id(
        self,
        *,
        store_id: str,
        document_id: str,
        doc_type: str = "faq",
    ) -> int:
        normalized_id = document_id.strip()
        if not normalized_id:
            return 0

        self.initialize_collection()
        qdrant_filter = self._build_filter(
            store_id=store_id,
            doc_type=doc_type,
            document_id=normalized_id,
        )
        if qdrant_filter is None:
            return 0

        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_filter,
            exact=True,
        )
        matched = int(getattr(count_result, "count", 0) or 0)
        if matched == 0:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=qdrant_filter),
        )
        return matched

    def hybrid_context_search(
        self,
        query: str,
        *,
        store_id: str | None = None,
        doc_type: str = "faq",
        limit: int = 5,
        min_score: float = 0.05,
    ) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        self.initialize_collection()
        vector = self.embeddings.embed_query(normalized_query)
        fetch_limit = max(limit * 8, 40)
        qdrant_filter = self._build_filter(store_id=store_id, doc_type=doc_type)
        raw_points = self._query_points(
            vector=vector,
            qdrant_filter=qdrant_filter,
            limit=fetch_limit,
        )

        results: list[SearchResult] = []
        for point in raw_points:
            payload = dict(point.payload or {})
            text = str(payload.get("text") or "").strip()
            if not text:
                continue

            vector_score = float(point.score or 0.0)
            lexical_score = _lexical_overlap_score(normalized_query, text)
            combined_score = (0.72 * _clamp(vector_score)) + (0.28 * lexical_score)
            if combined_score < min_score:
                continue

            metadata = {key: value for key, value in payload.items() if key != "text"}
            results.append(
                SearchResult(
                    text=text,
                    score=round(combined_score, 6),
                    vector_score=round(vector_score, 6),
                    lexical_score=round(lexical_score, 6),
                    metadata=metadata,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    def _ensure_payload_indexes(self) -> None:
        for field_name in ("store_id", "source", "doc_type", "document_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug("Payload index %s already exists or could not be created: %s", field_name, exc)

    def _build_filter(
        self,
        *,
        store_id: str | None,
        doc_type: str | None,
        source: str | None = None,
        document_id: str | None = None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if store_id:
            conditions.append(
                models.FieldCondition(key="store_id", match=models.MatchValue(value=store_id))
            )
        if doc_type:
            conditions.append(
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))
            )
        if source:
            conditions.append(
                models.FieldCondition(key="source", match=models.MatchValue(value=source.strip()))
            )
        if document_id:
            conditions.append(
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id.strip()))
            )
        if not conditions:
            return None
        return models.Filter(must=conditions)

    def _query_points(
        self,
        *,
        vector: list[float],
        qdrant_filter: models.Filter | None,
        limit: int,
    ) -> list[Any]:
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using="dense",
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )
            return list(response.points)
        except AttributeError:
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", vector),
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                )
            )


def _tokens(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(value)}


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxzh":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _stemmed(tokens: set[str]) -> set[str]:
    return {_singular(token) for token in tokens}


def _lexical_overlap_score(query: str, document: str) -> float:
    """Fraction of the question's content words the document actually contains."""
    tokens = _tokens(query)
    query_tokens = _stemmed(tokens - _STOPWORDS) or _stemmed(tokens)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _stemmed(_tokens(document))) / len(query_tokens)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))

