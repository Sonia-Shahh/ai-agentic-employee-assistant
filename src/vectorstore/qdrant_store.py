"""Qdrant vector store utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings


@dataclass(slots=True)
class StoredChunk:
    text: str
    source: str
    filename: str
    chunk_id: int


def create_client() -> QdrantClient:
    if settings.qdrant.url:
        return QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key, timeout= 60)
    return QdrantClient(location=settings.qdrant.location)


def ensure_collection(client: QdrantClient, vector_size: int, distance: qmodels.Distance = qmodels.Distance.COSINE) -> None:
    collection_name = settings.qdrant.collection_name
    exists = client.get_collections()
    if any(col.name == collection_name for col in exists.collections):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )


def upsert_chunks(
    client: QdrantClient,
    embeddings: Iterable[List[float]],
    chunks: Iterable[StoredChunk],
) -> None:
    collection_name = settings.qdrant.collection_name
    points = []
    for embedding, chunk in zip(embeddings, chunks):
        payload = {
            "text": chunk.text,
            "source": chunk.source,
            "filename": chunk.filename,
            "chunk_id": chunk.chunk_id,
        }
        point_id = uuid4().hex
        points.append(qmodels.PointStruct(id=point_id, vector=embedding, payload=payload))
    client.upsert(collection_name=collection_name, points=points)


def search_similar(
    client: QdrantClient,
    query_embedding: List[float],
    limit: int = 5,
    source_filter: Optional[str] = None,
) -> List[qmodels.ScoredPoint]:
    collection_name = settings.qdrant.collection_name
    query_filter: Optional[qmodels.Filter] = None
    if source_filter:
        query_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source_filter))])
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )
        return list(response.points)
    return client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit,
        query_filter=query_filter,
    )
