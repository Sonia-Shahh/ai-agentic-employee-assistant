"""RAG ingestion pipeline."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from openai import AsyncOpenAI

from src.config import settings
from src.data_loader.audio_transcriber import transcribe_audios
from src.data_loader.pdf_loader import load_pdfs
from src.embeddings.openai_embeddings import embed_texts
from src.text_processing.chunker import chunk_text, normalize_text
from src.vectorstore.qdrant_store import StoredChunk, create_client, ensure_collection, upsert_chunks


@dataclass(slots=True)
class IngestResult:
    source: str
    filename: str
    chunks: List[str]


class RAGIngestionPipeline:
    """Coordinate loading, chunking, embedding, and storage."""

    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(api_key=settings.openai.api_key)
        self.qdrant = create_client()

    async def ingest_pdfs(self, pdf_paths: Iterable[Path]) -> List[IngestResult]:
        results: List[IngestResult] = []
        pdfs = await load_pdfs(pdf_paths)
        for path, text in pdfs:
            normalized = normalize_text(text)
            chunks = chunk_text(normalized, settings.chunks.max_tokens, settings.chunks.overlap_tokens)
            results.append(IngestResult(source="pdf", filename=path.name, chunks=chunks))
        await self._store(results)
        return results

    async def ingest_audios(self, audio_paths: Iterable[Path]) -> List[IngestResult]:
        results: List[IngestResult] = []
        transcripts = await transcribe_audios(audio_paths, client=self.client)
        for path, transcript in transcripts:
            normalized = normalize_text(transcript)
            chunks = chunk_text(normalized, settings.chunks.max_tokens, settings.chunks.overlap_tokens)
            results.append(IngestResult(source="audio", filename=path.name, chunks=chunks))
        await self._store(results)
        return results

    async def _store(self, results: List[IngestResult]) -> None:
        all_chunks: List[str] = []
        stored_chunks: List[StoredChunk] = []
        for result in results:
            for idx, chunk in enumerate(result.chunks):
                all_chunks.append(chunk)
                stored_chunks.append(
                    StoredChunk(
                        text=chunk,
                        source=result.source,
                        filename=result.filename,
                        chunk_id=idx,
                    )
                )
        if not all_chunks:
            return
        embeddings = await embed_texts(all_chunks, client=self.client)
        if not embeddings:
            return
        ensure_collection(self.qdrant, vector_size=len(embeddings[0]))
        upsert_chunks(self.qdrant, embeddings, stored_chunks)

    async def ingest_all(self, pdf_paths: Iterable[Path], audio_paths: Iterable[Path]) -> List[IngestResult]:
        pdf_task = asyncio.create_task(self.ingest_pdfs(pdf_paths))
        audio_task = asyncio.create_task(self.ingest_audios(audio_paths))
        pdf_results, audio_results = await asyncio.gather(pdf_task, audio_task)
        return pdf_results + audio_results
