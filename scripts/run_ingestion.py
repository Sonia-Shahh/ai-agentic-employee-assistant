"""CLI for ingesting PDFs and audio files."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.rag.pipeline import RAGIngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs and audio files into Qdrant.")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing PDF files")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory containing audio files")
    parser.add_argument("--collection", type=str, default=settings.qdrant.collection_name, help="Qdrant collection name")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings.qdrant.collection_name = args.collection
    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
    audio_paths = sorted(args.audio_dir.glob("*"))
    pipeline = RAGIngestionPipeline()
    await pipeline.ingest_all(pdf_paths, audio_paths)
    print(f"Ingested {len(pdf_paths)} PDFs and {len(audio_paths)} audio files into collection '{settings.qdrant.collection_name}'.")


if __name__ == "__main__":
    asyncio.run(main())
