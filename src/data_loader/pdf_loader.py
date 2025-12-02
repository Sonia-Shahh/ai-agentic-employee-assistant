"""Async PDF text extraction utilities."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


async def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file asynchronously."""
    return await asyncio.to_thread(_read_pdf_text, path)


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


async def load_pdfs(paths: Iterable[Path]) -> List[tuple[Path, str]]:
    """Load multiple PDFs and return tuples of (path, text)."""
    tasks = [extract_pdf_text(path) for path in paths]
    texts = await asyncio.gather(*tasks)
    return list(zip(paths, texts))
