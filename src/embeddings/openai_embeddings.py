"""Async embeddings helper."""
from __future__ import annotations

from typing import Iterable, List

from openai import AsyncOpenAI

from src.config import settings


async def embed_texts(texts: Iterable[str], client: AsyncOpenAI | None = None) -> List[List[float]]:
    text_list = list(texts)
    if not text_list:
        return []
    active_client = client or AsyncOpenAI(api_key=settings.openai.api_key)
    response = await active_client.embeddings.create(
        model=settings.openai.embedding_model,
        input=text_list,
    )
    return [item.embedding for item in response.data]
