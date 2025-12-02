"""Configuration management for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class OpenAISettings:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"))
    transcription_model: str = field(default_factory=lambda: os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe"))

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it in your .env file.")


@dataclass(slots=True)
class ChunkSettings:
    max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", 400))
    overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", 60))


@dataclass(slots=True)
class QdrantSettings:
    collection_name: str = os.getenv("QDRANT_COLLECTION", "ragbot-collection")
    location: str = os.getenv("QDRANT_LOCATION", ":memory:")
    url: Optional[str] = os.getenv("QDRANT_URL")
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")


@dataclass(slots=True)
class WebSearchSettings:
    api_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))


@dataclass(slots=True)
class LogfireSettings:
    token: str = field(default_factory=lambda: os.getenv(
        "LOGFIRE_TOKEN",
        "",
    ))


@dataclass(slots=True)
class AppSettings:
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    chunks: ChunkSettings = field(default_factory=ChunkSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    web: WebSearchSettings = field(default_factory=WebSearchSettings)
    logfire: LogfireSettings = field(default_factory=LogfireSettings)
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))

    def validate(self) -> None:
        self.openai.validate()


settings = AppSettings()
