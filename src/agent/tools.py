"""Tool registrations for the RAG agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from serpapi import Client as SerpApiClient

from src.embeddings.openai_embeddings import embed_texts
from src.vectorstore.qdrant_store import search_similar


@dataclass(slots=True)
class AgentDeps:
    """Dependencies injected into tools for each agent run."""

    client: AsyncOpenAI
    qdrant: object  # QdrantClient, kept loose to avoid importing heavy types at import time
    serpapi_client: Optional[SerpApiClient] = None


def register_vector_search(agent: Agent[str, AgentDeps]) -> None:
    """Attach the vector_search tool to the given agent."""

    @agent.tool(name="vector_search", description="Search Qdrant for text chunks relevant to a query.")
    async def vector_search(ctx: RunContext[AgentDeps], query: str, limit: int = 5) -> str:
        [query_embedding] = await embed_texts([query], client=ctx.deps.client)
        results = search_similar(ctx.deps.qdrant, query_embedding, limit=limit)
        if not results:
            return "No results found."

        lines: List[str] = []
        for hit in results:
            payload = hit.payload or {}
            text = str(payload.get("text", ""))
            source = str(payload.get("source", ""))
            filename = str(payload.get("filename", ""))
            chunk_id = int(payload.get("chunk_id", 0))
            score = float(getattr(hit, "score", 0.0) or 0.0)
            lines.append(f"[source={source} file={filename} chunk={chunk_id} score={score:.4f}] {text}")
        return "\n".join(lines)


__all__ = ["AgentDeps", "register_vector_search"]


def register_web_search(agent: Agent[str, AgentDeps]) -> None:
    """Attach a SerpAPI-backed web search tool for open-domain queries."""

    @agent.tool(name="web_search", description="Search the public web via SerpAPI when KB context is missing.")
    async def web_search(ctx: RunContext[AgentDeps], query: str, num_results: int = 5) -> str:
        client = ctx.deps.serpapi_client
        if client is None:
            return "Web search unavailable: SERPAPI_API_KEY not set."

        def _search():
            return client.search(
                {
                    "engine": "google",
                    "q": query,
                    "num": num_results,
                    "hl": "en",
                    "gl": "us",
                    "google_domain": "google.com",
                    "safe": "active",
                }
            )

        try:
            data = await asyncio.to_thread(_search)
        except Exception as exc:  # surface network/key/other issues
            return f"SerpAPI exception: {exc}"

        # Return raw stringified payload; no post-processing.
        if hasattr(data, "as_dict"):
            try:
                data = data.as_dict()
            except Exception:
                pass
        return str(data)

__all__ = ["AgentDeps", "register_vector_search", "register_web_search"]
