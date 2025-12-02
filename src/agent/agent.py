"""Factory for building the Pydantic AI agent used in the RAG system."""
from __future__ import annotations

from typing import Tuple
import os

from openai import AsyncOpenAI
from pydantic_ai import Agent
from serpapi import Client as SerpApiClient
import logfire

from src.config import settings
from src.vectorstore.qdrant_store import create_client

from .prompt import SYSTEM_PROMPT
from .tools import AgentDeps, register_vector_search, register_web_search


def _resolve_model(name: str) -> str:
    return name if ":" in name else f"openai:{name}"


_LOGFIRE_INITIALIZED = False


def build_agent(
    *,
    client: AsyncOpenAI | None = None,
    qdrant: object | None = None,
) -> Tuple[Agent[str, AgentDeps], AgentDeps]:
    """Construct the agent and its dependency bundle."""

    global _LOGFIRE_INITIALIZED
    # Configure Logfire once; subsequent calls are no-ops.
    if settings.logfire.token and not _LOGFIRE_INITIALIZED:
        # Disable scrubbing so tool responses/attributes aren't redacted; toggle as needed.
        logfire.configure(
            token=settings.logfire.token,
            service_name="ragbot",
            environment=os.getenv("ENVIRONMENT", "local"),
            scrubbing=False,
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)
        _LOGFIRE_INITIALIZED = True

    active_client = client or AsyncOpenAI(api_key=settings.openai.api_key)
    active_qdrant = qdrant or create_client()

    serp_client = None
    if settings.web.api_key:
        serp_client = SerpApiClient(api_key=settings.web.api_key)

    deps = AgentDeps(client=active_client, qdrant=active_qdrant, serpapi_client=serp_client)

    agent: Agent[str, AgentDeps] = Agent(
        model=_resolve_model(settings.openai.chat_model),
        system_prompt=SYSTEM_PROMPT,
        deps_type=AgentDeps,
        output_type=str,
        retries=2,
    )

    register_vector_search(agent)
    register_web_search(agent)
    return agent, deps


__all__ = ["build_agent"]
