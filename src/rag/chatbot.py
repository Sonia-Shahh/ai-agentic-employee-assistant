"""Compatibility wrapper that exposes the agent runner via the old RAGChatbot API."""
from __future__ import annotations

from typing import Sequence, Tuple

from src.agent.runner import AgentRunner


class RAGChatbot:
    """Maintains the existing interface while delegating to the Pydantic AI agent."""

    def __init__(self):
        self.runner = AgentRunner()

    async def answer(
        self,
        question: str,
        limit: int = 5,
        conversation_history: Sequence[Tuple[str, str]] | None = None,
    ) -> str:
        return await self.runner.answer(question, limit=limit, conversation_history=conversation_history)


__all__ = ["RAGChatbot"]
