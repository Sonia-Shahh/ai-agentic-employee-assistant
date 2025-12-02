"""User-facing runner that executes the RAG agent with robust reflection/self-correction."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .agent import build_agent
from .tools import AgentDeps
from .metrics import compute_metrics


class AgentRunner:
    def __init__(self, agent=None, deps: AgentDeps | None = None):
        if agent is None and deps is None:
            agent, deps = build_agent()
        if agent is None or deps is None:
            raise ValueError("Both agent and deps must be provided together.")
        self.agent = agent
        self.deps = deps

    async def answer(
        self,
        question: str,
        limit: int = 5,
        conversation_history: Sequence[Tuple[str, str]] | None = None,
        reflection_threshold: float = 0.8,  # if score < 0.8, agent will self-reflect
        max_reflections: int = 2,  # allow multiple attempts to self-correct
    ) -> str:
        """
        Answer a question using vector search + web search, then evaluate.
        If answer is low-confidence or ambiguous, iteratively reflect and re-answer.
        """
        prompt = self._build_prompt(question, conversation_history, limit)
        answer = await self._run_agent_with_reflection(prompt, question, reflection_threshold, max_reflections)
        return answer

    async def _run_agent_with_reflection(
        self,
        prompt: str,
        question: str,
        reflection_threshold: float,
        max_reflections: int,
    ) -> str:
        """
        Internal helper: run agent and perform iterative self-reflection if necessary.
        """
        attempt = 0
        answer = await self.agent.run(prompt, deps=self.deps)
        answer_text = answer.output

        while attempt <= max_reflections:
            metrics = await compute_metrics(question, answer_text or "")
            print(f"[metrics] {metrics}")

            low_confidence = metrics["llm_judge_score"] < reflection_threshold
            ambiguous = "ambiguous" in metrics["llm_judge_reason"].lower()

            if not low_confidence and not ambiguous:
                break  # answer is good

            if attempt == max_reflections:
                print("[reflection] Max reflection attempts reached. Returning best-effort answer.")
                break

            print("[reflection] Low confidence or ambiguity detected. Attempting self-correction...")
            # Build reflection prompt
            reflection_prompt = (
                f"Review the previous answer carefully: {answer_text}\n"
                "Check for factual accuracy, completeness, clarity, and consistency using only retrieved context.\n"
                "If there are mistakes or missing details, provide a corrected, concise answer.\n"
                "If nothing relevant is available, say exactly: 'I do not know based on the provided context.'"
            )

            answer = await self.agent.run(reflection_prompt, deps=self.deps)
            answer_text = answer.output
            attempt += 1

        return answer_text

    def _build_prompt(
        self,
        question: str,
        history: Sequence[Tuple[str, str]] | None,
        limit: int,
    ) -> str:
        """Construct the conversation prompt including vector/web search instructions."""
        history_text = _format_history(history)
        return (
            f"Conversation so far:\n{history_text}\n\n"
            f"User question: {question}\n"
            f"First call `vector_search` with query=the question text and limit={limit} to fetch context. "
            "If vector search returns nothing useful, call `web_search` to gather public web snippets. "
            "Then answer concisely using only the retrieved context. "
            "If nothing relevant is returned, say 'I do not know based on the provided context.'"
        )


def _format_history(history: Sequence[Tuple[str, str]] | None) -> str:
    if not history:
        return "(no prior turns)"
    lines: List[str] = []
    for question, answer in history:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


__all__ = ["AgentRunner"]
