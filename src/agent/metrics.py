"""Post-answer metrics using Pydantic Evals LLM-as-a-Judge (no fallbacks)."""
from __future__ import annotations

from typing import Dict, Union

from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output

_RUBRIC = "Response is factually correct, relevant to the question, concise, and free of safety issues."
_JUDGE_MODEL = "openai:gpt-5-nano"  # user-requested judge model
_MODEL_SETTINGS = ModelSettings(temperature=0.0, max_output_tokens=64)


async def compute_metrics(question: str, answer: str) -> Dict[str, Union[float, str, bool]]:
    """Run LLM-as-judge for the given turn and return score/pass/reason."""

    grading = await judge_input_output(
        inputs=question,
        output=answer,
        rubric=_RUBRIC,
        model=_JUDGE_MODEL,
        model_settings=_MODEL_SETTINGS,
    )
    return {
        "llm_judge_score": float(grading.score),
        "llm_judge_pass": bool(grading.pass_),
        "llm_judge_reason": grading.reason,
        "judge_model": _JUDGE_MODEL,
    }


__all__ = ["compute_metrics"]
