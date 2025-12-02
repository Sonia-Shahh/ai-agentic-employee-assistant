"""CLI for querying the chatbot."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.rag.chatbot import RAGChatbot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions against the Qdrant-backed RAG chatbot.")
    parser.add_argument("questions", nargs="*", help="Questions to ask")
    parser.add_argument("--file", type=Path, help="Optional file with one question per line")
    parser.add_argument("--collection", type=str, default=settings.qdrant.collection_name, help="Qdrant collection name")
    parser.add_argument("--limit", type=int, default=5, help="Number of context chunks to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive turn-by-turn chat loop")
    return parser.parse_args()


def load_questions(args: argparse.Namespace) -> List[str]:
    questions: List[str] = []
    if args.file:
        questions.extend([line.strip() for line in args.file.read_text().splitlines() if line.strip()])
    questions.extend(args.questions)
    return questions


async def main() -> None:
    args = parse_args()
    settings.qdrant.collection_name = args.collection
    questions = load_questions(args)
    chatbot = RAGChatbot()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = Path("logs") / f"answers-{timestamp}.log"
    history: List[Tuple[str, str]] = []

    if args.interactive or not questions:
        await run_interactive_loop(chatbot, args.limit, history, log_path)
    else:
        responses: List[Tuple[str, str]] = []
        for question in questions:
            answer = await chatbot.answer(question, limit=args.limit, conversation_history=history)
            history.append((question, answer))
            responses.append((question, answer))
        _write_log(log_path, responses)
        print(f"Saved {len(responses)} answers to {log_path}")


async def run_interactive_loop(
    chatbot: RAGChatbot,
    limit: int,
    history: List[Tuple[str, str]],
    log_path: Path,
) -> None:
    print("Interactive mode. Press Enter on an empty line to exit.\n")
    responses: List[Tuple[str, str]] = []
    while True:
        question = (await asyncio.to_thread(input, "You: ")).strip()
        if not question:
            break
        answer = await chatbot.answer(question, limit=limit, conversation_history=history)
        history.append((question, answer))
        responses.append((question, answer))
        print(f"Assistant: {answer}\n")
    if responses:
        _write_log(log_path, responses)
        print(f"Saved {len(responses)} answers to {log_path}")
    else:
        print("No questions asked. Nothing saved.")


def _write_log(log_path: Path, responses: Sequence[Tuple[str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [f"Q: {q}\nA: {a}\n" for q, a in responses]
    log_path.write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
