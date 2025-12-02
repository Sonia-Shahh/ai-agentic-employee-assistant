"""Asynchronous audio transcription using OpenAI Whisper."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from openai import AsyncOpenAI, BadRequestError

from src.config import settings


AUDIO_CHUNK_MAX_SECONDS = int(os.getenv("AUDIO_CHUNK_MAX_SECONDS", "1250"))
AUDIO_CHUNK_OVERLAP_SECONDS = int(os.getenv("AUDIO_CHUNK_OVERLAP_SECONDS", "10"))


async def transcribe_audio(path: Path, client: AsyncOpenAI | None = None) -> str:
    """Transcribe an audio file, chunking if it exceeds the OpenAI duration limit."""
    active_client = client or AsyncOpenAI(api_key=settings.openai.api_key)

    # Try proactive chunking when ffmpeg is available and the file is too long.
    if _chunking_supported():
        duration = await asyncio.to_thread(_probe_audio_duration, path)
        if duration and duration > AUDIO_CHUNK_MAX_SECONDS:
            return await _transcribe_with_chunking(path, active_client)

    try:
        return await _transcribe_file(path, active_client)
    except BadRequestError as exc:
        # Fall back to chunking if Whisper rejects the request due to size limits.
        if _should_retry_with_chunking(exc) and _chunking_supported():
            return await _transcribe_with_chunking(path, active_client)
        raise


async def transcribe_audios(paths: Iterable[Path], client: AsyncOpenAI | None = None) -> List[Tuple[Path, str]]:
    """Transcribe multiple audio files asynchronously."""
    active_client = client or AsyncOpenAI(api_key=settings.openai.api_key)
    tasks = [transcribe_audio(path, client=active_client) for path in paths]
    transcripts = await asyncio.gather(*tasks)
    return list(zip(paths, transcripts))


async def _transcribe_file(path: Path, client: AsyncOpenAI) -> str:
    with path.open("rb") as file_handle:
        response = await client.audio.transcriptions.create(
            file=file_handle,
            model=settings.openai.transcription_model,
            response_format="text",
        )
    return response


async def _transcribe_with_chunking(path: Path, client: AsyncOpenAI) -> str:
    if not _chunking_supported():  # Defensive guard even though callers check first.
        raise RuntimeError("ffmpeg + ffprobe are required to chunk long audio files.")

    temp_dir = tempfile.mkdtemp(prefix="ragbot-audio-chunks-")
    temp_path = Path(temp_dir)
    try:
        chunk_paths = await asyncio.to_thread(
            _split_audio_file,
            path,
            temp_path,
            AUDIO_CHUNK_MAX_SECONDS,
            AUDIO_CHUNK_OVERLAP_SECONDS,
        )
        transcripts: List[str] = []
        for chunk_path in chunk_paths:
            transcripts.append(await _transcribe_file(chunk_path, client))
        return "\n".join(transcripts)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def _chunking_supported() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _should_retry_with_chunking(exc: BadRequestError) -> bool:
    message = (exc.body or {}).get("error", {}).get("message") if hasattr(exc, "body") else str(exc)
    if not isinstance(message, str):
        message = str(message)
    return "audio duration" in message.lower() and "maximum" in message.lower()


def _probe_audio_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _split_audio_file(source: Path, output_dir: Path, max_seconds: int, overlap_seconds: int) -> List[Path]:
    if max_seconds <= 0:
        raise ValueError("AUDIO_CHUNK_MAX_SECONDS must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap = max(0, min(overlap_seconds, max_seconds - 1))
    duration = _probe_audio_duration(source)
    if not duration:
        raise RuntimeError("Unable to determine audio duration for chunking.")
    if duration <= max_seconds:
        dest = output_dir / source.name
        shutil.copy2(source, dest)
        return [dest]

    chunk_paths: List[Path] = []
    start = 0.0
    step = max_seconds - overlap
    chunk_index = 0
    while start < duration:
        remaining = duration - start
        chunk_duration = min(max_seconds, remaining)
        chunk_path = output_dir / f"{source.stem}_chunk_{chunk_index}{source.suffix}"
        _cut_audio_segment(source, chunk_path, start, chunk_duration)
        chunk_paths.append(chunk_path)
        if remaining <= max_seconds:
            break
        start += step
        chunk_index += 1
    return chunk_paths


def _cut_audio_segment(source: Path, dest: Path, start: float, duration: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to chunk audio files.")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(dest),
    ]
    subprocess.run(command, check=True)
