"""faster-whisper wrapper: audio file -> German transcript with word timestamps.

Word timestamps are kept so evidence spans can later point back into the
audio. The model runs locally (CPU, int8) and is cached per size; the default
``small`` keeps CI and laptops fast — set CARENOTE_WHISPER_MODEL=medium for
the accuracy the eval reports.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcription:
    text: str
    words: tuple[Word, ...]
    duration_s: float
    model_size: str
    latency_s: float


@lru_cache(maxsize=2)
def _get_model(model_size: str):
    from faster_whisper import WhisperModel

    # CPU/int8 runs everywhere; CARENOTE_WHISPER_DEVICE=cuda (+ cuDNN 9
    # installed) moves transcription to the GPU with float16.
    device = os.getenv("CARENOTE_WHISPER_DEVICE", "cpu")
    compute = os.getenv("CARENOTE_WHISPER_COMPUTE", "int8" if device == "cpu" else "float16")
    return WhisperModel(model_size, device=device, compute_type=compute)


def transcribe(audio_path: str | Path, model_size: str | None = None) -> Transcription:
    """Transcribe one audio file to German text."""
    size = model_size or os.getenv("CARENOTE_WHISPER_MODEL", "small")
    model = _get_model(size)
    start = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path), language="de", word_timestamps=True, vad_filter=True
    )
    words: list[Word] = []
    parts: list[str] = []
    for segment in segments:
        parts.append(segment.text)
        for word in segment.words or []:
            words.append(Word(start=word.start, end=word.end, text=word.word.strip()))
    return Transcription(
        text="".join(parts).strip(),
        words=tuple(words),
        duration_s=info.duration,
        model_size=size,
        latency_s=time.perf_counter() - start,
    )


def main() -> None:
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Transcribe an audio file with faster-whisper.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default=None, help="Whisper model size (default: small)")
    args = parser.parse_args()

    result = transcribe(args.audio, model_size=args.model)
    print(result.text)
    print(
        f"\n[whisper-{result.model_size}] {result.duration_s:.1f}s audio, "
        f"{len(result.words)} words, {result.latency_s:.2f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
