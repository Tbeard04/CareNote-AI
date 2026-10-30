"""End-to-end orchestration: audio -> transcribe -> redact -> extract
(validated) -> QM -> structured CareNote, with one JSON-lines event per stage.

A linear pipeline on purpose — no agent framework, because nothing here needs
one. Each stage's duration and outcome is logged, so a pipeline run leaves an
auditable trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from carenote.logging_utils import get_logger, log_event
from carenote.providers import LLMProvider
from carenote.qm import QMResult, run_qm
from carenote.redact import Redaction, redact, unredact_note
from carenote.schema import CareNote
from carenote.transcribe import Transcription, transcribe
from carenote.validate import ValidatedExtraction, build_fallback_provider, extract_validated


@dataclass(frozen=True)
class PipelineResult:
    transcription: Transcription | None  # None when the input was already text
    redaction: Redaction
    extraction: ValidatedExtraction
    note: CareNote | None  # unredacted, ready for the care record
    qm: QMResult | None  # None when extraction already failed

    @property
    def needs_human_review(self) -> bool:
        if self.extraction.needs_human_review:
            return True
        return self.qm is not None and not self.qm.passed

    @property
    def total_cost_usd(self) -> float:
        qm_cost = self.qm.llm.cost_usd if self.qm and self.qm.llm else 0.0
        return self.extraction.total_cost_usd + qm_cost


def process_text(
    transcript: str,
    provider: LLMProvider,
    fallback: LLMProvider | None = None,
    qm_provider: LLMProvider | None = None,
    transcription: Transcription | None = None,
) -> PipelineResult:
    """Run the pipeline from a transcript string."""
    logger = get_logger()

    redaction = redact(transcript)
    log_event(logger, "redact", "done", placeholders=len(redaction.mapping))

    extraction = extract_validated(redaction.text, provider, fallback=fallback)

    note = None
    qm = None
    if extraction.note is not None:
        note = unredact_note(extraction.note, redaction.mapping)
        qm = run_qm(note, transcript, qm_provider)
        log_event(
            logger, "qm", "done",
            flags=[f.code for f in qm.flags], passed=qm.passed,
        )
    log_event(
        logger, "pipeline", "done",
        review=extraction.needs_human_review or (qm is not None and not qm.passed),
        attempts=len(extraction.attempts),
    )
    return PipelineResult(
        transcription=transcription,
        redaction=redaction,
        extraction=extraction,
        note=note,
        qm=qm,
    )


def process_audio(
    audio_path: str | Path,
    provider: LLMProvider,
    fallback: LLMProvider | None = None,
    qm_provider: LLMProvider | None = None,
) -> PipelineResult:
    """Run the pipeline from an audio file."""
    logger = get_logger()
    transcription = transcribe(audio_path)
    log_event(
        logger, "transcribe", "done",
        model=transcription.model_size,
        audio_s=round(transcription.duration_s, 1),
        latency_s=round(transcription.latency_s, 2),
    )
    return process_text(
        transcription.text, provider, fallback, qm_provider, transcription=transcription
    )


def main() -> None:
    import argparse
    import sys

    from dotenv import load_dotenv

    from carenote.providers import get_provider

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser(description="Audio -> validated structured care note.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--provider", default=None, help="anthropic | openai | ollama")
    parser.add_argument("--no-qm", action="store_true", help="skip the QM LLM call")
    args = parser.parse_args()

    provider = get_provider(args.provider)
    result = process_audio(
        args.audio,
        provider,
        fallback=build_fallback_provider(),
        qm_provider=None if args.no_qm else provider,
    )

    if result.note is not None:
        print(result.note.model_dump_json(indent=2))
    if result.qm is not None:
        for flag in result.qm.flags:
            print(f"[QM {flag.severity}] {flag.code}: {flag.message}", file=sys.stderr)
    print(
        f"\nreview={'JA' if result.needs_human_review else 'nein'} "
        f"attempts={len(result.extraction.attempts)} "
        f"cost=${result.total_cost_usd:.5f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
