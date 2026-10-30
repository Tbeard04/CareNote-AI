"""Validation chain: repair -> retry -> fallback -> needs_human_review.

Triggered by validation failure only — never by guessed confidence:

    initial extraction fails to parse
      -> repair prompt carrying the exact Pydantic error -> one retry
      -> fall back to a stronger model (fresh extraction)
      -> still failing -> needs_human_review, no note

Every attempt is logged as a JSON-lines event and recorded on the result, so
the retry rate and the cost overhead of reliability are measurable from the
logs and from the eval harness alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from carenote.extract import ExtractionParseError, build_prompts, parse_care_note
from carenote.logging_utils import get_logger, log_event
from carenote.providers import LLMProvider, LLMResult, get_provider
from carenote.schema import CareNote

REPAIR_PROMPT = """\
Deine vorherige Antwort war kein gültiger Eintrag. Der Validierungsfehler:

---
{error}
---

Deine vorherige Antwort:

---
{previous}
---

Korrigiere GENAU diesen Fehler und gib den vollständigen Eintrag erneut als \
ein einziges JSON-Objekt zurück. Kein Markdown, kein Text davor oder danach.
"""


@dataclass(frozen=True)
class Attempt:
    """One model call in the chain and how it went."""

    stage: str  # "initial" | "repair" | "fallback"
    provider: str
    model: str
    ok: bool
    error: str | None
    llm: LLMResult


@dataclass
class ValidatedExtraction:
    """Outcome of the whole chain."""

    note: CareNote | None
    needs_human_review: bool
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(a.llm.cost_usd for a in self.attempts)

    @property
    def total_latency_s(self) -> float:
        return sum(a.llm.latency_s for a in self.attempts)


def build_fallback_provider() -> LLMProvider | None:
    """The stronger model the chain escalates to; None when not configured."""
    name = os.getenv("CARENOTE_FALLBACK_PROVIDER")
    if not name:
        return None
    return get_provider(name, model=os.getenv("CARENOTE_FALLBACK_MODEL") or None)


def _short(error: str, limit: int = 500) -> str:
    return error if len(error) <= limit else error[:limit] + "…"


def extract_validated(
    transcript: str,
    provider: LLMProvider,
    fallback: LLMProvider | None = None,
) -> ValidatedExtraction:
    """Run the full chain for one transcript."""
    logger = get_logger()
    schema = CareNote.model_json_schema()
    system, user = build_prompts(transcript)
    result = ValidatedExtraction(note=None, needs_human_review=False)

    def attempt(stage: str, prov: LLMProvider, user_prompt: str) -> CareNote | None:
        try:
            llm = prov.complete(system, user_prompt, json_schema=schema)
        except Exception as exc:
            # Transport or provider failure (timeout, connection, 5xx): a
            # failed attempt the chain continues past, never a crash.
            error = _short(f"{type(exc).__name__}: {exc}")
            zero = LLMResult(
                text="", provider=prov.name, model=prov.model,
                input_tokens=0, output_tokens=0, latency_s=0.0,
            )
            result.attempts.append(
                Attempt(stage, prov.name, prov.model, ok=False, error=error, llm=zero)
            )
            log_event(
                logger, "validate", f"{stage}_provider_error",
                provider=prov.name, model=prov.model, error=error,
            )
            raise ExtractionParseError(error, "") from exc
        try:
            note = parse_care_note(llm.text)
        except ExtractionParseError as exc:
            result.attempts.append(
                Attempt(stage, prov.name, prov.model, ok=False, error=_short(str(exc)), llm=llm)
            )
            log_event(
                logger,
                "validate",
                f"{stage}_failed",
                provider=prov.name,
                model=prov.model,
                error=_short(str(exc)),
            )
            raise
        result.attempts.append(Attempt(stage, prov.name, prov.model, ok=True, error=None, llm=llm))
        log_event(logger, "validate", f"{stage}_ok", provider=prov.name, model=prov.model)
        return note

    # 1. Initial extraction.
    try:
        result.note = attempt("initial", provider, user)
        return result
    except ExtractionParseError as exc:
        last_error, last_raw = str(exc), exc.raw_text

    # 2. One repair retry on the same model, carrying the exact error.
    log_event(logger, "validate", "repair_retry", provider=provider.name, model=provider.model)
    repair_user = user + "\n\n" + REPAIR_PROMPT.format(
        error=_short(last_error), previous=last_raw[:4000]
    )
    try:
        result.note = attempt("repair", provider, repair_user)
        return result
    except ExtractionParseError:
        pass

    # 3. Fall back to a stronger model with a fresh extraction.
    if fallback is not None:
        log_event(logger, "validate", "fallback", provider=fallback.name, model=fallback.model)
        try:
            result.note = attempt("fallback", fallback, user)
            return result
        except ExtractionParseError:
            pass

    # 4. Out of options: a human has to look at this one.
    result.needs_human_review = True
    log_event(
        logger,
        "validate",
        "needs_human_review",
        provider=provider.name,
        attempts=len(result.attempts),
    )
    return result
