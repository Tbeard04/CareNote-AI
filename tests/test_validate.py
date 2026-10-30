"""Unit tests for the repair -> retry -> fallback validation chain."""

import json

import pytest

from carenote.providers import LLMProvider
from carenote.validate import build_fallback_provider, extract_validated

VALID_NOTE_JSON = json.dumps(
    {
        "resident_ref": "Frau Weber",
        "time_mention": None,
        "entry_type": "berichteblatt_eintrag",
        "observations": [],
        "risk_flags": [],
        "follow_up_actions": [],
    }
)
BROKEN_JSON = '{"entry_type": "arztbrief"}'  # invalid enum value


class ScriptedProvider(LLMProvider):
    """Plays back a list of responses, one per call."""

    def __init__(self, name: str, responses: list[str]) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.responses = list(responses)
        self.prompts: list[str] = []

    def _complete(self, system: str, user: str, json_schema: dict | None):
        self.prompts.append(user)
        return self.responses.pop(0), 10, 10


def test_clean_output_needs_one_attempt() -> None:
    provider = ScriptedProvider("p", [VALID_NOTE_JSON])
    result = extract_validated("egal", provider)
    assert result.note is not None
    assert not result.needs_human_review
    assert [a.stage for a in result.attempts] == ["initial"]


def test_repair_retry_carries_exact_error_and_previous_output() -> None:
    provider = ScriptedProvider("p", [BROKEN_JSON, VALID_NOTE_JSON])
    result = extract_validated("egal", provider)
    assert result.note is not None
    assert [a.stage for a in result.attempts] == ["initial", "repair"]
    repair_prompt = provider.prompts[1]
    assert "entry_type" in repair_prompt  # the exact Pydantic error travels along
    assert BROKEN_JSON in repair_prompt  # so does the previous raw output


def test_fallback_runs_fresh_extraction_on_stronger_model() -> None:
    provider = ScriptedProvider("weak", [BROKEN_JSON, BROKEN_JSON])
    fallback = ScriptedProvider("strong", [VALID_NOTE_JSON])
    result = extract_validated("egal", provider, fallback=fallback)
    assert result.note is not None
    assert [a.stage for a in result.attempts] == ["initial", "repair", "fallback"]
    assert result.attempts[-1].provider == "strong"
    # Fresh extraction: the fallback gets the original prompt, not the repair one.
    assert "Validierungsfehler" not in fallback.prompts[0]


def test_exhausted_chain_flags_human_review() -> None:
    provider = ScriptedProvider("weak", [BROKEN_JSON, BROKEN_JSON])
    fallback = ScriptedProvider("strong", [BROKEN_JSON])
    result = extract_validated("egal", provider, fallback=fallback)
    assert result.note is None
    assert result.needs_human_review
    assert len(result.attempts) == 3
    assert all(not a.ok for a in result.attempts)


def test_no_fallback_configured_flags_human_review_after_repair() -> None:
    provider = ScriptedProvider("weak", [BROKEN_JSON, BROKEN_JSON])
    result = extract_validated("egal", provider, fallback=None)
    assert result.needs_human_review
    assert [a.stage for a in result.attempts] == ["initial", "repair"]


def test_costs_and_latency_accumulate_across_attempts() -> None:
    provider = ScriptedProvider("p", [BROKEN_JSON, VALID_NOTE_JSON])
    result = extract_validated("egal", provider)
    assert result.total_latency_s == pytest.approx(
        sum(a.llm.latency_s for a in result.attempts)
    )
    assert len(result.attempts) == 2


def test_build_fallback_provider_none_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("CARENOTE_FALLBACK_PROVIDER", raising=False)
    assert build_fallback_provider() is None
