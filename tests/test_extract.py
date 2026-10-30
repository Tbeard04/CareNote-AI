"""Unit tests for extraction parsing and the provider interface (no live LLM)."""

import json

import pytest

from carenote.extract import (
    ExtractionParseError,
    build_prompts,
    extract_note,
    parse_care_note,
    strip_to_json,
)
from carenote.providers import LLMProvider, LLMResult, get_provider
from carenote.schema import CareNote

VALID_NOTE_JSON = json.dumps(
    {
        "resident_ref": "Frau Weber",
        "time_mention": "heute Mittag",
        "entry_type": "berichteblatt_eintrag",
        "observations": [
            {
                "text": "Nahrungsaufnahme reduziert.",
                "themenfeld": "selbstversorgung",
                "evidence_span": "hat heute Mittag kaum etwas gegessen",
                "measurement": None,
            }
        ],
        "risk_flags": [],
        "follow_up_actions": [],
    },
    ensure_ascii=False,
)


class FakeProvider(LLMProvider):
    """Returns a canned response; records what it was asked."""

    name = "fake"
    model = "fake-model"

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, str, dict | None]] = []

    def _complete(self, system: str, user: str, json_schema: dict | None):
        self.calls.append((system, user, json_schema))
        return self.response_text, 100, 50


def test_strip_to_json_handles_code_fences_and_prose() -> None:
    wrapped = f"Hier ist der Eintrag:\n```json\n{VALID_NOTE_JSON}\n```\nFertig."
    assert json.loads(strip_to_json(wrapped)) == json.loads(VALID_NOTE_JSON)


def test_strip_to_json_rejects_text_without_json() -> None:
    with pytest.raises(ExtractionParseError):
        strip_to_json("Es tut mir leid, das kann ich nicht.")


def test_parse_care_note_accepts_valid_output() -> None:
    note = parse_care_note(VALID_NOTE_JSON)
    assert isinstance(note, CareNote)
    assert note.resident_ref == "Frau Weber"


def test_parse_care_note_carries_exact_validation_error() -> None:
    broken = VALID_NOTE_JSON.replace("selbstversorgung", "ernaehrung")
    with pytest.raises(ExtractionParseError) as excinfo:
        parse_care_note(broken)
    assert "themenfeld" in str(excinfo.value)
    assert excinfo.value.raw_text == broken


def test_build_prompts_embed_schema_and_transcript() -> None:
    system, user = build_prompts("Frau Weber hat gut geschlafen.")
    assert "evidence_span" in system  # schema embedded for schema-less JSON modes
    assert "Frau Weber hat gut geschlafen." in user


def test_extract_note_passes_schema_and_returns_result() -> None:
    provider = FakeProvider(VALID_NOTE_JSON)
    result = extract_note("hat heute Mittag kaum etwas gegessen", provider)
    assert result.note.resident_ref == "Frau Weber"
    assert result.llm.provider == "fake"
    _, _, passed_schema = provider.calls[0]
    assert passed_schema == CareNote.model_json_schema()


def test_extract_note_raises_on_broken_output() -> None:
    provider = FakeProvider("kein json hier")
    with pytest.raises(ExtractionParseError):
        extract_note("egal", provider)


def test_llm_result_cost_known_and_unknown_models() -> None:
    kwargs = dict(text="", provider="x", input_tokens=1_000_000, output_tokens=1_000_000)
    haiku = LLMResult(model="claude-haiku-5-5", latency_s=0.0, **kwargs)
    assert haiku.cost_usd == pytest.approx(0.60)
    local = LLMResult(model="llama3.1:8b", latency_s=0.0, **kwargs)
    assert local.cost_usd == 0.0


def test_get_provider_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("gemini")


def test_get_provider_builds_ollama_without_any_keys() -> None:
    provider = get_provider("ollama")
    assert provider.name == "ollama"
    assert provider.base_url.startswith("http")
