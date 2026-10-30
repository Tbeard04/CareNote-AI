"""Unit tests for the QM pass and pipeline wiring (no live LLM)."""

import json

from carenote.pipeline import process_text
from carenote.providers import LLMProvider
from carenote.qm import rule_flags, run_qm
from carenote.schema import CareNote, EntryType, Measurement, Observation, Themenfeld

CLEAN_REVIEW = json.dumps({"widersprueche": [], "fehlende_angaben": []})


def _note(**overrides) -> CareNote:
    data = dict(
        resident_ref="Frau Weber",
        entry_type=EntryType.BERICHTEBLATT_EINTRAG,
        observations=[
            Observation(
                text="Blutdruck gemessen.",
                themenfeld=Themenfeld.KRANKHEITSBEZOGENE_ANFORDERUNGEN,
                evidence_span="Blutdruck war 145 zu 90",
                measurement=Measurement(value=145, unit="mmHg"),
            )
        ],
    )
    data.update(overrides)
    return CareNote(**data)


class CannedProvider(LLMProvider):
    name = "canned"
    model = "canned-model"

    def __init__(self, response: str) -> None:
        self.response = response

    def _complete(self, system: str, user: str, json_schema: dict | None):
        return self.response, 50, 20


class BrokenProvider(LLMProvider):
    name = "broken"
    model = "broken-model"

    def _complete(self, system: str, user: str, json_schema: dict | None):
        raise ConnectionError("kaputt")


def test_plausible_note_passes_rules() -> None:
    assert rule_flags(_note()) == []


def test_missing_resident_is_a_warning() -> None:
    flags = rule_flags(_note(resident_ref=None))
    assert any(f.code == "missing_resident" and f.severity == "warning" for f in flags)


def test_empty_note_is_only_an_info() -> None:
    flags = rule_flags(_note(observations=[]))
    assert [f.code for f in flags] == ["empty_note"]
    assert flags[0].severity == "info"


def test_implausible_vital_value_is_a_warning() -> None:
    note = _note()
    note.observations[0].measurement.value = 700
    flags = rule_flags(note)
    assert any(f.code == "implausible_value" for f in flags)


def test_unknown_unit_is_an_info() -> None:
    note = _note()
    note.observations[0].measurement.unit = "Zentner"
    flags = rule_flags(note)
    assert [f.code for f in flags] == ["unknown_unit"]


def test_llm_contradiction_becomes_blocking_warning() -> None:
    review = json.dumps(
        {
            "widersprueche": ["'gut gegessen' widerspricht 'Essen komplett verweigert'."],
            "fehlende_angaben": [],
        },
        ensure_ascii=False,
    )
    result = run_qm(_note(), "transcript", CannedProvider(review))
    assert not result.passed
    assert any(f.code == "contradiction" and f.severity == "warning" for f in result.flags)


def test_qm_llm_failure_blocks_instead_of_silently_approving() -> None:
    result = run_qm(_note(), "transcript", BrokenProvider())
    assert not result.passed
    assert any(f.code == "qm_unavailable" for f in result.flags)


def test_qm_without_provider_runs_rules_only() -> None:
    result = run_qm(_note(), "transcript", None)
    assert result.passed
    assert result.llm is None


def test_pipeline_wires_redaction_extraction_and_qm() -> None:
    note_json = json.dumps(
        {
            "resident_ref": "B1",
            "time_mention": None,
            "entry_type": "berichteblatt_eintrag",
            "observations": [
                {
                    "text": "B1 hat kaum gegessen.",
                    "themenfeld": "selbstversorgung",
                    "evidence_span": "Frau B1 hat kaum gegessen",
                    "measurement": None,
                }
            ],
            "risk_flags": [],
            "follow_up_actions": [],
        }
    )

    class ExtractThenQM(LLMProvider):
        name = "scripted"
        model = "scripted-model"

        def __init__(self) -> None:
            self.responses = [note_json, CLEAN_REVIEW]

        def _complete(self, system: str, user: str, json_schema: dict | None):
            return self.responses.pop(0), 10, 10

    provider = ExtractThenQM()
    result = process_text("Frau Weber hat kaum gegessen", provider, qm_provider=provider)
    assert result.note is not None
    assert result.note.resident_ref == "Weber"  # unredacted back to the real name
    assert result.note.observations[0].evidence_span == "Frau Weber hat kaum gegessen"
    assert result.qm is not None and result.qm.passed
    assert not result.needs_human_review


def test_pipeline_qm_warning_sets_review() -> None:
    note_json = json.dumps(
        {
            "resident_ref": None,
            "time_mention": None,
            "entry_type": "berichteblatt_eintrag",
            "observations": [],
            "risk_flags": [],
            "follow_up_actions": [],
        }
    )

    class ExtractThenQM(LLMProvider):
        name = "scripted"
        model = "scripted-model"

        def __init__(self) -> None:
            self.responses = [note_json, CLEAN_REVIEW]

        def _complete(self, system: str, user: str, json_schema: dict | None):
            return self.responses.pop(0), 10, 10

    provider = ExtractThenQM()
    result = process_text("Alles unauffällig.", provider, qm_provider=provider)
    assert result.needs_human_review  # missing_resident warning blocks sign-off
