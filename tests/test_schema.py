"""Unit tests for the CareNote schema and the unsupported-claim check."""

import pytest
from pydantic import ValidationError

from carenote.schema import (
    CareNote,
    EntryType,
    Measurement,
    Observation,
    RiskFlag,
    RiskType,
    Themenfeld,
)

TRANSCRIPT = (
    "So, kurze Notiz zu Frau Weber. Sie hat heute Mittag kaum etwas gegessen, "
    "vielleicht zwei, drei Löffel Suppe. Beim Aufstehen vom Stuhl ist sie fast "
    "gestürzt, ich musste sie stützen. Blutdruck war 145 zu 90. Bitte morgen "
    "nochmal nach dem Essen schauen."
)


def _valid_note() -> CareNote:
    return CareNote(
        resident_ref="Frau Weber",
        time_mention="heute Mittag",
        entry_type=EntryType.BERICHTEBLATT_EINTRAG,
        observations=[
            Observation(
                text="Nahrungsaufnahme deutlich reduziert, nur wenige Löffel Suppe.",
                themenfeld=Themenfeld.SELBSTVERSORGUNG,
                evidence_span="hat heute Mittag kaum etwas gegessen",
            ),
            Observation(
                text="Blutdruck erhöht gemessen.",
                themenfeld=Themenfeld.KRANKHEITSBEZOGENE_ANFORDERUNGEN,
                evidence_span="Blutdruck war 145 zu 90",
                measurement=Measurement(value=145, unit="mmHg"),
            ),
        ],
        risk_flags=[
            RiskFlag(
                risk=RiskType.STURZ,
                evidence_span="Beim Aufstehen vom Stuhl ist sie fast gestürzt",
            )
        ],
        follow_up_actions=["Morgen nach dem Essen schauen."],
    )


def test_valid_note_round_trips_through_json() -> None:
    note = _valid_note()
    restored = CareNote.model_validate_json(note.model_dump_json())
    assert restored == note


def test_unknown_fields_are_rejected() -> None:
    data = _valid_note().model_dump()
    data["diagnosis"] = "erfunden"
    with pytest.raises(ValidationError, match="diagnosis"):
        CareNote.model_validate(data)


def test_empty_evidence_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Observation(
            text="Etwas",
            themenfeld=Themenfeld.MOBILITAET_UND_BEWEGUNG,
            evidence_span="   ",
        )


def test_empty_follow_up_action_is_rejected() -> None:
    data = _valid_note().model_dump()
    data["follow_up_actions"] = ["Arzt informieren", "  "]
    with pytest.raises(ValidationError, match="follow_up_actions"):
        CareNote.model_validate(data)


def test_absent_information_defaults_to_null_or_empty() -> None:
    note = CareNote(entry_type=EntryType.RISIKO_BEOBACHTUNG)
    assert note.resident_ref is None
    assert note.time_mention is None
    assert note.observations == []
    assert note.risk_flags == []
    assert note.follow_up_actions == []


def test_unsupported_claims_empty_for_faithful_note() -> None:
    assert _valid_note().unsupported_claims(TRANSCRIPT) == []


def test_unsupported_claims_catches_invented_span() -> None:
    note = _valid_note()
    note.observations[0] = note.observations[0].model_copy(
        update={"evidence_span": "sie hat über starke Schmerzen geklagt"}
    )
    missing = note.unsupported_claims(TRANSCRIPT)
    assert missing == ["sie hat über starke Schmerzen geklagt"]


def test_unsupported_claims_ignores_case_and_whitespace() -> None:
    note = _valid_note()
    note.observations[0] = note.observations[0].model_copy(
        update={"evidence_span": "HAT heute   mittag kaum\netwas gegessen"}
    )
    assert note.unsupported_claims(TRANSCRIPT) == []


def test_themenfelder_are_exactly_the_six_sis_fields() -> None:
    assert {t.value for t in Themenfeld} == {
        "kognition_und_kommunikation",
        "mobilitaet_und_bewegung",
        "krankheitsbezogene_anforderungen",
        "selbstversorgung",
        "leben_in_sozialen_beziehungen",
        "wohnen_und_haeuslichkeit",
    }
