"""Unit tests for PII pseudonymisation and restoration."""

from carenote.redact import redact, unredact_note, unredact_text
from carenote.schema import CareNote, EntryType, Observation, RiskFlag, RiskType, Themenfeld


def test_redact_title_names_and_rooms() -> None:
    text = "Kurz zu Herrn Brandt, Zimmer zwölf. Frau Keller hat nach ihrer Tochter gefragt."
    result = redact(text)
    assert "Brandt" not in result.text
    assert "Keller" not in result.text
    assert "zwölf" not in result.text
    assert result.text.startswith("Kurz zu Herrn B1, Zimmer Z1.")
    assert result.mapping == {"B1": "Brandt", "B2": "Keller", "Z1": "zwölf"}


def test_redact_same_name_twice_gets_one_placeholder() -> None:
    text = "Frau Weber hat gegessen. Später hat Frau Weber geschlafen."
    result = redact(text)
    assert result.text.count("B1") == 2
    assert result.mapping == {"B1": "Weber"}


def test_redact_keeps_titles_for_grammar() -> None:
    result = redact("Herr Yilmaz hat über Schmerzen geklagt.")
    assert result.text.startswith("Herr B1 ")


def test_unredact_text_round_trips() -> None:
    original = "Frau Weber und Herr Brandt waren im Zimmer acht."
    redaction = redact(original)
    assert unredact_text(redaction.text, redaction.mapping) == original


def test_domain_nouns_are_not_mistaken_for_names() -> None:
    # spaCy's small German NER labels "Blutdruck" as PER; the multi-token
    # filter must keep it out of the redaction.
    result = redact("Blutdruck war 145 zu 90. Puls war 92.")
    assert result.text == "Blutdruck war 145 zu 90. Puls war 92."
    assert result.mapping == {}


def test_full_name_without_title_is_redacted_via_ner() -> None:
    result = redact("Besuch von Anna Weber am Nachmittag.")
    assert "Anna Weber" not in result.text
    assert "B1" in result.text


def test_text_without_pii_is_unchanged() -> None:
    text = "Alles unauffällig heute, keine Besonderheiten."
    result = redact(text)
    assert result.text == text
    assert result.mapping == {}


def test_unredact_note_restores_all_fields() -> None:
    redaction = redact("Frau Weber ist beim Aufstehen fast gestürzt.")
    note = CareNote(
        resident_ref="B1",
        entry_type=EntryType.BERICHTEBLATT_EINTRAG,
        observations=[
            Observation(
                text="B1 ist fast gestürzt.",
                themenfeld=Themenfeld.MOBILITAET_UND_BEWEGUNG,
                evidence_span="B1 ist beim Aufstehen fast gestürzt",
            )
        ],
        risk_flags=[
            RiskFlag(risk=RiskType.STURZ, evidence_span="B1 ist beim Aufstehen fast gestürzt")
        ],
        follow_up_actions=["B1 beim Aufstehen begleiten"],
    )
    restored = unredact_note(note, redaction.mapping)
    assert restored.resident_ref == "Weber"
    assert restored.observations[0].text == "Weber ist fast gestürzt."
    assert restored.observations[0].evidence_span == "Weber ist beim Aufstehen fast gestürzt"
    assert restored.risk_flags[0].evidence_span == "Weber ist beim Aufstehen fast gestürzt"
    assert restored.follow_up_actions == ["Weber beim Aufstehen begleiten"]


def test_note_scripts_cover_fifteen_plus_unique_notes() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "generate_notes", Path(__file__).parent.parent / "data" / "generate_notes.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_notes"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    ids = [n.note_id for n in module.NOTES]
    assert len(ids) >= 15
    assert len(set(ids)) == len(ids)
    assert all(n.text and n.voice and n.edge_case for n in module.NOTES)
