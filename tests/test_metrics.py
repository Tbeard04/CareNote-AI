"""Unit tests for the hand-rolled eval metrics."""

import json
from pathlib import Path

import pytest

from carenote.schema import CareNote
from metrics import aggregate, normalise_resident, score_note, wer

GOLD = CareNote.model_validate_json(
    (Path(__file__).parent.parent / "data" / "gold" / "note_01.json").read_text(encoding="utf-8")
)
TRANSCRIPT = (
    Path(__file__).parent.parent / "data" / "scripted" / "note_01.txt"
).read_text(encoding="utf-8")


def _score(predicted: CareNote, **overrides):
    kwargs = dict(
        provider="test",
        model="test-model",
        condition="clean",
        attempts=1,
        latency_s=1.0,
        cost_usd=0.001,
    )
    kwargs.update(overrides)
    return score_note("note_01", predicted, GOLD, TRANSCRIPT, **kwargs)


def test_perfect_prediction_scores_perfectly() -> None:
    score = _score(GOLD)
    assert score.obs_precision == 1.0
    assert score.obs_recall == 1.0
    assert score.obs_f1 == 1.0
    assert score.themenfeld_accuracy == 1.0
    assert score.measurement_accuracy == 1.0
    assert score.risk_precision == 1.0 and score.risk_recall == 1.0
    assert score.resident_match and score.entry_type_match
    assert score.unsupported_rate == 0.0


def test_missing_observation_lowers_recall_not_precision() -> None:
    predicted = GOLD.model_copy(update={"observations": GOLD.observations[:2]})
    score = _score(predicted)
    assert score.obs_precision == 1.0
    assert score.obs_recall == pytest.approx(2 / 3)


def test_invented_observation_lowers_precision_and_counts_unsupported() -> None:
    invented = GOLD.observations[0].model_copy(
        update={"evidence_span": "sie hat über starke Kopfschmerzen geklagt"}
    )
    predicted = GOLD.model_copy(update={"observations": [*GOLD.observations, invented]})
    score = _score(predicted)
    assert score.obs_precision == pytest.approx(3 / 4)
    assert score.unsupported_claims == 1


def test_wrong_themenfeld_counts_as_mapping_error_not_miss() -> None:
    wrong = GOLD.observations[0].model_copy(update={"themenfeld": "wohnen_und_haeuslichkeit"})
    predicted = GOLD.model_copy(update={"observations": [wrong, *GOLD.observations[1:]]})
    score = _score(predicted)
    assert score.obs_recall == 1.0  # content was found…
    assert score.themenfeld_accuracy == pytest.approx(2 / 3)  # …but mapped wrong


def test_wrong_risk_type_hits_both_precision_and_recall() -> None:
    wrong_flag = GOLD.risk_flags[0].model_copy(update={"risk": "dehydration"})
    predicted = GOLD.model_copy(update={"risk_flags": [wrong_flag]})
    score = _score(predicted)
    assert score.risk_precision == 0.0
    assert score.risk_recall == 0.0


def test_empty_prediction_against_empty_gold_is_perfect() -> None:
    gold_04 = CareNote.model_validate_json(
        (Path(__file__).parent.parent / "data" / "gold" / "note_04.json").read_text(
            encoding="utf-8"
        )
    )
    score = score_note(
        "note_04",
        gold_04,
        gold_04,
        "Frau Krüger, Frühdienst. Alles unauffällig heute, keine Besonderheiten.",
        provider="test",
        model="m",
        condition="clean",
        attempts=1,
        latency_s=0.5,
        cost_usd=0.0,
    )
    assert score.obs_precision == 1.0 and score.obs_recall == 1.0


def test_normalise_resident_strips_titles() -> None:
    assert normalise_resident("Frau Weber") == normalise_resident("Weber")
    assert normalise_resident("Herrn Brandt") == normalise_resident("Herr Brandt")
    assert normalise_resident(None) == ""
    assert normalise_resident("Die Dame von Zimmer acht") == normalise_resident(
        "Dame von Zimmer acht"
    )


def test_wer_basics() -> None:
    assert wer("das ist ein Test", "das ist ein Test") == 0.0
    assert wer("das ist ein Test", "das ist kein Test") == pytest.approx(0.25)
    assert wer("das ist ein Test", "das ist ein Test,") == 0.0  # punctuation ignored


def test_aggregate_groups_by_provider_and_condition() -> None:
    s1 = _score(GOLD)
    s2 = _score(GOLD, condition="asr")
    s3 = _score(GOLD, provider="other")
    groups = aggregate([s1, s2, s3])
    assert len(groups) == 3
    clean = next(g for g in groups if g.provider == "test" and g.condition == "clean")
    assert clean.notes == 1
    assert clean.obs_f1 == 1.0
    assert clean.cost_per_1000_notes_usd == pytest.approx(1.0)


def test_gold_files_are_internally_consistent() -> None:
    root = Path(__file__).parent.parent / "data"
    gold_files = sorted((root / "gold").glob("note_*.json"))
    assert len(gold_files) == 16
    for gold_path in gold_files:
        gold = CareNote.model_validate_json(gold_path.read_text(encoding="utf-8"))
        script = (root / "scripted" / f"{gold_path.stem}.txt").read_text(encoding="utf-8")
        assert gold.unsupported_claims(script) == [], f"{gold_path.stem} has invented spans"
        # Perfect self-score: the metric must agree the gold matches itself.
        score = score_note(
            gold_path.stem, gold, gold, script,
            provider="t", model="m", condition="clean",
            attempts=1, latency_s=0.0, cost_usd=0.0,
        )
        assert score.obs_f1 == 1.0, f"{gold_path.stem} does not self-match"


def test_gold_json_rejects_unknown_fields() -> None:
    data = json.loads(
        (Path(__file__).parent.parent / "data" / "gold" / "note_01.json").read_text(
            encoding="utf-8"
        )
    )
    data["erfunden"] = True
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CareNote.model_validate(data)
