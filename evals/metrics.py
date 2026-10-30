"""Hand-rolled eval metrics — transparent on purpose, no eval framework.

Scoring one extracted CareNote against its gold label:

- Observations are aligned by evidence-span token overlap (both quote the same
  transcript), greedy best-match-first with an overlap coefficient >= 0.5.
  Precision/recall come from that alignment; Themenfeld and measurement
  accuracy are judged on the matched pairs only, so a mapping mistake does not
  double-count as a missed observation.
- Risk flags compare as sets of risk types.
- Follow-up actions align by token overlap like observations.
- The unsupported-claim rate (spans not found verbatim in the transcript the
  model saw) is a LOWER BOUND on hallucination: a model can quote correctly
  and still misattribute — that surfaces here as precision/recall loss.
- WER between the clean script and the ASR transcript quantifies what the
  speech layer costs (the clean-vs-ASR ablation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean, quantiles

from carenote.schema import CareNote

_TITLE_RE = re.compile(r"^(frau|herrn?|bewohnerin|bewohner|die|der)\s+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zäöüß0-9]+", text.lower()))


def _overlap(a: str, b: str) -> float:
    """Overlap coefficient of token sets: |A∩B| / min(|A|, |B|)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _greedy_match(predicted: list[str], gold: list[str], threshold: float) -> list[tuple[int, int]]:
    """Greedy best-first alignment of two string lists; returns index pairs."""
    candidates = sorted(
        (
            (_overlap(p, g), pi, gi)
            for pi, p in enumerate(predicted)
            for gi, g in enumerate(gold)
        ),
        key=lambda t: t[0],
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, pi, gi in candidates:
        if score < threshold:
            break
        if pi in used_p or gi in used_g:
            continue
        pairs.append((pi, gi))
        used_p.add(pi)
        used_g.add(gi)
    return pairs


def _precision_recall(matched: int, predicted: int, gold: int) -> tuple[float, float]:
    precision = matched / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = matched / gold if gold else 1.0
    return precision, recall


def normalise_resident(ref: str | None) -> str:
    if ref is None:
        return ""
    out = ref.strip().lower()
    while True:
        stripped = _TITLE_RE.sub("", out)
        if stripped == out:
            return out
        out = stripped


@dataclass(frozen=True)
class NoteScore:
    """Everything the eval measures for one note x provider x condition."""

    note_id: str
    provider: str
    model: str
    condition: str  # "clean" | "asr"
    resident_match: bool
    entry_type_match: bool
    obs_precision: float
    obs_recall: float
    themenfeld_accuracy: float | None  # on matched observations; None if nothing matched
    measurement_accuracy: float | None  # on matched observations with gold measurements
    risk_precision: float
    risk_recall: float
    followup_precision: float
    followup_recall: float
    unsupported_claims: int
    total_claims: int
    needs_human_review: bool
    attempts: int
    latency_s: float
    cost_usd: float

    @property
    def obs_f1(self) -> float:
        p, r = self.obs_precision, self.obs_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def unsupported_rate(self) -> float:
        return self.unsupported_claims / self.total_claims if self.total_claims else 0.0


def score_note(
    note_id: str,
    predicted: CareNote,
    gold: CareNote,
    transcript_seen: str,
    *,
    provider: str,
    model: str,
    condition: str,
    attempts: int,
    latency_s: float,
    cost_usd: float,
    needs_human_review: bool = False,
) -> NoteScore:
    """Score one prediction against gold."""
    pred_spans = [o.evidence_span for o in predicted.observations]
    gold_spans = [o.evidence_span for o in gold.observations]
    pairs = _greedy_match(pred_spans, gold_spans, threshold=0.5)
    obs_p, obs_r = _precision_recall(len(pairs), len(pred_spans), len(gold_spans))

    themenfeld_hits = [
        predicted.observations[pi].themenfeld == gold.observations[gi].themenfeld
        for pi, gi in pairs
    ]
    themenfeld_acc = mean(themenfeld_hits) if themenfeld_hits else None

    measurement_hits = []
    for pi, gi in pairs:
        gold_m = gold.observations[gi].measurement
        if gold_m is None:
            continue
        pred_m = predicted.observations[pi].measurement
        measurement_hits.append(pred_m is not None and pred_m.value == gold_m.value)
    measurement_acc = mean(measurement_hits) if measurement_hits else None

    pred_risks = {f.risk for f in predicted.risk_flags}
    gold_risks = {f.risk for f in gold.risk_flags}
    risk_p, risk_r = _precision_recall(
        len(pred_risks & gold_risks), len(pred_risks), len(gold_risks)
    )

    fu_pairs = _greedy_match(predicted.follow_up_actions, gold.follow_up_actions, threshold=0.3)
    fu_p, fu_r = _precision_recall(
        len(fu_pairs), len(predicted.follow_up_actions), len(gold.follow_up_actions)
    )

    unsupported = predicted.unsupported_claims(transcript_seen)
    total_claims = len(predicted.observations) + len(predicted.risk_flags)

    return NoteScore(
        note_id=note_id,
        provider=provider,
        model=model,
        condition=condition,
        resident_match=normalise_resident(predicted.resident_ref)
        == normalise_resident(gold.resident_ref),
        entry_type_match=predicted.entry_type == gold.entry_type,
        obs_precision=obs_p,
        obs_recall=obs_r,
        themenfeld_accuracy=themenfeld_acc,
        measurement_accuracy=measurement_acc,
        risk_precision=risk_p,
        risk_recall=risk_r,
        followup_precision=fu_p,
        followup_recall=fu_r,
        unsupported_claims=len(unsupported),
        total_claims=total_claims,
        needs_human_review=needs_human_review,
        attempts=attempts,
        latency_s=latency_s,
        cost_usd=cost_usd,
    )


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via word-level Levenshtein distance."""
    ref = re.findall(r"\S+", _strip_punct(reference.lower()))
    hyp = re.findall(r"\S+", _strip_punct(hypothesis.lower()))
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def _strip_punct(text: str) -> str:
    return re.sub(r"[^\wäöüß\s]", "", text)


@dataclass
class Aggregate:
    """Mean scores for one provider x condition slice."""

    provider: str
    model: str
    condition: str
    notes: int
    scores: list[NoteScore] = field(repr=False, default_factory=list)

    @property
    def obs_f1(self) -> float:
        return mean(s.obs_f1 for s in self.scores)

    @property
    def obs_precision(self) -> float:
        return mean(s.obs_precision for s in self.scores)

    @property
    def obs_recall(self) -> float:
        return mean(s.obs_recall for s in self.scores)

    @property
    def themenfeld_accuracy(self) -> float:
        vals = [s.themenfeld_accuracy for s in self.scores if s.themenfeld_accuracy is not None]
        return mean(vals) if vals else 0.0

    @property
    def risk_recall(self) -> float:
        return mean(s.risk_recall for s in self.scores)

    @property
    def risk_precision(self) -> float:
        return mean(s.risk_precision for s in self.scores)

    @property
    def resident_accuracy(self) -> float:
        return mean(1.0 if s.resident_match else 0.0 for s in self.scores)

    @property
    def unsupported_rate(self) -> float:
        claims = sum(s.total_claims for s in self.scores)
        return sum(s.unsupported_claims for s in self.scores) / claims if claims else 0.0

    @property
    def human_review_rate(self) -> float:
        return mean(1.0 if s.needs_human_review else 0.0 for s in self.scores)

    @property
    def retry_rate(self) -> float:
        return mean(1.0 if s.attempts > 1 else 0.0 for s in self.scores)

    @property
    def latency_p50(self) -> float:
        return _quantile([s.latency_s for s in self.scores], 0.50)

    @property
    def latency_p95(self) -> float:
        return _quantile([s.latency_s for s in self.scores], 0.95)

    @property
    def cost_per_note_usd(self) -> float:
        return mean(s.cost_usd for s in self.scores)

    @property
    def cost_per_1000_notes_usd(self) -> float:
        return self.cost_per_note_usd * 1000


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    cuts = quantiles(sorted(values), n=100, method="inclusive")
    return cuts[max(0, min(98, round(q * 100) - 1))]


def aggregate(scores: list[NoteScore]) -> list[Aggregate]:
    """Group scores into provider x condition aggregates, stable order."""
    groups: dict[tuple[str, str, str], list[NoteScore]] = {}
    for score in scores:
        groups.setdefault((score.provider, score.model, score.condition), []).append(score)
    return [
        Aggregate(provider=p, model=m, condition=c, notes=len(group), scores=group)
        for (p, m, c), group in groups.items()
    ]
