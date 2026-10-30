"""Run the full eval: every note x every provider x {clean, asr}.

The clean condition extracts from the scripted text; the asr condition from the
faster-whisper transcript of the TTS audio. Comparing the two isolates speech
errors from extraction errors (the ablation). ASR transcripts are cached under
evals/results/transcripts/ so repeated runs don't re-transcribe.

Providers are given as NAME or NAME=MODEL, e.g.:

    uv run python evals/run_eval.py --providers ollama anthropic openai "ollama=qwen3:8b"

Fairness note: the eval runs the repair-retry (same model) but NOT the
cross-provider fallback, so every column's accuracy and cost belong to exactly
one model. needs_human_review notes score with an empty note (recall 0).

Outputs: evals/results/raw_scores.json, results.md, f1_by_provider.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from carenote.logging_utils import get_logger, log_event
from carenote.providers import get_provider
from carenote.redact import redact, unredact_note
from carenote.schema import CareNote, EntryType
from carenote.transcribe import transcribe
from carenote.validate import extract_validated
from metrics import Aggregate, NoteScore, aggregate, score_note, wer

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RESULTS = Path(__file__).parent / "results"
TRANSCRIPT_CACHE = RESULTS / "transcripts"

# Chart series colors validated with the dataviz palette checker (CVD-safe).
COLOR_CLEAN = "#3B6FD4"
COLOR_ASR = "#C65A11"
INK = "#444444"


def note_ids(limit: int | None = None) -> list[str]:
    ids = sorted(p.stem for p in (DATA / "scripted").glob("note_*.txt"))
    return ids[:limit] if limit else ids


def asr_text(note_id: str) -> str:
    """Whisper transcript for a note, cached on disk."""
    TRANSCRIPT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = TRANSCRIPT_CACHE / f"{note_id}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    text = transcribe(DATA / "audio" / f"{note_id}.mp3").text
    cache.write_text(text, encoding="utf-8")
    return text


def run_one(note_id: str, provider_name: str, model: str | None, condition: str) -> NoteScore:
    gold = CareNote.model_validate_json(
        (DATA / "gold" / f"{note_id}.json").read_text(encoding="utf-8")
    )
    original = (
        (DATA / "scripted" / f"{note_id}.txt").read_text(encoding="utf-8")
        if condition == "clean"
        else asr_text(note_id)
    )

    provider = get_provider(provider_name, model=model)
    redaction = redact(original)
    outcome = extract_validated(redaction.text, provider, fallback=None)

    if outcome.note is not None:
        predicted = unredact_note(outcome.note, redaction.mapping)
    else:
        predicted = CareNote(entry_type=EntryType.BERICHTEBLATT_EINTRAG)

    # Persist the prediction so the demo dashboard can show output vs gold.
    pred_dir = RESULTS / "predictions" / f"{provider.model.replace(':', '_')}_{condition}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / f"{note_id}.json").write_text(
        predicted.model_dump_json(indent=1), encoding="utf-8"
    )

    return score_note(
        note_id,
        predicted,
        gold,
        original,
        provider=provider.name,
        model=provider.model,
        condition=condition,
        attempts=len(outcome.attempts),
        latency_s=outcome.total_latency_s,
        cost_usd=outcome.total_cost_usd,
        needs_human_review=outcome.needs_human_review,
    )


def wer_report(ids: list[str]) -> dict[str, float]:
    return {
        note_id: wer(
            (DATA / "scripted" / f"{note_id}.txt").read_text(encoding="utf-8"),
            asr_text(note_id),
        )
        for note_id in ids
    }


def write_results_md(
    aggregates: list[Aggregate], wers: dict[str, float], eur_per_usd: float
) -> None:
    lines = [
        "# Eval results",
        "",
        f"{len(wers)} notes x {len({(a.provider, a.model) for a in aggregates})} model columns "
        "x 2 conditions (clean script vs ASR transcript). "
        "Repair-retry enabled, cross-provider fallback disabled for fair per-model columns.",
        "",
        "## Quality",
        "",
        "| Provider | Model | Condition | Obs P | Obs R | Obs F1 | Themenfeld acc "
        "| Risk P | Risk R | Resident acc | Unsupported claims |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(aggregates, key=lambda a: (a.provider, a.model, a.condition))
    for a in ordered:
        lines.append(
            f"| {a.provider} | {a.model} | {a.condition} | {a.obs_precision:.2f} "
            f"| {a.obs_recall:.2f} | **{a.obs_f1:.2f}** | {a.themenfeld_accuracy:.2f} "
            f"| {a.risk_precision:.2f} | {a.risk_recall:.2f} | {a.resident_accuracy:.2f} "
            f"| {a.unsupported_rate:.1%} |"
        )
    lines += [
        "",
        "## Latency, reliability, cost",
        "",
        "| Provider | Model | Condition | p50 latency | p95 latency | Retry rate "
        "| Human review | $/note | $/1k notes | €/1k notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in ordered:
        eur_1k = a.cost_per_1000_notes_usd * eur_per_usd
        lines.append(
            f"| {a.provider} | {a.model} | {a.condition} | {a.latency_p50:.1f}s "
            f"| {a.latency_p95:.1f}s | {a.retry_rate:.0%} | {a.human_review_rate:.0%} "
            f"| ${a.cost_per_note_usd:.5f} | ${a.cost_per_1000_notes_usd:.2f} "
            f"| €{eur_1k:.2f} |"
        )
    mean_wer = sum(wers.values()) / len(wers)
    worst_id = max(wers, key=wers.get)
    lines += [
        "",
        "## ASR quality (faster-whisper)",
        "",
        f"Mean WER across {len(wers)} notes: **{mean_wer:.1%}** "
        f"(worst: {worst_id} at {wers[worst_id]:.1%}). "
        "Local models run at zero marginal cost; $/note covers API providers only.",
        "",
        "![Observation F1 by provider](f1_by_provider.png)",
        "",
    ]
    (RESULTS / "results.md").write_text("\n".join(lines), encoding="utf-8")


def write_chart(aggregates: list[Aggregate]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted({(a.provider, a.model) for a in aggregates})
    labels = [f"{p}\n{m}" for p, m in keys]

    def series(condition: str) -> list[float]:
        out = []
        for key in keys:
            match = [
                a
                for a in aggregates
                if (a.provider, a.model) == key and a.condition == condition
            ]
            out.append(match[0].obs_f1 if match else 0.0)
        return out

    x = range(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    bars_clean = ax.bar(
        [i - width / 2 for i in x], series("clean"), width, label="clean script",
        color=COLOR_CLEAN, edgecolor="none",
    )
    bars_asr = ax.bar(
        [i + width / 2 for i in x], series("asr"), width, label="ASR transcript",
        color=COLOR_ASR, edgecolor="none",
    )
    for bars in (bars_clean, bars_asr):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=INK)
    ax.set_xticks(list(x), labels, fontsize=8, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Observation F1 vs gold", fontsize=9, color=INK)
    ax.set_title("Extraction quality by provider: clean text vs ASR", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "f1_by_provider.png")


def parse_provider_spec(spec: str) -> tuple[str, str | None]:
    name, _, model = spec.partition("=")
    return name.strip(), (model.strip() or None)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the CareNote eval.")
    parser.add_argument(
        "--providers", nargs="+", default=["ollama", "anthropic", "openai"],
        help="Provider specs: NAME or NAME=MODEL",
    )
    parser.add_argument("--conditions", nargs="+", default=["clean", "asr"])
    parser.add_argument("--notes", type=int, default=None, help="Limit to first N notes")
    parser.add_argument("--eur-per-usd", type=float, default=0.92)
    args = parser.parse_args()

    logger = get_logger()
    ids = note_ids(args.notes)
    RESULTS.mkdir(parents=True, exist_ok=True)

    scores: list[NoteScore] = []
    for spec in args.providers:
        name, model = parse_provider_spec(spec)
        for condition in args.conditions:
            for note_id in ids:
                score = run_one(note_id, name, model, condition)
                scores.append(score)
                log_event(
                    logger, "eval", "note_scored",
                    note=note_id, provider=score.provider, model=score.model,
                    condition=condition, f1=round(score.obs_f1, 3),
                    attempts=score.attempts, review=score.needs_human_review,
                )

    (RESULTS / "raw_scores.json").write_text(
        json.dumps([asdict(s) for s in scores], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    aggregates = aggregate(scores)
    wers = wer_report(ids)
    write_results_md(aggregates, wers, args.eur_per_usd)
    write_chart(aggregates)
    print(f"\n{len(scores)} runs scored -> {RESULTS / 'results.md'}")


if __name__ == "__main__":
    main()
