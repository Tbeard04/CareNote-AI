"""Mini-eval for CI: re-score stored predictions against gold, no LLM needed.

Every push re-validates the eval harness end-to-end — gold labels parse and
self-match, stored predictions parse against the strict schema, the scorer
runs, and aggregate F1 stays above a floor that would catch a silently broken
metric or schema change. Live model quality is measured by the full eval
(evals/run_eval.py); this guards the machinery around it.

Run:  uv run python evals/mini_eval.py [--notes 3] [--min-f1 0.4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from carenote.schema import CareNote  # noqa: E402
from metrics import aggregate, score_note  # noqa: E402

RESULTS = Path(__file__).parent / "results"
DATA = ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score stored predictions against gold.")
    parser.add_argument("--notes", type=int, default=3)
    parser.add_argument("--min-f1", type=float, default=0.4)
    args = parser.parse_args()

    note_ids = sorted(p.stem for p in (DATA / "gold").glob("note_*.json"))[: args.notes]
    pred_dirs = sorted(d for d in (RESULTS / "predictions").iterdir() if d.is_dir())
    if not note_ids or not pred_dirs:
        print("FEHLER: keine Gold-Labels oder keine gespeicherten Predictions gefunden")
        return 1

    scores = []
    for pred_dir in pred_dirs:
        model, _, condition = pred_dir.name.rpartition("_")
        for note_id in note_ids:
            pred_path = pred_dir / f"{note_id}.json"
            if not pred_path.exists():
                print(f"FEHLER: {pred_path} fehlt")
                return 1
            gold = CareNote.model_validate_json(
                (DATA / "gold" / f"{note_id}.json").read_text(encoding="utf-8")
            )
            predicted = CareNote.model_validate_json(pred_path.read_text(encoding="utf-8"))
            transcript_path = (
                DATA / "scripted" / f"{note_id}.txt"
                if condition == "clean"
                else RESULTS / "transcripts" / f"{note_id}.txt"
            )
            scores.append(
                score_note(
                    note_id, predicted, gold,
                    transcript_path.read_text(encoding="utf-8"),
                    provider="stored", model=model, condition=condition,
                    attempts=1, latency_s=0.0, cost_usd=0.0,
                )
            )

    failed = False
    for agg in sorted(aggregate(scores), key=lambda a: (a.model, a.condition)):
        marker = "ok " if agg.obs_f1 >= args.min_f1 else "LOW"
        if agg.obs_f1 < args.min_f1:
            failed = True
        print(
            f"[{marker}] {agg.model:20s} {agg.condition:5s} "
            f"F1={agg.obs_f1:.2f} P={agg.obs_precision:.2f} R={agg.obs_recall:.2f} "
            f"unsupported={agg.unsupported_rate:.1%} ({agg.notes} notes)"
        )

    print(f"\n{len(scores)} stored runs re-scored across {len(pred_dirs)} model columns.")
    if failed:
        print(f"FEHLER: mindestens eine Spalte unter F1-Schwelle {args.min_f1}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
