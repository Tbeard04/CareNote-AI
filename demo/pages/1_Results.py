"""Results dashboard: the eval, explorable.

Reads evals/results/raw_scores.json (written by evals/run_eval.py) and the
persisted predictions, and renders the provider comparison interactively:
headline tiles, quality/cost/latency charts, a note x model heatmap, and a
per-note drill-down showing each model's output next to gold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from metrics import NoteScore, aggregate, wer  # noqa: E402

RESULTS = ROOT / "evals" / "results"
DATA = ROOT / "data"

# Validated CVD-safe pair (light and dark surfaces): clean vs ASR everywhere.
COLOR_CLEAN = "#3B6FD4"
COLOR_ASR = "#C65A11"
CONDITION_SCALE = alt.Scale(domain=["clean", "asr"], range=[COLOR_CLEAN, COLOR_ASR])

st.set_page_config(page_title="CareNote AI — Results", page_icon="📊", layout="wide")
st.title("Eval results")

raw_path = RESULTS / "raw_scores.json"
if not raw_path.exists():
    st.info("No eval results yet — run `uv run python evals/run_eval.py` first.")
    st.stop()

scores = [NoteScore(**row) for row in json.loads(raw_path.read_text(encoding="utf-8"))]
aggs = aggregate(scores)
df = pd.DataFrame(
    {
        "model": s.model,
        "condition": s.condition,
        "note": s.note_id,
        "f1": s.obs_f1,
        "precision": s.obs_precision,
        "recall": s.obs_recall,
        "themenfeld_acc": s.themenfeld_accuracy,
        "risk_recall": s.risk_recall,
        "unsupported": s.unsupported_claims,
        "claims": s.total_claims,
        "attempts": s.attempts,
        "latency_s": s.latency_s,
        "cost_usd": s.cost_usd,
        "review": s.needs_human_review,
    }
    for s in scores
)
st.caption(
    f"{len(scores)} scored runs · {df['note'].nunique()} notes · "
    f"{df['model'].nunique()} models · conditions: clean script vs ASR transcript. "
    "Repair-retry on, cross-provider fallback off for fair per-model columns."
)

# ---- Headline tiles ---------------------------------------------------------
wers = {
    p.stem: wer(
        (DATA / "scripted" / f"{p.stem}.txt").read_text(encoding="utf-8"),
        p.read_text(encoding="utf-8"),
    )
    for p in sorted((RESULTS / "transcripts").glob("note_*.txt"))
}
best = max((a for a in aggs if a.condition == "asr"), key=lambda a: a.obs_f1)
api_costs = [a for a in aggs if a.condition == "asr" and a.cost_per_1000_notes_usd > 0]
cheapest = min(api_costs, key=lambda a: a.cost_per_1000_notes_usd) if api_costs else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model (ASR, F1)", best.model, f"{best.obs_f1:.2f} F1")
if cheapest is not None:
    c2.metric(
        "Cheapest API model", cheapest.model, f"${cheapest.cost_per_1000_notes_usd:.2f} / 1k notes"
    )
local_count = sum(1 for a in aggs if a.cost_per_1000_notes_usd == 0) // 2
c3.metric("Local models", f"{local_count}", "$0 marginal cost")
c4.metric("Mean WER (whisper-small)", f"{sum(wers.values()) / len(wers):.1%}" if wers else "–")

# ---- Charts -----------------------------------------------------------------
agg_df = pd.DataFrame(
    {
        "model": a.model,
        "condition": a.condition,
        "f1": a.obs_f1,
        "precision": a.obs_precision,
        "recall": a.obs_recall,
        "themenfeld_acc": a.themenfeld_accuracy,
        "risk_precision": a.risk_precision,
        "risk_recall": a.risk_recall,
        "resident_acc": a.resident_accuracy,
        "unsupported_rate": a.unsupported_rate,
        "p50_s": a.latency_p50,
        "p95_s": a.latency_p95,
        "usd_per_1k": a.cost_per_1000_notes_usd,
    }
    for a in aggs
)

st.subheader("Extraction quality (Observation F1)")
chart = (
        alt.Chart(agg_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("condition:N", sort=["clean", "asr"]),
            y=alt.Y("f1:Q", title="Observation F1 vs gold", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("condition:N", scale=CONDITION_SCALE, title=None),
            tooltip=[
                "model", "condition",
                alt.Tooltip("f1:Q", format=".2f"),
                alt.Tooltip("precision:Q", format=".2f"),
                alt.Tooltip("recall:Q", format=".2f"),
                alt.Tooltip("unsupported_rate:Q", format=".1%", title="unsupported claims"),
            ],
        )
    .properties(height=280)
)
st.altair_chart(chart, use_container_width=True)
st.caption("The clean-vs-ASR gap isolates what speech recognition costs each model.")

st.subheader("Cost per 1,000 notes")
cost_chart = (
    alt.Chart(agg_df[agg_df["condition"] == "asr"])
    .mark_bar(cornerRadiusEnd=4, color=COLOR_ASR)
    .encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("usd_per_1k:Q", title="USD per 1,000 notes"),
        tooltip=["model", alt.Tooltip("usd_per_1k:Q", format="$.2f")],
    )
    .properties(height=220)
)
st.altair_chart(cost_chart, use_container_width=True)
st.caption("Local models (Ollama) run at zero marginal cost on the GPU.")

st.subheader("Latency")
lat_long = agg_df[agg_df["condition"] == "asr"].melt(
    id_vars=["model"], value_vars=["p50_s", "p95_s"], var_name="percentile", value_name="seconds"
)
lat_chart = (
    alt.Chart(lat_long)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="percentile:N",
        y=alt.Y("seconds:Q", title="Extraction latency (s)"),
        color=alt.Color(
            "percentile:N",
            scale=alt.Scale(domain=["p50_s", "p95_s"], range=[COLOR_CLEAN, COLOR_ASR]),
            title=None,
        ),
        tooltip=["model", "percentile", alt.Tooltip("seconds:Q", format=".1f")],
    )
    .properties(height=220)
)
st.altair_chart(lat_chart, use_container_width=True)

# ---- Full comparison table --------------------------------------------------
st.subheader("Full comparison")
st.dataframe(
    agg_df.sort_values(["condition", "f1"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "f1": st.column_config.ProgressColumn("Obs F1", format="%.2f", min_value=0, max_value=1),
        "precision": st.column_config.ProgressColumn(
            "Obs P", format="%.2f", min_value=0, max_value=1
        ),
        "recall": st.column_config.ProgressColumn(
            "Obs R", format="%.2f", min_value=0, max_value=1
        ),
        "themenfeld_acc": st.column_config.ProgressColumn(
            "Themenfeld", format="%.2f", min_value=0, max_value=1
        ),
        "risk_precision": st.column_config.NumberColumn("Risk P", format="%.2f"),
        "risk_recall": st.column_config.NumberColumn("Risk R", format="%.2f"),
        "resident_acc": st.column_config.NumberColumn("Resident", format="%.2f"),
        "unsupported_rate": st.column_config.NumberColumn("Unsupported", format="percent"),
        "p50_s": st.column_config.NumberColumn("p50 (s)", format="%.1f"),
        "p95_s": st.column_config.NumberColumn("p95 (s)", format="%.1f"),
        "usd_per_1k": st.column_config.NumberColumn("$/1k notes", format="$%.2f"),
    },
)

# ---- Note x model heatmap ---------------------------------------------------
st.subheader("Where each model struggles")
heat_condition = st.radio(
    "Condition", ["asr", "clean"], horizontal=True, label_visibility="collapsed"
)
heat = (
    alt.Chart(df[df["condition"] == heat_condition])
    .mark_rect(stroke="#00000000", cornerRadius=2)
    .encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("note:N", title=None),
        color=alt.Color(
            "f1:Q",
            scale=alt.Scale(scheme="blues", domain=[0, 1]),
            title="Obs F1",
        ),
        tooltip=[
            "note", "model",
            alt.Tooltip("f1:Q", format=".2f"),
            alt.Tooltip("unsupported:Q", title="unsupported claims"),
            alt.Tooltip("attempts:Q"),
        ],
    )
    .properties(height=24 * df["note"].nunique())
)
st.altair_chart(heat, use_container_width=True)
st.caption("One row per note, one column per model — pale cells are the interesting ones.")

# ---- Drill-down -------------------------------------------------------------
st.subheader("Per-note drill-down: model output vs gold")
sel1, sel2, sel3 = st.columns(3)
note_id = sel1.selectbox("Note", sorted(df["note"].unique()))
model = sel2.selectbox("Model", sorted(df["model"].unique()))
condition = sel3.selectbox("Condition", ["asr", "clean"])

row = df[(df["note"] == note_id) & (df["model"] == model) & (df["condition"] == condition)]
if not row.empty:
    r = row.iloc[0]
    st.caption(
        f"F1 {r['f1']:.2f} · precision {r['precision']:.2f} · recall {r['recall']:.2f} · "
        f"{int(r['unsupported'])} unsupported of {int(r['claims'])} claims · "
        f"{int(r['attempts'])} attempt(s) · {r['latency_s']:.1f}s · ${r['cost_usd']:.5f}"
    )

script_text = (DATA / "scripted" / f"{note_id}.txt").read_text(encoding="utf-8")
asr_file = RESULTS / "transcripts" / f"{note_id}.txt"
if condition == "asr" and asr_file.exists():
    asr = asr_file.read_text(encoding="utf-8")
    st.markdown(f"**ASR transcript** (WER {wer(script_text, asr):.1%} vs script)")
    st.code(asr, language=None)
else:
    st.markdown("**Clean script**")
    st.code(script_text, language=None)

pred_path = (
    RESULTS / "predictions" / f"{model.replace(':', '_')}_{condition}" / f"{note_id}.json"
)
gold_path = DATA / "gold" / f"{note_id}.json"
colp, colg = st.columns(2)
with colp:
    st.markdown(f"**{model} output**")
    if pred_path.exists():
        st.json(json.loads(pred_path.read_text(encoding="utf-8")))
    else:
        st.info("No stored prediction — re-run the eval to persist model outputs.")
with colg:
    st.markdown("**Gold label**")
    st.json(json.loads(gold_path.read_text(encoding="utf-8")))
