"""Results dashboard: the eval, explorable and bilingual.

Reads evals/results/raw_scores.json (written by evals/run_eval.py) and the
persisted predictions: headline tiles, quality/cost/latency charts, a
note x model heatmap, and a per-note drill-down showing each model's output
next to gold.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from i18n import t
from metrics import NoteScore, aggregate, wer

ROOT = Path(__file__).parent.parent.parent
RESULTS = ROOT / "evals" / "results"
DATA = ROOT / "data"

# Validated CVD-safe pair (light and dark surfaces): clean vs ASR everywhere.
COLOR_CLEAN = "#3B6FD4"
COLOR_ASR = "#C65A11"
CONDITION_SCALE = alt.Scale(domain=["clean", "asr"], range=[COLOR_CLEAN, COLOR_ASR])


def main() -> None:
    T = t()
    st.title(T["r_title"])

    raw_path = RESULTS / "raw_scores.json"
    if not raw_path.exists():
        st.info(T["r_no_results"])
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
            "unsupported": s.unsupported_claims,
            "claims": s.total_claims,
            "attempts": s.attempts,
            "latency_s": s.latency_s,
            "cost_usd": s.cost_usd,
        }
        for s in scores
    )
    st.caption(
        T["r_caption"].format(
            runs=len(scores), notes=df["note"].nunique(), models=df["model"].nunique()
        )
    )

    wers = {
        p.stem: wer(
            (DATA / "scripted" / f"{p.stem}.txt").read_text(encoding="utf-8"),
            p.read_text(encoding="utf-8"),
        )
        for p in sorted((RESULTS / "transcripts").glob("note_*.txt"))
    }
    best = max((a for a in aggs if a.condition == "asr"), key=lambda a: a.obs_f1)
    api = [a for a in aggs if a.condition == "asr" and a.cost_per_1000_notes_usd > 0]
    cheapest = min(api, key=lambda a: a.cost_per_1000_notes_usd) if api else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["r_best"], best.model, f"{best.obs_f1:.2f} F1")
    if cheapest is not None:
        c2.metric(
            T["r_cheapest"], cheapest.model,
            f"${cheapest.cost_per_1000_notes_usd:.2f} / 1k",
        )
    local_count = sum(1 for a in aggs if a.cost_per_1000_notes_usd == 0) // 2
    c3.metric(T["r_local"], f"{local_count}", T["r_local_delta"])
    c4.metric(T["r_wer"], f"{sum(wers.values()) / len(wers):.1%}" if wers else "–")

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

    st.subheader(T["r_quality"])
    chart = (
        alt.Chart(agg_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("condition:N", sort=["clean", "asr"]),
            y=alt.Y("f1:Q", title=T["r_f1_axis"], scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("condition:N", scale=CONDITION_SCALE, title=None),
            tooltip=[
                "model", "condition",
                alt.Tooltip("f1:Q", format=".2f"),
                alt.Tooltip("precision:Q", format=".2f"),
                alt.Tooltip("recall:Q", format=".2f"),
                alt.Tooltip("unsupported_rate:Q", format=".1%"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(T["r_quality_caption"])

    st.subheader(T["r_cost"])
    cost_chart = (
        alt.Chart(agg_df[agg_df["condition"] == "asr"])
        .mark_bar(cornerRadiusEnd=4, color=COLOR_ASR)
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("usd_per_1k:Q", title=T["r_cost_axis"]),
            tooltip=["model", alt.Tooltip("usd_per_1k:Q", format="$.2f")],
        )
        .properties(height=220)
    )
    st.altair_chart(cost_chart, use_container_width=True)
    st.caption(T["r_cost_caption"])

    st.subheader(T["r_latency"])
    lat_long = agg_df[agg_df["condition"] == "asr"].melt(
        id_vars=["model"], value_vars=["p50_s", "p95_s"],
        var_name="percentile", value_name="seconds",
    )
    lat_chart = (
        alt.Chart(lat_long)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="percentile:N",
            y=alt.Y("seconds:Q", title=T["r_latency_axis"]),
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

    st.subheader(T["r_table"])
    st.dataframe(
        agg_df.sort_values(["condition", "f1"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "f1": st.column_config.ProgressColumn(
                "Obs F1", format="%.2f", min_value=0, max_value=1
            ),
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
            "usd_per_1k": st.column_config.NumberColumn("$/1k", format="$%.2f"),
        },
    )

    st.subheader(T["r_heatmap"])
    heat_condition = st.radio(
        "condition", ["asr", "clean"], horizontal=True, label_visibility="collapsed"
    )
    heat = (
        alt.Chart(df[df["condition"] == heat_condition])
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("note:N", title=None),
            color=alt.Color("f1:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="F1"),
            tooltip=[
                "note", "model",
                alt.Tooltip("f1:Q", format=".2f"),
                alt.Tooltip("unsupported:Q"),
                alt.Tooltip("attempts:Q"),
            ],
        )
        .properties(height=24 * df["note"].nunique())
    )
    st.altair_chart(heat, use_container_width=True)
    st.caption(T["r_heatmap_caption"])

    st.subheader(T["r_drill"])
    sel1, sel2, sel3 = st.columns(3)
    note_id = sel1.selectbox(T["r_note"], sorted(df["note"].unique()))
    model = sel2.selectbox(T["r_model"], sorted(df["model"].unique()))
    condition = sel3.selectbox(T["r_condition"], ["asr", "clean"])

    row = df[(df["note"] == note_id) & (df["model"] == model) & (df["condition"] == condition)]
    if not row.empty:
        r = row.iloc[0]
        st.caption(
            T["r_run_stats"].format(
                f1=r["f1"], p=r["precision"], r=r["recall"],
                unsup=int(r["unsupported"]), claims=int(r["claims"]),
                attempts=int(r["attempts"]), latency=r["latency_s"], cost=r["cost_usd"],
            )
        )

    script_text = (DATA / "scripted" / f"{note_id}.txt").read_text(encoding="utf-8")
    asr_file = RESULTS / "transcripts" / f"{note_id}.txt"
    if condition == "asr" and asr_file.exists():
        asr = asr_file.read_text(encoding="utf-8")
        st.markdown("**" + T["r_asr_header"].format(wer=wer(script_text, asr)) + "**")
        st.code(asr, language=None)
    else:
        st.markdown(f"**{T['r_script_header']}**")
        st.code(script_text, language=None)

    pred_path = (
        RESULTS / "predictions" / f"{model.replace(':', '_')}_{condition}" / f"{note_id}.json"
    )
    colp, colg = st.columns(2)
    with colp:
        st.markdown("**" + T["r_model_output"].format(model=model) + "**")
        if pred_path.exists():
            st.json(json.loads(pred_path.read_text(encoding="utf-8")))
        else:
            st.info(T["r_no_prediction"])
    with colg:
        st.markdown(f"**{T['r_gold']}**")
        st.json(json.loads((DATA / "gold" / f"{note_id}.json").read_text(encoding="utf-8")))
