"""Process page: voice note in, validated structured care note out.

Shows every pipeline stage — transcript, pseudonymisation, validated
extraction with evidence spans, QM flags — because the stages ARE the
product: a draft a caregiver confirms in seconds, never an auto-committed
entry.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from carenote.pipeline import PipelineResult, process_audio
from carenote.providers import get_provider
from carenote.validate import build_fallback_provider
from i18n import t

AUDIO_DIR = Path(__file__).parent.parent.parent / "data" / "audio"


@st.cache_resource(show_spinner=False)
def _providers(name: str, qm_name: str, fallback_on: bool):
    provider = get_provider(name)
    qm_provider = None if qm_name in ("aus", "off") else get_provider(qm_name)
    fallback = build_fallback_provider() if fallback_on else None
    return provider, qm_provider, fallback


def _render(result: PipelineResult, T: dict) -> None:
    if result.note is None:
        st.error(T["failed"])
    elif result.needs_human_review:
        st.warning(T["review"])
    else:
        st.success(T["ready"])

    left, right = st.columns([1, 1])

    with left:
        st.subheader(T["transcript"])
        if result.transcription is not None:
            st.write(result.transcription.text)
            st.caption(
                T["transcript_stats"].format(
                    model=result.transcription.model_size,
                    audio=result.transcription.duration_s,
                    latency=result.transcription.latency_s,
                )
            )
        st.subheader(T["redaction"])
        st.code(result.redaction.text, language=None)
        if result.redaction.mapping:
            st.caption(
                T["redaction_map"]
                + ", ".join(f"{k} ↔ {v}" for k, v in result.redaction.mapping.items())
            )
        else:
            st.caption(T["redaction_none"])

    with right:
        if result.note is not None:
            note = result.note
            st.subheader(T["note"])
            st.markdown(
                f"**{T['resident']}:** {note.resident_ref or '–'} &nbsp;·&nbsp; "
                f"**{T['time']}:** {note.time_mention or '–'} &nbsp;·&nbsp; "
                f"**{T['entry_type']}:** {note.entry_type.value}"
            )
            for obs in note.observations:
                with st.container(border=True):
                    st.markdown(f"**{T['themenfelder'][obs.themenfeld.value]}** — {obs.text}")
                    if obs.measurement:
                        st.markdown(
                            f"{T['measurement']}: "
                            f"`{obs.measurement.value} {obs.measurement.unit}`"
                        )
                    st.caption(f"{T['evidence']}: „{obs.evidence_span}“")
            if note.risk_flags:
                st.markdown(
                    f"**{T['risks']}:** "
                    + ", ".join(f"⚠️ {T['risk_labels'][f.risk.value]}" for f in note.risk_flags)
                )
            if note.follow_up_actions:
                st.markdown(f"**{T['follow_ups']}:**")
                for action in note.follow_up_actions:
                    st.markdown(f"- {action}")
            with st.expander(T["json_export"]):
                st.json(note.model_dump(mode="json"))

    st.subheader(T["qm_section"])
    if T["qm_language_hint"]:
        st.caption(T["qm_language_hint"])
    if result.qm is not None:
        for flag in result.qm.flags:
            icon = "🛑" if flag.severity == "warning" else "ℹ️"
            st.markdown(f"{icon} `{flag.code}` — {flag.message}")
        if not result.qm.flags:
            st.markdown(T["qm_clean"])
    for attempt in result.extraction.attempts:
        status = (
            T["attempt_ok"] if attempt.ok else T["attempt_failed"].format(error=attempt.error)
        )
        st.caption(
            f"{attempt.stage}: {attempt.provider}/{attempt.model} — {status}, "
            f"{attempt.llm.latency_s:.1f}s"
        )
    st.caption(
        T["totals"].format(
            cost=result.total_cost_usd, attempts=len(result.extraction.attempts)
        )
    )


def main() -> None:
    load_dotenv()
    T = t()

    with st.sidebar:
        st.divider()
        provider_name = st.selectbox(T["extraction"], ["ollama", "anthropic", "openai"], index=0)
        qm_choice = st.selectbox(
            T["qm_check"], ["anthropic", "openai", "ollama", T["qm_off"]], index=0
        )
        use_fallback = st.checkbox(T["fallback"], value=True)
        st.divider()
        sample = st.selectbox(
            T["sample"], sorted(p.name for p in AUDIO_DIR.glob("note_*.mp3"))
        )
        uploaded = st.file_uploader(T["upload"], type=["mp3", "wav", "m4a"])

    st.title("CareNote AI")
    st.caption(T["caption"])

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            audio_path = Path(tmp.name)
        st.audio(uploaded)
    else:
        audio_path = AUDIO_DIR / sample
        st.audio(str(audio_path))

    if st.button(T["process"], type="primary"):
        provider, qm_provider, fallback = _providers(provider_name, qm_choice, use_fallback)
        with st.spinner(T["spinner"]):
            result = process_audio(
                audio_path, provider, fallback=fallback, qm_provider=qm_provider
            )
        # Streamlit re-runs the script on every widget change (language toggle
        # included), so the result must survive in session state to stay visible.
        st.session_state["last_result"] = result
        st.session_state["last_source"] = uploaded.name if uploaded is not None else sample

    if "last_result" in st.session_state:
        st.caption(f"🎧 {st.session_state['last_source']}")
        _render(st.session_state["last_result"], T)
