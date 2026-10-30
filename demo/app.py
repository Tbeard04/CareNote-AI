"""Streamlit demo: voice note in, validated structured care note out.

Run:  uv run --group demo streamlit run demo/app.py

One file on purpose. The demo shows every pipeline stage — transcript,
pseudonymisation, validated extraction with evidence spans, QM flags — because
the stages ARE the product: a draft a caregiver confirms in seconds, never an
auto-committed entry.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from carenote.pipeline import PipelineResult, process_audio  # noqa: E402
from carenote.providers import get_provider  # noqa: E402
from carenote.validate import build_fallback_provider  # noqa: E402

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"
THEMENFELD_LABELS = {
    "kognition_und_kommunikation": "Kognition & Kommunikation",
    "mobilitaet_und_bewegung": "Mobilität & Bewegung",
    "krankheitsbezogene_anforderungen": "Krankheitsbezogene Anforderungen",
    "selbstversorgung": "Selbstversorgung",
    "leben_in_sozialen_beziehungen": "Soziale Beziehungen",
    "wohnen_und_haeuslichkeit": "Wohnen & Häuslichkeit",
}

load_dotenv()
st.set_page_config(page_title="CareNote AI", page_icon="📝", layout="wide")
st.title("CareNote AI")
st.caption(
    "Deutsche Pflege-Sprachnotiz → validierter, strukturierter Berichteblatt-Eintrag "
    "(Strukturmodell). Entwurf + Bestätigung, niemals automatisch signiert."
)

with st.sidebar:
    provider_name = st.selectbox("Extraktion", ["ollama", "anthropic", "openai"], index=0)
    qm_choice = st.selectbox("QM-Prüfung", ["anthropic", "openai", "ollama", "aus"], index=0)
    use_fallback = st.checkbox("Fallback auf stärkeres Modell", value=True)
    st.divider()
    sample = st.selectbox(
        "Beispielnotiz", sorted(p.name for p in AUDIO_DIR.glob("note_*.mp3"))
    )
    uploaded = st.file_uploader("…oder eigene Audiodatei", type=["mp3", "wav", "m4a"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        audio_path = Path(tmp.name)
    st.audio(uploaded)
else:
    audio_path = AUDIO_DIR / sample
    st.audio(str(audio_path))


@st.cache_resource(show_spinner=False)
def _providers(name: str, qm_name: str, fallback_on: bool):
    provider = get_provider(name)
    qm_provider = None if qm_name == "aus" else get_provider(qm_name)
    fallback = build_fallback_provider() if fallback_on else None
    return provider, qm_provider, fallback


def render(result: PipelineResult) -> None:
    review = result.needs_human_review
    if result.note is None:
        st.error("Extraktion fehlgeschlagen — Eintrag braucht menschliche Bearbeitung.")
    elif review:
        st.warning("Entwurf erstellt — QM-Hinweise müssen vor dem Signieren geklärt werden.")
    else:
        st.success("Entwurf erstellt — bereit zur Bestätigung durch die Pflegekraft.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Transkript")
        if result.transcription is not None:
            st.write(result.transcription.text)
            st.caption(
                f"faster-whisper ({result.transcription.model_size}), "
                f"{result.transcription.duration_s:.0f}s Audio, "
                f"{result.transcription.latency_s:.1f}s Transkription"
            )
        st.subheader("Pseudonymisierung")
        st.code(result.redaction.text, language=None)
        if result.redaction.mapping:
            st.caption(
                "Vor jedem externen API-Aufruf ersetzt: "
                + ", ".join(f"{k} ↔ {v}" for k, v in result.redaction.mapping.items())
            )
        else:
            st.caption("Keine personenbezogenen Namen erkannt.")

    with right:
        if result.note is not None:
            note = result.note
            st.subheader("Strukturierter Eintrag")
            st.markdown(
                f"**Bewohner:in:** {note.resident_ref or '–'} &nbsp;·&nbsp; "
                f"**Zeit:** {note.time_mention or '–'} &nbsp;·&nbsp; "
                f"**Art:** {note.entry_type.value}"
            )
            for obs in note.observations:
                with st.container(border=True):
                    st.markdown(f"**{THEMENFELD_LABELS[obs.themenfeld.value]}** — {obs.text}")
                    if obs.measurement:
                        st.markdown(f"Messwert: `{obs.measurement.value} {obs.measurement.unit}`")
                    st.caption(f"Beleg: „{obs.evidence_span}“")
            if note.risk_flags:
                st.markdown(
                    "**Risiken:** "
                    + ", ".join(f"⚠️ {f.risk.value}" for f in note.risk_flags)
                )
            if note.follow_up_actions:
                st.markdown("**Weiteres Vorgehen:**")
                for action in note.follow_up_actions:
                    st.markdown(f"- {action}")
            with st.expander("JSON (Export-Format)"):
                st.json(note.model_dump(mode="json"))

    st.subheader("Qualitätsprüfung & Verlauf")
    if result.qm is not None:
        for flag in result.qm.flags:
            icon = "🛑" if flag.severity == "warning" else "ℹ️"
            st.markdown(f"{icon} `{flag.code}` — {flag.message}")
        if not result.qm.flags:
            st.markdown("✅ Keine QM-Auffälligkeiten.")
    for attempt in result.extraction.attempts:
        status = "ok" if attempt.ok else f"fehlgeschlagen ({attempt.error})"
        st.caption(
            f"{attempt.stage}: {attempt.provider}/{attempt.model} — {status}, "
            f"{attempt.llm.latency_s:.1f}s"
        )
    st.caption(
        f"Gesamtkosten: ${result.total_cost_usd:.5f} · "
        f"Versuche: {len(result.extraction.attempts)}"
    )


if st.button("Notiz verarbeiten", type="primary"):
    provider, qm_provider, fallback = _providers(provider_name, qm_choice, use_fallback)
    with st.spinner("Transkribieren, pseudonymisieren, extrahieren, prüfen…"):
        result = process_audio(audio_path, provider, fallback=fallback, qm_provider=qm_provider)
    render(result)
