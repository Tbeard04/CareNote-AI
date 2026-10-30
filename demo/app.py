"""Streamlit demo: voice note in, validated structured care note out.

Run:  uv run --group demo streamlit run demo/app.py

One file on purpose. The demo shows every pipeline stage — transcript,
pseudonymisation, validated extraction with evidence spans, QM flags — because
the stages ARE the product: a draft a caregiver confirms in seconds, never an
auto-committed entry. The UI is bilingual (DE/EN); the note content itself is
and stays German, since that is the data.
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

STRINGS = {
    "de": {
        "caption": (
            "Deutsche Pflege-Sprachnotiz → validierter, strukturierter "
            "Berichteblatt-Eintrag (Strukturmodell). Entwurf + Bestätigung, "
            "niemals automatisch signiert."
        ),
        "extraction": "Extraktion",
        "qm_check": "QM-Prüfung",
        "qm_off": "aus",
        "fallback": "Fallback auf stärkeres Modell",
        "sample": "Beispielnotiz",
        "upload": "…oder eigene Audiodatei",
        "process": "Notiz verarbeiten",
        "spinner": "Transkribieren, pseudonymisieren, extrahieren, prüfen…",
        "failed": "Extraktion fehlgeschlagen — Eintrag braucht menschliche Bearbeitung.",
        "review": "Entwurf erstellt — QM-Hinweise müssen vor dem Signieren geklärt werden.",
        "ready": "Entwurf erstellt — bereit zur Bestätigung durch die Pflegekraft.",
        "transcript": "Transkript",
        "transcript_stats": (
            "faster-whisper ({model}), {audio:.0f}s Audio, {latency:.1f}s Transkription"
        ),
        "redaction": "Pseudonymisierung",
        "redaction_map": "Vor jedem externen API-Aufruf ersetzt: ",
        "redaction_none": "Keine personenbezogenen Namen erkannt.",
        "note": "Strukturierter Eintrag",
        "resident": "Bewohner:in",
        "time": "Zeit",
        "entry_type": "Art",
        "measurement": "Messwert",
        "evidence": "Beleg",
        "risks": "Risiken",
        "follow_ups": "Weiteres Vorgehen",
        "json_export": "JSON (Export-Format)",
        "qm_section": "Qualitätsprüfung & Verlauf",
        "qm_clean": "✅ Keine QM-Auffälligkeiten.",
        "attempt_ok": "ok",
        "attempt_failed": "fehlgeschlagen ({error})",
        "totals": "Gesamtkosten: ${cost:.5f} · Versuche: {attempts}",
        "qm_language_hint": "",
        "themenfelder": {
            "kognition_und_kommunikation": "Kognition & Kommunikation",
            "mobilitaet_und_bewegung": "Mobilität & Bewegung",
            "krankheitsbezogene_anforderungen": "Krankheitsbezogene Anforderungen",
            "selbstversorgung": "Selbstversorgung",
            "leben_in_sozialen_beziehungen": "Soziale Beziehungen",
            "wohnen_und_haeuslichkeit": "Wohnen & Häuslichkeit",
        },
        "risk_labels": {
            "sturz": "Sturz",
            "dekubitus": "Dekubitus",
            "dehydration": "Dehydration",
            "schmerz": "Schmerz",
        },
    },
    "en": {
        "caption": (
            "German caregiver voice note → validated, structured care-record "
            "entry (Strukturmodell). Draft + confirmation, never auto-signed."
        ),
        "extraction": "Extraction model",
        "qm_check": "QM check model",
        "qm_off": "off",
        "fallback": "Fall back to a stronger model",
        "sample": "Sample note",
        "upload": "…or upload your own audio",
        "process": "Process note",
        "spinner": "Transcribing, pseudonymising, extracting, checking…",
        "failed": "Extraction failed — this entry needs human handling.",
        "review": "Draft created — QM findings must be resolved before signing.",
        "ready": "Draft created — ready for the caregiver to confirm.",
        "transcript": "Transcript",
        "transcript_stats": (
            "faster-whisper ({model}), {audio:.0f}s audio, {latency:.1f}s to transcribe"
        ),
        "redaction": "Pseudonymisation",
        "redaction_map": "Replaced before any external API call: ",
        "redaction_none": "No personal names detected.",
        "note": "Structured entry",
        "resident": "Resident",
        "time": "Time",
        "entry_type": "Type",
        "measurement": "Measurement",
        "evidence": "Evidence",
        "risks": "Risks",
        "follow_ups": "Follow-up actions",
        "json_export": "JSON (export format)",
        "qm_section": "Quality check & audit trail",
        "qm_clean": "✅ No QM findings.",
        "attempt_ok": "ok",
        "attempt_failed": "failed ({error})",
        "totals": "Total cost: ${cost:.5f} · attempts: {attempts}",
        "qm_language_hint": (
            "Note content and QM findings are in German — the language of the "
            "care record itself."
        ),
        "themenfelder": {
            "kognition_und_kommunikation": "Cognition & communication (Kognition & Kommunikation)",
            "mobilitaet_und_bewegung": "Mobility (Mobilität & Bewegung)",
            "krankheitsbezogene_anforderungen": (
                "Illness-related demands (Krankheitsbezogene Anforderungen)"
            ),
            "selbstversorgung": "Self-care (Selbstversorgung)",
            "leben_in_sozialen_beziehungen": "Social relationships (Soziale Beziehungen)",
            "wohnen_und_haeuslichkeit": "Housing (Wohnen & Häuslichkeit)",
        },
        "risk_labels": {
            "sturz": "fall (Sturz)",
            "dekubitus": "pressure ulcer (Dekubitus)",
            "dehydration": "dehydration",
            "schmerz": "pain (Schmerz)",
        },
    },
}

load_dotenv()
st.set_page_config(page_title="CareNote AI", page_icon="📝", layout="wide")

with st.sidebar:
    lang = "en" if st.toggle("🇬🇧 English UI", key="english_ui") else "de"
    T = STRINGS[lang]
    st.divider()
    provider_name = st.selectbox(T["extraction"], ["ollama", "anthropic", "openai"], index=0)
    qm_choice = st.selectbox(
        T["qm_check"], ["anthropic", "openai", "ollama", T["qm_off"]], index=0
    )
    use_fallback = st.checkbox(T["fallback"], value=True)
    st.divider()
    sample = st.selectbox(T["sample"], sorted(p.name for p in AUDIO_DIR.glob("note_*.mp3")))
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


@st.cache_resource(show_spinner=False)
def _providers(name: str, qm_name: str, fallback_on: bool):
    provider = get_provider(name)
    qm_provider = None if qm_name in ("aus", "off") else get_provider(qm_name)
    fallback = build_fallback_provider() if fallback_on else None
    return provider, qm_provider, fallback


def render(result: PipelineResult) -> None:
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
                            f"{T['measurement']}: `{obs.measurement.value} {obs.measurement.unit}`"
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


if st.button(T["process"], type="primary"):
    provider, qm_provider, fallback = _providers(provider_name, qm_choice, use_fallback)
    with st.spinner(T["spinner"]):
        result = process_audio(audio_path, provider, fallback=fallback, qm_provider=qm_provider)
    render(result)
