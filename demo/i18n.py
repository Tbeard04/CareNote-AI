"""All UI strings for the demo, German and English.

The note content itself is and stays German — that is the care record's
language. Only the UI translates.
"""

from __future__ import annotations

import streamlit as st

STRINGS = {
    "de": {
        "nav": {"guide": "Anleitung", "process": "Notiz verarbeiten", "results": "Ergebnisse"},
        "language_toggle": "🇬🇧 English UI",
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
        # --- Results page ---
        "r_title": "Eval-Ergebnisse",
        "r_no_results": "Noch keine Ergebnisse — zuerst `uv run python evals/run_eval.py` ausführen.",
        "r_caption": (
            "{runs} bewertete Läufe · {notes} Notizen · {models} Modelle · Bedingungen: "
            "sauberes Skript vs. ASR-Transkript. Reparatur-Retry an, modellübergreifender "
            "Fallback aus (faire Spalten pro Modell)."
        ),
        "r_best": "Bestes Modell (ASR, F1)",
        "r_cheapest": "Günstigstes API-Modell",
        "r_local": "Lokale Modelle",
        "r_local_delta": "0 $ Grenzkosten",
        "r_wer": "Mittlere WER (whisper-small)",
        "r_quality": "Extraktionsqualität (Beobachtungs-F1)",
        "r_quality_caption": (
            "Der Abstand sauber vs. ASR zeigt, was Spracherkennung jedes Modell kostet."
        ),
        "r_cost": "Kosten pro 1.000 Notizen",
        "r_cost_caption": "Lokale Modelle (Ollama) laufen ohne Grenzkosten auf der GPU.",
        "r_latency": "Latenz",
        "r_latency_axis": "Extraktionslatenz (s)",
        "r_f1_axis": "Beobachtungs-F1 vs. Gold",
        "r_cost_axis": "USD pro 1.000 Notizen",
        "r_table": "Gesamtvergleich",
        "r_heatmap": "Wo welches Modell Schwierigkeiten hat",
        "r_heatmap_caption": (
            "Eine Zeile pro Notiz, eine Spalte pro Modell — die hellen Zellen sind "
            "die interessanten."
        ),
        "r_drill": "Detailansicht: Modell-Ausgabe vs. Gold",
        "r_note": "Notiz",
        "r_model": "Modell",
        "r_condition": "Bedingung",
        "r_run_stats": (
            "F1 {f1:.2f} · Präzision {p:.2f} · Recall {r:.2f} · {unsup} unbelegte von "
            "{claims} Aussagen · {attempts} Versuch(e) · {latency:.1f}s · ${cost:.5f}"
        ),
        "r_asr_header": "ASR-Transkript (WER {wer:.1%} vs. Skript)",
        "r_script_header": "Sauberes Skript",
        "r_model_output": "Ausgabe von {model}",
        "r_gold": "Gold-Label",
        "r_no_prediction": "Keine gespeicherte Ausgabe — Eval erneut ausführen.",
    },
    "en": {
        "nav": {"guide": "Guide", "process": "Process a note", "results": "Results"},
        "language_toggle": "🇬🇧 English UI",
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
        # --- Results page ---
        "r_title": "Eval results",
        "r_no_results": "No eval results yet — run `uv run python evals/run_eval.py` first.",
        "r_caption": (
            "{runs} scored runs · {notes} notes · {models} models · conditions: clean "
            "script vs ASR transcript. Repair-retry on, cross-provider fallback off "
            "for fair per-model columns."
        ),
        "r_best": "Best model (ASR, F1)",
        "r_cheapest": "Cheapest API model",
        "r_local": "Local models",
        "r_local_delta": "$0 marginal cost",
        "r_wer": "Mean WER (whisper-small)",
        "r_quality": "Extraction quality (Observation F1)",
        "r_quality_caption": "The clean-vs-ASR gap isolates what speech recognition costs each model.",
        "r_cost": "Cost per 1,000 notes",
        "r_cost_caption": "Local models (Ollama) run at zero marginal cost on the GPU.",
        "r_latency": "Latency",
        "r_latency_axis": "Extraction latency (s)",
        "r_f1_axis": "Observation F1 vs gold",
        "r_cost_axis": "USD per 1,000 notes",
        "r_table": "Full comparison",
        "r_heatmap": "Where each model struggles",
        "r_heatmap_caption": (
            "One row per note, one column per model — pale cells are the interesting ones."
        ),
        "r_drill": "Per-note drill-down: model output vs gold",
        "r_note": "Note",
        "r_model": "Model",
        "r_condition": "Condition",
        "r_run_stats": (
            "F1 {f1:.2f} · precision {p:.2f} · recall {r:.2f} · {unsup} unsupported of "
            "{claims} claims · {attempts} attempt(s) · {latency:.1f}s · ${cost:.5f}"
        ),
        "r_asr_header": "ASR transcript (WER {wer:.1%} vs script)",
        "r_script_header": "Clean script",
        "r_model_output": "{model} output",
        "r_gold": "Gold label",
        "r_no_prediction": "No stored prediction — re-run the eval to persist model outputs.",
    },
}


def lang() -> str:
    return "en" if st.session_state.get("english_ui") else "de"


def t() -> dict:
    return STRINGS[lang()]
