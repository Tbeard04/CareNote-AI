"""Guide page: the landing page that explains what everything is and does.

A written walkthrough was chosen over a spotlight-overlay tour on purpose:
Streamlit sandboxes custom JavaScript, so overlay tours need fragile
third-party components — a clear bilingual guide with page links is robust
and reviewable.
"""

from __future__ import annotations

import streamlit as st

from i18n import lang

GUIDE_DE = {
    "intro": """
CareNote AI verwandelt eine **gesprochene deutsche Pflegenotiz** in einen
**validierten, strukturierten Berichteblatt-Eintrag** nach dem Strukturmodell —
als Entwurf, den eine Pflegekraft bestätigt. Nichts wird automatisch signiert.

```text
Audio → Transkription → Pseudonymisierung → Extraktion (LLM) → Validierung → QM-Prüfung → Eintrag
```
""",
    "steps_title": "In drei Schritten ausprobieren",
    "steps": """
1. 📝 Auf **Notiz verarbeiten** eine Beispielnotiz wählen (oder eigene Audiodatei hochladen).
2. Den roten Button **Notiz verarbeiten** drücken und ~20 Sekunden warten.
3. 📊 Unter **Ergebnisse** ansehen, wie vier Modelle auf 16 Notizen abschneiden.
""",
    "process_title": "📝 Seite: Notiz verarbeiten",
    "process_controls": """
**Einstellungen (Seitenleiste):**

| Einstellung | Bedeutung |
|---|---|
| **Extraktion** | Welches Modell die Notiz strukturiert. `ollama` = lokales Llama auf der eigenen GPU (kostenlos), `anthropic` = Claude Haiku, `openai` = GPT-4o-mini. |
| **QM-Prüfung** | Welches Modell den fertigen Eintrag auf Widersprüche und Lücken prüft (`aus` = nur Regelprüfung). |
| **Fallback** | Wenn das Extraktionsmodell wiederholt ungültige Ausgaben liefert, übernimmt einmalig ein stärkeres Modell. Greift nur bei echten Validierungsfehlern. |
| **Beispielnotiz / Upload** | 100 synthetische Notizen mit eingebauten Stolperfallen (Dialekt, Füllwörter, zwei Bewohner, unklare Dosierung) — oder eigenes Audio. |

**Ergebnisbereiche:**

- **Transkript** — was faster-whisper (lokal) verstanden hat.
- **Pseudonymisierung** — Namen und Zimmernummern werden durch Platzhalter (`B1`, `Z1`) ersetzt, *bevor* irgendetwas an eine externe API geht. Genau dieser Text geht ans Modell.
- **Strukturierter Eintrag** — Beobachtungen je SIS-Themenfeld mit wörtlichem **Beleg**-Zitat (Anti-Halluzinations-Mechanismus), Risiken, Messwerte, weiteres Vorgehen.
- **Qualitätsprüfung & Verlauf** — 🛑 blockiert das Signieren, ℹ️ ist ein Hinweis. Darunter: welche Modellversuche liefen, wie lange, was sie kosteten.
""",
    "results_title": "📊 Seite: Ergebnisse",
    "results_body": """
Die Auswertung von **100 Notizen × 6 Modellen × 2 Bedingungen** (sauberes Skript vs. ASR-Transkript):

- **Kennzahlen oben** — bestes Modell, günstigstes API-Modell, mittlere Worterkennungsfehlerrate.
- **F1-Diagramm** — Extraktionsqualität je Modell; der Abstand zwischen blau (sauber) und orange (ASR) zeigt, was Spracherkennung kostet.
- **Kosten & Latenz** — was 1.000 Notizen kosten würden und wie schnell jedes Modell antwortet.
- **Heatmap** — eine Zeile pro Notiz: helle Zellen zeigen, *welche* Notiz ein Modell verpatzt hat.
- **Detailansicht** — Modell-Ausgabe und Gold-Label nebeneinander, mit ASR-Transkript und WER.
""",
    "privacy_title": "🔒 Datenschutz-Hinweis",
    "privacy": """
Alle Notizen sind synthetisch (TTS-Stimmen, erfundene Namen). Die Pseudonymisierung
läuft trotzdem vor jedem API-Aufruf — so wie es bei echten Gesundheitsdaten (Art. 9
DSGVO) Pflicht wäre. Ehrliche Einordnung: Pseudonymisierte Daten bleiben
personenbezogen; ein echter Einsatz bräuchte AVV, EU-Endpunkte ohne Speicherung
oder Self-Hosting.
""",
    "cta_sidebar": (
        "Links in der Seitenleiste navigieren: **📝 Notiz verarbeiten** zum Ausprobieren, "
        "**📊 Ergebnisse** für den Modellvergleich. Der Schalter darunter stellt die "
        "Oberfläche auf Englisch um."
    ),
}

GUIDE_EN = {
    "intro": """
CareNote AI turns a **spoken German caregiver note** into a **validated,
structured care-record entry** following the German Strukturmodell — as a
draft a caregiver confirms. Nothing is ever auto-signed.

```text
audio → transcription → pseudonymisation → extraction (LLM) → validation → QM check → entry
```
""",
    "steps_title": "Try it in three steps",
    "steps": """
1. 📝 On **Process a note**, pick a sample note (or upload your own audio).
2. Hit the red **Process note** button and wait ~20 seconds.
3. 📊 Open **Results** to see how four models compare across 16 notes.
""",
    "process_title": "📝 Page: Process a note",
    "process_controls": """
**Settings (sidebar):**

| Setting | What it does |
|---|---|
| **Extraction model** | Which model structures the note. `ollama` = local Llama on your own GPU (free), `anthropic` = Claude Haiku, `openai` = GPT-4o-mini. |
| **QM check model** | Which model reviews the finished entry for contradictions and omissions (`off` = rule checks only). |
| **Fallback** | If the extraction model keeps producing invalid output, a stronger model takes over once. Triggers only on real validation failures. |
| **Sample / upload** | 100 synthetic notes with built-in traps (dialect, filler words, two residents, ambiguous dosage) — or your own audio. |

**Output areas:**

- **Transcript** — what faster-whisper (local) heard.
- **Pseudonymisation** — names and room numbers become placeholders (`B1`, `Z1`) *before* anything reaches an external API. This exact text is what the model sees.
- **Structured entry** — observations per SIS care domain, each with a verbatim **evidence** quote (the anti-hallucination mechanism), risks, measurements, follow-ups.
- **Quality check & audit trail** — 🛑 blocks signing, ℹ️ is advisory. Below it: which model attempts ran, how long they took, what they cost.
""",
    "results_title": "📊 Page: Results",
    "results_body": """
The evaluation of **100 notes × 6 models × 2 conditions** (clean script vs ASR transcript):

- **Headline tiles** — best model, cheapest API model, mean word error rate.
- **F1 chart** — extraction quality per model; the gap between blue (clean) and orange (ASR) is what speech recognition costs.
- **Cost & latency** — what 1,000 notes would cost and how fast each model responds.
- **Heatmap** — one row per note: pale cells show *which* note a model fumbled.
- **Drill-down** — model output and gold label side by side, with the ASR transcript and its WER.
""",
    "privacy_title": "🔒 Privacy note",
    "privacy": """
All notes are synthetic (TTS voices, invented names). Pseudonymisation still runs
before every API call — as it would have to for real health data (GDPR Art. 9).
Honest scope: pseudonymised data is still personal data; a real deployment needs
DPAs, zero-retention EU endpoints, or self-hosting.
""",
    "cta_sidebar": (
        "Navigate with the sidebar on the left: **📝 Process a note** to try it, "
        "**📊 Results** for the model comparison. The toggle below it switches the "
        "UI language."
    ),
}


def main() -> None:
    G = GUIDE_EN if lang() == "en" else GUIDE_DE

    st.title("CareNote AI")
    st.markdown(G["intro"])

    st.info(G["cta_sidebar"], icon="👈")

    st.subheader(G["steps_title"])
    st.markdown(G["steps"])

    with st.expander(G["process_title"], expanded=True):
        st.markdown(G["process_controls"])
    with st.expander(G["results_title"]):
        st.markdown(G["results_body"])
    with st.expander(G["privacy_title"]):
        st.markdown(G["privacy"])
