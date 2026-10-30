"""Transcript -> structured CareNote extraction.

One shared prompt for every provider. The schema travels twice: embedded in
the prompt (for providers whose JSON mode takes no schema) and as a native
JSON schema (for providers that enforce one). Parsing is strict: this module
raises ``ExtractionParseError`` on bad output — the repair/retry/fallback
chain lives in validate.py and decides what happens next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from carenote.providers import LLMProvider, LLMResult
from carenote.schema import CareNote

SYSTEM_PROMPT = """\
Du bist ein Dokumentationsassistent für die ambulante und stationäre Pflege in \
Deutschland. Du wandelst das Transkript einer gesprochenen Pflegenotiz in einen \
strukturierten Berichteblatt-Eintrag nach dem Strukturmodell um.

Regeln — halte dich strikt daran:
1. Antworte AUSSCHLIESSLICH mit einem einzigen JSON-Objekt nach dem unten \
angegebenen Schema. Kein Markdown, kein Text davor oder danach.
2. Jede Beobachtung und jedes Risiko braucht einen evidence_span: ein \
WÖRTLICHES Zitat aus dem Transkript, zeichengenau kopiert. Erfinde nichts, \
paraphrasiere nicht, fasse im evidence_span keine Textstellen zusammen.
3. Was im Transkript NICHT gesagt wird, ist null (bzw. eine leere Liste). \
Rate niemals Werte, Namen oder Zeiten.
4. Ordne jede Beobachtung genau einem der sechs SIS-Themenfelder zu.
5. Risiken nur aus diesen Kategorien: sturz, dekubitus, dehydration, schmerz.
6. Werden mehrere Bewohner genannt, dokumentiere nur die Person, um die es \
hauptsächlich geht, und notiere die zweite Person NICHT als eigene Beobachtung.
7. Unklare Angaben (z. B. 'die übliche Dosis') übernimmst du wörtlich als \
Beobachtung, ohne sie zu interpretieren oder zu beziffern.
8. Ein unauffälliger Verlauf ('alles unauffällig', 'keine Besonderheiten') \
ist KEINE Beobachtung: observations, risk_flags und follow_up_actions \
bleiben dann leer. Das Berichteblatt dokumentiert nur Abweichungen.

JSON-Schema:
{schema}
"""

USER_PROMPT = """\
Transkript der Sprachnotiz:

---
{transcript}
---

Erzeuge den strukturierten Eintrag als JSON.
"""


class ExtractionParseError(Exception):
    """The model's output was not a valid CareNote; carries the exact error."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


@dataclass(frozen=True)
class ExtractionResult:
    note: CareNote
    llm: LLMResult


def build_prompts(transcript: str) -> tuple[str, str]:
    """Return the (system, user) prompt pair shared by every provider."""
    schema = json.dumps(CareNote.model_json_schema(), ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT.format(schema=schema), USER_PROMPT.format(transcript=transcript)


def strip_to_json(text: str) -> str:
    """Cut away code fences and prose around the first top-level JSON object."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ExtractionParseError("Kein JSON-Objekt in der Antwort gefunden", text)
    return text[start : end + 1]


def parse_care_note(text: str) -> CareNote:
    """Parse model output into a CareNote; raise with the exact error otherwise."""
    try:
        return CareNote.model_validate_json(strip_to_json(text))
    except ValidationError as exc:
        raise ExtractionParseError(str(exc), text) from exc


def extract_note(transcript: str, provider: LLMProvider) -> ExtractionResult:
    """One extraction attempt: prompt -> provider -> strictly parsed CareNote."""
    system, user = build_prompts(transcript)
    result = provider.complete(system, user, json_schema=CareNote.model_json_schema())
    return ExtractionResult(note=parse_care_note(result.text), llm=result)


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    from carenote.providers import get_provider

    # Windows consoles default to a legacy codepage; the notes are German.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extract a CareNote from a transcript file.")
    parser.add_argument("transcript", type=Path, help="Pfad zur Transkript-Textdatei")
    parser.add_argument("--provider", default=None, help="anthropic | openai | ollama")
    args = parser.parse_args()

    provider = get_provider(args.provider)
    result = extract_note(args.transcript.read_text(encoding="utf-8"), provider)
    print(result.note.model_dump_json(indent=2))
    print(
        f"\n[{result.llm.provider}/{result.llm.model}] "
        f"{result.llm.input_tokens}->{result.llm.output_tokens} tokens, "
        f"{result.llm.latency_s:.2f}s, ${result.llm.cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()
