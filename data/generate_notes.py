"""Synthetic German caregiver voice notes: scripted texts -> TTS audio.

The 100 note scripts live in note_scripts.json; each is written around a
deliberate edge case (fillers, self-corrections, ambiguous dosages, two
residents in one note, missing information, dialect flavour, contradictions,
spoken-number vitals) and classed by length (kurz/mittel/lang), because the
eval has to measure exactly those failure modes. Voices and speaking rates
rotate across edge-tts's German, Austrian and Swiss neural voices.

Outputs, per note:
    data/scripted/note_XX.txt  — the clean script (ablation input: clean text
                                 vs ASR output isolates speech errors)
    data/audio/note_XX.mp3     — the spoken version (skipped if it exists;
                                 pass --force to regenerate)

Run:  uv run python data/generate_notes.py [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import edge_tts

DATA_DIR = Path(__file__).parent
SCRIPTS_FILE = DATA_DIR / "note_scripts.json"
SCRIPTED_DIR = DATA_DIR / "scripted"
AUDIO_DIR = DATA_DIR / "audio"


@dataclass(frozen=True)
class NoteScript:
    note_id: str
    text: str
    voice: str
    rate: str = "+0%"
    length: str = "mittel"  # kurz | mittel | lang
    # What this note deliberately tests; ends up in the eval report.
    edge_case: str = ""


def load_notes() -> list[NoteScript]:
    entries = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
    return [
        NoteScript(
            note_id=e["id"],
            text=e["text"],
            voice=e["voice"],
            rate=e.get("rate", "+0%"),
            length=e.get("length", "mittel"),
            edge_case=e.get("edge_case", ""),
        )
        for e in entries
    ]


NOTES: list[NoteScript] = load_notes()


async def synthesise(note: NoteScript) -> None:
    communicate = edge_tts.Communicate(note.text, note.voice, rate=note.rate)
    await communicate.save(str(AUDIO_DIR / f"{note.note_id}.mp3"))


async def generate_all(force: bool = False) -> None:
    SCRIPTED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for note in NOTES:
        (SCRIPTED_DIR / f"{note.note_id}.txt").write_text(note.text + "\n", encoding="utf-8")
        audio_path = AUDIO_DIR / f"{note.note_id}.mp3"
        if audio_path.exists() and not force:
            continue
        await synthesise(note)
        print(f"{note.note_id}: {note.voice} ({note.rate}, {note.length}) — {note.edge_case}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scripted texts and TTS audio.")
    parser.add_argument("--force", action="store_true", help="regenerate existing audio too")
    args = parser.parse_args()
    asyncio.run(generate_all(force=args.force))
