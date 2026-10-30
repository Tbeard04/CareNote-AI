"""PII pseudonymisation before any external API call.

Care notes are special-category health data (GDPR Art. 9), so resident names
and room numbers are replaced with placeholders BEFORE a transcript leaves the
machine, and mapped back AFTER extraction. Detection combines a title regex
(Frau/Herr + name — the dominant pattern in German care speech) with spaCy's
German NER for names the regex misses.

Honest scope, stated in the README too: pseudonymised data is still personal
data under GDPR. This step reduces exposure; it does not make an external API
call compliant by itself.

Placeholders look like ``B1`` (Bewohner) and ``Z1`` (Zimmer): single tokens
that TTS, Whisper and LLM quoting all pass through unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from carenote.schema import CareNote, Observation, RiskFlag

# "Frau Weber", "Herrn Brandt", "Herr van Dyk" — title + capitalised name(s).
_TITLE_NAME_RE = re.compile(
    r"\b(?P<title>Frau|Herrn?|Bewohnerin|Bewohner)\s+"
    r"(?P<name>(?:(?:van|von|de|der)\s+)?[A-ZÄÖÜ][a-zäöüß]+)"
)
_ROOM_RE = re.compile(r"\b(?P<zimmer>Zimmer)\s+(?P<nr>[A-Za-zäöüß0-9]+)")

_nlp = None


def _ner_names(text: str) -> list[str]:
    """Person names from spaCy's German NER (lazy-loaded)."""
    global _nlp
    if _nlp is None:
        import de_core_news_sm

        _nlp = de_core_news_sm.load(disable=["tagger", "parser", "lemmatizer"])
    names = []
    for ent in _nlp(text).ents:
        if ent.label_ == "PER":
            # Strip a leading title so "Frau Weber" and "Weber" dedupe.
            names.append(re.sub(r"^(Frau|Herrn?|Bewohnerin|Bewohner)\s+", "", ent.text).strip())
    return [n for n in names if n]


@dataclass(frozen=True)
class Redaction:
    """Pseudonymised text plus the mapping to restore the original."""

    text: str
    mapping: dict[str, str]  # placeholder -> original


def redact(text: str) -> Redaction:
    """Replace names and room numbers with stable placeholders, first-seen order."""
    name_to_placeholder: dict[str, str] = {}

    # Names the title regex finds, in order of appearance…
    ordered_names = [m.group("name") for m in _TITLE_NAME_RE.finditer(text)]
    # …then NER-only finds (e.g. a full name spoken without Frau/Herr). The
    # small German NER model mislabels domain nouns ("Blutdruck") as PER, so a
    # bare NER name only counts when it is multi-token; single-token names are
    # covered by the title regex, which is how care speech names residents.
    ordered_names += [
        n for n in _ner_names(text) if n not in ordered_names and " " in n
    ]
    for name in ordered_names:
        if name not in name_to_placeholder:
            name_to_placeholder[name] = f"B{len(name_to_placeholder) + 1}"

    redacted = text
    # Longest first, so "Anna Weber" is replaced before a bare "Weber" could be.
    for name in sorted(name_to_placeholder, key=len, reverse=True):
        redacted = re.sub(rf"\b{re.escape(name)}\b", name_to_placeholder[name], redacted)

    room_mapping: dict[str, str] = {}
    def _room_sub(match: re.Match) -> str:
        placeholder = f"Z{len(room_mapping) + 1}"
        room_mapping[placeholder] = match.group("nr")
        return f"{match.group('zimmer')} {placeholder}"

    redacted = _ROOM_RE.sub(_room_sub, redacted)

    mapping = {ph: name for name, ph in name_to_placeholder.items()} | room_mapping
    return Redaction(text=redacted, mapping=mapping)


def unredact_text(text: str, mapping: dict[str, str]) -> str:
    """Swap placeholders back to the original names."""
    for placeholder in sorted(mapping, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(placeholder)}\b", mapping[placeholder], text)
    return text


def unredact_note(note: CareNote, mapping: dict[str, str]) -> CareNote:
    """Restore originals in every text field of an extracted CareNote."""

    def restore(value: str | None) -> str | None:
        return None if value is None else unredact_text(value, mapping)

    return note.model_copy(
        update={
            "resident_ref": restore(note.resident_ref),
            "time_mention": restore(note.time_mention),
            "observations": [
                Observation(
                    text=restore(o.text),
                    themenfeld=o.themenfeld,
                    evidence_span=restore(o.evidence_span),
                    measurement=o.measurement,
                )
                for o in note.observations
            ],
            "risk_flags": [
                RiskFlag(
                    risk=f.risk,
                    evidence_span=restore(f.evidence_span),
                    note=restore(f.note),
                )
                for f in note.risk_flags
            ],
            "follow_up_actions": [unredact_text(a, mapping) for a in note.follow_up_actions],
        }
    )
