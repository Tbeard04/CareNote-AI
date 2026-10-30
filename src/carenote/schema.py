"""Pydantic models for structured care documentation.

The schema follows the German Strukturmodell: a caregiver's daily voice note is
a Berichteblatt entry (deviations from the care plan), with each observation
mapped to one of the six SIS Themenfelder and risks drawn from the
Risikomatrix. Absence is a first-class value: anything the caregiver did not
say is an explicit ``null``, never an invented default.

Evidence spans are the anti-hallucination mechanism: every observation and
risk flag must quote the transcript verbatim. A claim with no matching span in
the transcript counts as unsupported (a lower bound on hallucination).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Themenfeld(StrEnum):
    """The six SIS Themenfelder of the Strukturmodell."""

    KOGNITION_UND_KOMMUNIKATION = "kognition_und_kommunikation"
    MOBILITAET_UND_BEWEGUNG = "mobilitaet_und_bewegung"
    KRANKHEITSBEZOGENE_ANFORDERUNGEN = "krankheitsbezogene_anforderungen"
    SELBSTVERSORGUNG = "selbstversorgung"
    LEBEN_IN_SOZIALEN_BEZIEHUNGEN = "leben_in_sozialen_beziehungen"
    WOHNEN_UND_HAEUSLICHKEIT = "wohnen_und_haeuslichkeit"


class EntryType(StrEnum):
    """What kind of Strukturmodell entry this voice note represents."""

    BERICHTEBLATT_EINTRAG = "berichteblatt_eintrag"
    VITALWERT_MESSUNG = "vitalwert_messung"
    RISIKO_BEOBACHTUNG = "risiko_beobachtung"


class RiskType(StrEnum):
    """Risikomatrix categories tracked by the pipeline."""

    STURZ = "sturz"
    DEKUBITUS = "dekubitus"
    DEHYDRATION = "dehydration"
    SCHMERZ = "schmerz"


class StrictModel(BaseModel):
    """Base config: unknown fields are rejected, whitespace is stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Measurement(StrictModel):
    """A numeric value the caregiver stated, e.g. blood pressure or weight."""

    value: float = Field(description="Der genannte Zahlenwert.")
    unit: str = Field(min_length=1, description="Die genannte Einheit, z. B. 'mmHg', 'kg', '°C'.")


class Observation(StrictModel):
    """One deviation-style observation, mapped to a SIS Themenfeld."""

    text: str = Field(
        min_length=1,
        description="Die Beobachtung in knapper Berichteblatt-Sprache (eine Abweichung).",
    )
    themenfeld: Themenfeld = Field(
        description="Das SIS-Themenfeld, dem diese Beobachtung zugeordnet ist."
    )
    evidence_span: str = Field(
        min_length=1,
        description="Woertliches Zitat aus dem Transkript, das diese Beobachtung belegt.",
    )
    measurement: Measurement | None = Field(
        default=None,
        description="Zahlenwert mit Einheit, falls genannt — sonst null.",
    )


class RiskFlag(StrictModel):
    """A Risikomatrix flag raised by the note."""

    risk: RiskType = Field(description="Die Risikokategorie.")
    evidence_span: str = Field(
        min_length=1,
        description="Woertliches Zitat aus dem Transkript, das dieses Risiko belegt.",
    )
    note: str | None = Field(
        default=None, description="Kurze Einordnung des Risikos, falls hilfreich — sonst null."
    )


class CareNote(StrictModel):
    """One structured care-documentation entry extracted from a voice note."""

    resident_ref: str | None = Field(
        default=None,
        description=(
            "Referenz auf die Bewohnerin / den Bewohner genau wie im Transkript genannt "
            "(Name oder Pseudonym) — null, wenn niemand genannt wird."
        ),
    )
    time_mention: str | None = Field(
        default=None,
        description=(
            "Zeitangabe woertlich wie im Transkript genannt (z. B. 'heute Morgen', "
            "'gegen 14 Uhr') — null, wenn keine genannt wird."
        ),
    )
    entry_type: EntryType = Field(description="Art des Eintrags.")
    observations: list[Observation] = Field(
        default_factory=list, description="Alle Beobachtungen aus der Notiz."
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list, description="Alle erkannten Risiken aus der Notiz."
    )
    follow_up_actions: list[str] = Field(
        default_factory=list,
        description="Vom Pflegenden genannte naechste Schritte — leer, wenn keine genannt werden.",
    )

    @field_validator("follow_up_actions")
    @classmethod
    def _no_empty_actions(cls, v: list[str]) -> list[str]:
        if any(not action.strip() for action in v):
            raise ValueError("follow_up_actions darf keine leeren Eintraege enthalten")
        return v

    def unsupported_claims(self, transcript: str) -> list[str]:
        """Return evidence spans that do not occur verbatim in the transcript.

        This is the unsupported-claim check: spans are matched after whitespace
        normalisation (ASR output and model quotes differ in spacing), but no
        fuzzy matching — a span must be a real quote.
        """
        norm_transcript = _normalise(transcript)
        missing = []
        for span in self._all_evidence_spans():
            if _normalise(span) not in norm_transcript:
                missing.append(span)
        return missing

    def _all_evidence_spans(self) -> list[str]:
        spans = [obs.evidence_span for obs in self.observations]
        spans += [flag.evidence_span for flag in self.risk_flags]
        return spans


def _normalise(text: str) -> str:
    """Lowercase and collapse all whitespace so quoting is layout-insensitive."""
    return " ".join(text.lower().split())
