"""Quality pass over an extracted note: cheap rules first, then one LLM call.

Mirrors a documentation-quality check (dexter.qm style): deterministic rules
catch what rules can catch (missing resident reference, implausible vital
values), and a second, cheap LLM call reads note + transcript together to flag
contradictions and omissions rules cannot see ("hat gut gegessen" vs "Essen
komplett verweigert").

QM never edits the note. Warnings are meant to block sign-off until a human
resolves them; infos are advisory. If the QM model call fails, that is itself
a flag — the note is then unreviewed, not silently approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from carenote.providers import LLMProvider, LLMResult
from carenote.schema import CareNote, StrictModel

# Plausible ranges per unit for spoken vital signs.
_UNIT_RANGES: dict[str, tuple[float, float]] = {
    "mmhg": (40, 260),
    "°c": (30, 45),
    "grad": (30, 45),
    "pro minute": (20, 250),
    "bpm": (20, 250),
    "kg": (25, 250),
    "%": (0, 100),
    "l": (0, 6),
    "ml": (0, 5000),
}

Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class QMFlag:
    code: str
    severity: Severity
    message: str


@dataclass(frozen=True)
class QMResult:
    flags: list[QMFlag]
    llm: LLMResult | None  # None when the LLM check didn't run

    @property
    def passed(self) -> bool:
        """True when nothing blocks sign-off (infos are fine)."""
        return all(flag.severity != "warning" for flag in self.flags)


class QMReview(StrictModel):
    """Structured output of the LLM contradiction/completeness check."""

    widersprueche: list[str]
    fehlende_angaben: list[str]


QM_SYSTEM_PROMPT = """\
Du prüfst die Qualität eines strukturierten Pflegeeintrags gegen das Transkript \
der zugrundeliegenden Sprachnotiz. Antworte AUSSCHLIESSLICH mit einem JSON-Objekt:

{"widersprueche": [...], "fehlende_angaben": [...]}

- widersprueche: Aussagen im Eintrag oder Transkript, die sich gegenseitig \
widersprechen (z. B. 'hat gut gegessen' und 'Essen komplett verweigert' zur \
selben Mahlzeit). Jede als kurzer Satz. Zwei verschiedene Mahlzeiten oder \
Zeitpunkte sind KEIN Widerspruch.
- fehlende_angaben: Wichtige Informationen, die der Pflegende anspricht, die \
aber im Eintrag fehlen. Keine Spekulation über Dinge, die nie gesagt wurden.

Beide Listen sind leer, wenn es nichts zu beanstanden gibt.
"""

QM_USER_PROMPT = """\
Transkript:
---
{transcript}
---

Strukturierter Eintrag:
---
{note_json}
---
"""


def rule_flags(note: CareNote) -> list[QMFlag]:
    """Deterministic checks, no LLM involved."""
    flags: list[QMFlag] = []
    if note.resident_ref is None:
        flags.append(
            QMFlag(
                "missing_resident",
                "warning",
                "Kein Bewohnerbezug — Eintrag kann niemandem zugeordnet werden.",
            )
        )
    if not note.observations and not note.risk_flags:
        flags.append(
            QMFlag(
                "empty_note",
                "info",
                "Keine Beobachtungen oder Risiken — bei 'alles unauffällig' in Ordnung.",
            )
        )
    for i, obs in enumerate(note.observations, start=1):
        m = obs.measurement
        if m is None:
            continue
        bounds = _UNIT_RANGES.get(m.unit.strip().lower())
        if bounds is None:
            flags.append(
                QMFlag(
                    "unknown_unit",
                    "info",
                    f"Beobachtung {i}: Einheit '{m.unit}' nicht bekannt — Wert ungeprüft.",
                )
            )
        elif not bounds[0] <= m.value <= bounds[1]:
            flags.append(
                QMFlag(
                    "implausible_value",
                    "warning",
                    f"Beobachtung {i}: {m.value} {m.unit} liegt außerhalb des "
                    f"plausiblen Bereichs {bounds[0]}–{bounds[1]}.",
                )
            )
    return flags


def llm_flags(
    note: CareNote, transcript: str, provider: LLMProvider
) -> tuple[list[QMFlag], LLMResult | None]:
    """One cheap LLM call for contradictions and omissions."""
    user = QM_USER_PROMPT.format(
        transcript=transcript, note_json=note.model_dump_json(indent=1)
    )
    try:
        result = provider.complete(QM_SYSTEM_PROMPT, user, json_schema=QMReview.model_json_schema())
        review = QMReview.model_validate_json(_to_json(result.text))
    except Exception as exc:  # noqa: BLE001 — any QM failure must surface as a flag
        return (
            [
                QMFlag(
                    "qm_unavailable",
                    "warning",
                    f"QM-Prüfung fehlgeschlagen ({type(exc).__name__}) — Eintrag ungeprüft.",
                )
            ],
            None,
        )
    flags = [QMFlag("contradiction", "warning", text) for text in review.widersprueche]
    flags += [QMFlag("missing_information", "info", text) for text in review.fehlende_angaben]
    return flags, result


def _to_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("Kein JSON-Objekt in der QM-Antwort")
    return text[start : end + 1]


def run_qm(note: CareNote, transcript: str, provider: LLMProvider | None) -> QMResult:
    """Full QM pass: rules always, LLM check when a provider is given."""
    flags = rule_flags(note)
    llm_result = None
    if provider is not None:
        extra, llm_result = llm_flags(note, transcript, provider)
        flags += extra
    return QMResult(flags=flags, llm=llm_result)
