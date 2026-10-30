# CareNote AI

German caregiver voice notes → validated, structured care documentation
(Strukturmodell: Berichteblatt entries + Risikomatrix flags mapped to the six
SIS Themenfelder), with an eval harness comparing three LLM providers on
accuracy, hallucination rate, latency and cost per note.

> **Status: under construction.** See [SPEC.md](SPEC.md) for the full build
> specification. Results table and architecture will land here as the build
> progresses.

## Quickstart

```bash
uv sync
cp .env.example .env   # optional — runs with zero paid keys via Ollama default
uv run pytest
```

## Pipeline

```text
audio (.mp3) → transcribe → redact PII → extract (LLM) → validate (retry/fallback) → QM check → structured CareNote (JSON)
```
