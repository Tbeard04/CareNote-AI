# CareNote AI — System Build Specification

Oct 30, 2026 · @Thomas

## 1. Overview

CareNote AI is a pipeline that turns German caregiver voice notes into validated, structured care documentation, with an evaluation harness comparing three LLM providers on accuracy, hallucination rate, latency and cost per note.

**End-to-end flow:**

```text
audio (.mp3) → transcribe → redact PII → extract (LLM) → validate (retry/fallback) → QM check → structured CareNote (JSON)
```

### Goals

1. **Strukturmodell-correct output** — Berichteblatt-style deviation entries and Risikomatrix flags, with observations mapped to the six SIS Themenfelder.
2. **Robustness to messy spoken German** — filler words, interruptions, ambiguity, missing information; robustness is measured, not assumed.
3. **Provider-agnostic extraction** — one interface, three backends (Anthropic, OpenAI, self-hosted Llama via Ollama), swappable by config.
4. **Measured quality, latency and cost** — field-level accuracy, hallucination rate, p50/p95 latency, € per note and per 1,000 notes.
5. **Privacy by design** — PII pseudonymisation before any external API call (care notes are GDPR Art. 9 special-category data).
6. **Reproducibility** — `uv sync` and it runs; Ollama is the default provider so the full pipeline works with zero paid API keys.

### Non-goals

- No proprietary ASR training — the bet is strong engineering on top of general models.
- No agent framework — a linear pipeline doesn't need one.
- No eval framework (LangSmith, promptfoo, etc.) — hand-rolled metrics are the point.
- No production serving — the demo measures quality parity and cost, not throughput.
- No real patient data — all audio is synthetic, all labels hand-written.

## 2. Domain model

The system targets the German **Strukturmodell** (project Ein-STEP), the official lean-documentation standard since 2014/15. Its four elements:

1. **Master data sheet** — out of scope.
2. **SIS** (Strukturierte Informationssammlung) — admission-time structured collection covering **six Themenfelder**: cognition/communication, mobility, illness-related demands, self-care, social relationships, housing — plus a **Risikomatrix** (falls, pain, pressure ulcers, etc.).
3. **Maßnahmenplan** — individual care plan; out of scope.
4. **Berichteblatt** — records only *deviations* from the plan. **A caregiver's daily voice note is a Berichteblatt entry.**

**Design implication:** the schema extracts deviation-style report entries and risk flags mapped to the six Themenfelder — not a generic "medical note" schema. The Strukturmodell vocabulary is used throughout the codebase and output.

## 3. Tech stack

Deliberately boring, framework-light choices; the thin layers are hand-rolled on purpose.

| Component | Tool | Why this one |
| --- | --- | --- |
| Environment | Python 3.12 + uv | `uv sync` and it runs — zero-friction setup |
| Schema & validation | Pydantic v2 | Strict structured outputs |
| Synthetic voice notes | Scripted texts + `edge-tts` (German voices) | Free, reproducible; edge cases written in deliberately |
| Transcription | `faster-whisper` (medium, int8) | Local and free; word timestamps feed evidence spans |
| PII redaction | spaCy `de_core_news_sm` + regex | Lightweight pseudonymisation before any API call |
| LLM providers | `anthropic` + `openai` SDKs + Ollama (`llama3.1:8b`) behind a hand-rolled ~80-line `LLMProvider` protocol | Provider-agnostic as running code; no LangChain/LiteLLM |
| Prompting | Native structured-output/JSON mode, temperature 0 | Deterministic enough to eval |
| Eval metrics | Plain Python + pandas, matplotlib for one chart | Hand-rolled metrics; no eval framework |
| Tests & CI | pytest + ruff + GitHub Actions running a 3-note mini-eval | Production discipline on every push |
| Logging | stdlib `logging`, JSON lines per pipeline stage | Retry/fallback events become auditable artifacts |
| Demo UI | Streamlit | One file; first thing cut if time slips |

## 4. Repository layout

```text
carenote-ai/
├── README.md                 # results table + architecture up top
├── pyproject.toml            # uv-managed, pinned deps
├── .env.example              # every key documented; runs with zero paid keys via Ollama default
├── .github/workflows/ci.yml  # lint + tests + a 3-note mini-eval on every push
├── data/
│   ├── generate_notes.py     # synthetic German caregiver notes (text → TTS audio)
│   ├── audio/                # 15–20 .mp3 voice notes, deliberately messy
│   └── gold/                 # hand-checked gold labels, one JSON per note
├── src/carenote/
│   ├── schema.py             # Pydantic models: CareNote, Observation, RiskFlag…
│   ├── transcribe.py         # faster-whisper wrapper
│   ├── providers.py          # one LLMProvider interface: Anthropic / OpenAI / Ollama
│   ├── extract.py            # transcript → CareNote structured output
│   ├── redact.py             # PII pseudonymisation before any external API call
│   ├── validate.py           # schema validation, repair-retry, fallback, review flag
│   ├── qm.py                 # quality pass: completeness + contradiction checks
│   └── pipeline.py           # end-to-end orchestration + per-stage logging
├── evals/
│   ├── run_eval.py           # all notes × all providers
│   ├── metrics.py            # field accuracy, hallucination rate, latency, cost/note
│   └── results/results.md    # comparison table + charts
├── tests/                    # unit tests for schema, validation, metrics
└── demo/app.py               # tiny Streamlit UI: upload audio → validated note
```

## 5. Component specifications

### 5.1 Synthetic data — `data/generate_notes.py`

- Scripts 15–20 German caregiver notes as text, then renders audio with `edge-tts` (German voices).
- Edge cases written in deliberately:
  - interruptions and filler words
  - ambiguous dosages ("gib ihr die übliche Dosis")
  - two residents mentioned in one note
  - information that is simply absent (must map to explicit `null`, not an invented value)
- Gold labels are hand-corrected JSON, one file per note in `data/gold/`. The hand-labelling pass is what makes the eval credible — budget ~1 hour for it.

### 5.2 Transcription — `transcribe.py`

- `faster-whisper`, small or medium model, int8 quantisation for laptop-class hardware.
- **Word timestamps are kept** so evidence spans can point back into the audio.
- WER is reported in the eval so the ASR cost to downstream accuracy is visible (see ablation in §7).

### 5.3 Schema — `schema.py`

Strict Pydantic v2 models:

- `CareNote`: resident reference, timestamp, entry type (Berichteblatt report entry, vital-sign measurement, risk observation).
- `Observation[]`: each mapped to **one of the six SIS Themenfelder**, carrying:
  - free-text content
  - an `evidence_span` quoting the transcript verbatim
  - optional value/unit for measurements
- `RiskFlag[]`: fall, pressure ulcer, dehydration, pain.
- `follow_up_actions[]`.
- **Explicit `null` for anything not stated** — absence is a first-class value.

**Evidence spans are the anti-hallucination mechanism:** a claim with no matching span in the transcript counts as hallucinated. (Honest metric name: *unsupported-claim rate* — a lower bound on hallucination, since a model can quote the transcript and still misattribute; misattribution surfaces separately as field-level errors against gold.)

### 5.4 PII redaction — `redact.py`

- spaCy `de_core_news_sm` NER + regex for names and IDs.
- Pseudonymise **before any external API call**; re-map placeholders back after extraction.
- Known weak point: residents' names colliding with the pseudonymisation step — note it in Limitations.
- Honest scope: pseudonymised data is still personal data under GDPR. Real deployment needs DPAs, EU-resident zero-retention endpoints, or self-hosting — which is the strongest argument for the self-hosted track. State this in the README.

### 5.5 Providers — `providers.py`

- One hand-rolled `LLMProvider` protocol (~80 lines), three implementations:
  1. Anthropic (Haiku class)
  2. OpenAI (gpt-4o-mini class)
  3. Self-hosted Llama 3.1 8B via Ollama (**default** — zero paid keys)
- One shared prompt across all providers; native structured-output/JSON mode; temperature 0; provider picked by config.
- The interface is the abstraction boundary for the whole system: swapping the ASR layer later is the same adapter pattern.

### 5.6 Extraction — `extract.py`

- Transcript in → `CareNote` structured output.
- Prompt instructs: quote evidence spans verbatim, emit explicit `null` for unstated fields, never merge two residents into one note.

### 5.7 Validation + fallback — `validate.py`

Deterministic chain, triggered by **validation failure only** (never by guessed confidence):

```text
Pydantic parse fails
  → repair prompt carrying the exact validation error
  → one retry on the same model
  → fall back to a stronger model
  → still failing → flag needs_human_review
```

- **Every retry and fallback event is logged** (JSON lines) so retry rate and the cost overhead of reliability are quantifiable from the logs.

### 5.8 QM pass — `qm.py`

- Cheap second LLM call plus rule checks:
  - required fields present
  - units plausible
  - contradictions flagged (e.g. "hat gut gegessen" + "Essen komplett verweigert")
- QM flags block the note from counting as clean; in a production design they would block signing until resolved.

### 5.9 Pipeline — `pipeline.py`

- End-to-end orchestration of the stages above.
- Per-stage structured logging (stdlib `logging`, JSON lines): stage name, duration, token counts, retry/fallback events, QM flags.

### 5.10 Demo UI — `demo/app.py`

- Single-file Streamlit app: upload audio → validated note with flags.
- First thing cut if time slips (see scope guard, §8).

### 5.11 Tests & CI

- pytest unit tests for schema, validation logic, and metrics.
- ruff for lint.
- GitHub Actions on every push: lint + tests + a **3-note mini-eval** against the Ollama backend so eval regressions are caught in CI.

## 6. Human-in-the-loop design

No accuracy level auto-ships: a wrong entry has legal weight (the Medizinischer Dienst audits documentation). The design is **draft + mandatory caregiver confirmation**, with QM flags blocking sign-off until resolved. The system's job is to make confirmation take seconds, not to remove the human. In production the key metric becomes **caregiver edit-rate** — how much of each draft gets corrected before signing — which turns every real user into a labeller for free.

## 7. Evaluation methodology — `evals/`

Per note × provider:

| Metric | Definition |
| --- | --- |
| Field accuracy | Field-level precision/recall against hand-checked gold labels |
| Unsupported-claim rate | Claims without a matching evidence span in the transcript (lower bound on hallucination) |
| Latency | p50 / p95 wall-clock per note |
| Cost | Tokens and € per note, projected to € per 1,000 notes |

**Ablation:** extraction scored on clean scripted text vs ASR output, isolating speech errors from extraction errors. This quantifies exactly how much accuracy ASR costs and makes the case for (or against) a specialised German ASR swap.

**Output:** `evals/results/results.md` with one comparison table and a bar chart. The README leads with this table.

**Determinism:** temperature 0 and structured-output mode everywhere, so runs are comparable; the mini-eval in CI guards against silent regressions.

## 8. Build plan (7 evenings)

Get something end-to-end on day 1, then widen. Never spend a day on something that doesn't run.

| Day | Deliverable |
| --- | --- |
| 1 | `schema.py` + `extract.py` working on 3 hand-written text transcripts, one provider. End-to-end by bedtime. |
| 2 | `providers.py` interface, add Ollama + second API provider; `validate.py` repair→retry→fallback chain with logging. |
| 3 | `generate_notes.py` + TTS + `transcribe.py`; expand to 15–20 audio notes. Add `redact.py`. |
| 4 | Hand-correct gold labels; write `metrics.py` + `run_eval.py`; first full eval run. |
| 5 | `qm.py` quality pass; Streamlit demo (upload audio → validated note with flags). |
| 6 | Tests, CI workflow, `results.md` table + chart, README polish. |
| 7 | Buffer: fix what the eval surfaced, record demo, final pass. |

**Scope guard:** if anything slips, the cut order is **Streamlit → QM pass → third provider**. The eval harness and fallback logic are never cut.

## 9. Risks & known limitations

| Risk / limitation | Mitigation / honest position |
| --- | --- |
| Synthetic data may not transfer to real caregiver speech | Stated openly; the synthetic set is a development harness that makes iteration measurable. Production metric: caregiver edit-rate. |
| German prose quality unchecked | Translation tooling, stated openly. Evals score field structure and factual mapping, not prose. Have a German speaker sanity-check 5 notes if possible. |
| Evidence spans can be gamed (quote + misattribute) | Metric named honestly as *unsupported-claim rate*; misattribution still surfaces as field-level errors against gold. |
| Whisper vs German-specialised ASR | Pragmatic measured baseline; WER reported, clean-vs-ASR ablation quantifies the cost. ASR is one adapter — provider-agnosticism applies to the speech layer too. |
| Ollama on a laptop isn't production serving | Demo measures quality parity and cost, not throughput. Production self-hosting: vLLM on an L4/A10-class GPU with batching; back-of-envelope €/1,000 notes in the README. |
| Pseudonymisation ≠ GDPR compliance | Pseudonymised data is still personal data. Real deployment: DPAs, EU-resident zero-retention endpoints, or self-hosting. |
| Name collisions in the redaction step | Known weak point; documented in Limitations. |
| ASR breaks first on dialect and background noise | Exactly why the ablation exists; swapping ASR is designed to be cheap. |

## 10. Future work (production direction)

- **Streaming transcription** for latency.
- **Fine-tune a small self-hosted model** (e.g. Llama 3.2 3B) on distilled labels from the strongest API model, targeting quality parity at ~1/10 the cost.
- **Human-in-the-loop review queue** with edit-rate telemetry as the live quality metric.
- **Export adapters** — output JSON shaped so it could export into incumbent care software (Medifox Dan MD Stationär, Vivendi PD).

### Sources

- [Strukturmodell / SIS background — medi-karriere.de](https://medi-karriere.de/magazin/pflege/strukturierte-informationssammlung-sis)
- [Strukturmodell four elements — Wiley sample chapter](https://application.wiley-vch.de/books/sample/3527721452_c01.pdf)
