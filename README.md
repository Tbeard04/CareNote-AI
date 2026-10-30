# CareNote AI

**German caregiver voice notes → validated, structured care documentation** (Strukturmodell: Berichteblatt entries + Risikomatrix flags mapped to the six SIS Themenfelder), with an eval harness comparing four models on accuracy, hallucination rate, latency and cost per note.

## Results

16 synthetic voice notes × 4 models × 2 conditions (clean script vs ASR transcript), scored field-by-field against hand-checked gold labels. The eval ran four full times during prompt iteration, so the table reports the final run **plus each model's F1 range across all four** — on a 16-note set, single-run rankings move by ±0.05–0.08 and ranges are the honest unit. Full tables and charts in [evals/results/results.md](evals/results/results.md); explore interactively on the demo's Results page.

| Model | Obs F1 (ASR) | Obs F1 (clean) | F1 range, 4 runs | Themenfeld acc | Unsupported claims (range) | p50 latency | $/1k notes |
|---|---|---|---|---|---|---|---|
| **claude-haiku-5-5** | 0.80 | **0.84** | **0.78–0.88** | 0.93 | **0.0% (always)** | 5.2s | $1.07 |
| llama3.1:8b (local) | **0.83** | 0.77 | 0.73–0.83 | 0.66 | 7.3% (2.1–13.2%) | 5.1s | **$0.00** |
| qwen3:8b (local) | 0.70 | 0.70 | 0.68–0.79 | 0.66 | **0.0%** (0–2.6%) | 5.0s | **$0.00** |
| gpt-4o-mini | 0.65 | 0.66 | 0.65–0.78 | 0.93 | 10.3% (3.7–21.2%) | **3.0s** | $0.37 |

What the numbers say:

- **Claude Haiku is the consistent quality leader** — top or tied-top F1 range in every run, and in **512 scored runs it never once produced a claim without a verbatim transcript quote** — at roughly **€1 per 1,000 notes**.
- **The self-hosted track is viable, with a trade-off the ranges expose**: the two local models swap F1 ranking between runs (noise on 16 notes), but one difference is stable — qwen3 almost never fabricates a quote, llama3.1 does so in every run. For a legally audited document, that reliability axis matters as much as F1.
- **gpt-4o-mini's failure mode is instructive**: high precision but the lowest recall, and it *paraphrases* its "quotes" in every run — its JSON mode doesn't enforce our schema, and the unsupported-claim rate exposes exactly that.
- **The eval loop earned its keep during prompt iteration**: an eval-surfaced fix (an explicit "unremarkable status is not an observation" rule) lifted the worst cell from 0.00 to 1.00 for all four models — while an aggressive "extract every detail" rule was tried and *rejected* because the harness showed it cost the leading model ~0.1 F1 in over-extraction.
- **ASR costs little on clean synthetic speech** (mean WER 7.2%) but 25.5% on the dialect note — precisely where real-world voice documentation gets hard.
- **0 retries, 0 human-review flags in 512 runs** — the repair/fallback chain existed and never had to fire; its cost when it does fire is measured per-attempt in the logs.

## Architecture

```text
audio (.mp3)
  → transcribe      faster-whisper, local, word timestamps kept
  → redact          names/rooms → placeholders BEFORE any external API call
  → extract (LLM)   one shared German prompt, native JSON mode, schema enforced
  → validate        parse fail → repair prompt w/ exact error → retry → stronger
                    model fallback → needs_human_review; every attempt logged
  → unredact        placeholders → real names
  → QM check        rules (vitals plausibility, required fields) + one cheap
                    LLM call for contradictions; warnings block sign-off
  → CareNote        Berichteblatt entry: observations per SIS Themenfeld, each
                    with a verbatim evidence span; Risikomatrix flags; explicit
                    null for anything not stated
```

Evidence spans are the anti-hallucination mechanism: a claim whose quote does not occur in the transcript counts as unsupported — honestly named an *unsupported-claim rate*, a lower bound on hallucination (a model can quote correctly and still misattribute; that surfaces as field-level error against gold instead).

## Quickstart

```bash
uv sync
uv run pytest                      # 58 tests, no keys or models needed
uv run python -m carenote.pipeline data/audio/note_01.mp3   # needs Ollama
```

Runs with **zero paid keys**: the default provider is a local Llama via [Ollama](https://ollama.com) (`ollama pull llama3.1:8b`). Copy `.env.example` to `.env` to add Anthropic/OpenAI keys, pick models, or configure the fallback chain.

```bash
uv run --group demo streamlit run demo/app.py   # bilingual demo UI (DE/EN)
uv run python evals/run_eval.py                 # full eval, all providers
```

## Design decisions

- **Hand-rolled ~100-line provider interface** (Anthropic / OpenAI / Ollama) instead of LangChain or LiteLLM — the pipeline needs exactly one operation, and the interface *is* the provider-agnosticism demonstration. Swapping the ASR layer later is the same adapter pattern.
- **No eval framework** — the metrics are the point: greedy evidence-span alignment so a wrong Themenfeld counts as a mapping error rather than a missed observation, risk flags as set comparison, WER for the speech layer. Plain Python, readable in one sitting ([evals/metrics.py](evals/metrics.py)).
- **No agent framework** — a linear pipeline doesn't need one.
- **Fallback triggered by validation failure only**, never guessed confidence; every retry/fallback event is a JSON-lines log entry, so reliability has an auditable, measured price.
- **Draft + mandatory confirmation, never auto-signed** — a wrong entry has legal weight (Medizinischer Dienst audits). QM warnings block sign-off; a *failed* QM check is itself a blocking flag, so a note is never silently approved.
- **Pseudonymisation before any external call** (spaCy NER + title regex; single-token NER names are filtered because the small German model labels "Blutdruck" a person). Honest scope: pseudonymised data is still personal data under GDPR Art. 9 — real deployment needs DPAs, zero-retention EU endpoints, or self-hosting, which is the strongest commercial argument for the self-hosted track.
- **Eval fairness**: repair-retry (same model) on, cross-provider fallback off, so every column's quality and cost belong to one model.

## Limitations

- **Synthetic data.** TTS voices are cleaner than real caregivers in a hallway; the 7.2% WER flatters ASR. The set is a development harness that makes iteration measurable — in production the metric becomes **caregiver edit-rate**, which turns every user into a labeller.
- **Small gold set** (16 notes, single labeller). Repeated identical eval runs move per-model F1 by ±0.05–0.08, which is why the results table reports ranges across four runs; occasional clean-vs-ASR inversions are sampling variance, not magic.
- **German written with translation tooling** and reviewed by a non-native speaker. The evals score structure and factual mapping, not prose quality.
- **An 8B model as QM judge over-flags** (it called lunch-vs-dinner a contradiction despite instructions). It fails safe — blocking, not approving — but the demo defaults QM to Haiku for a reason.
- **Not the real Strukturmodell exchange format** — the schema borrows its vocabulary and structure, shaped so an export adapter to incumbent care software (Medifox, Vivendi) would be a mapping, not a redesign.

## In production I would…

- **Stream transcription** (whisper streaming or a specialised German ASR — the ASR layer is one adapter) to cut perceived latency to near-zero.
- **Fine-tune a small self-hosted model** (qwen3/Llama 3.2 class) on labels distilled from the strongest API model, targeting Haiku-parity extraction at ~1/10 the cost — the measured gap here says that's plausible.
- **Serve locally on vLLM** (L4/A10-class GPU, batching) — this demo measures quality parity and cost, not throughput.
- **Build the review queue around edit-rate telemetry**: which fields get corrected, by whom, on which note types — the live quality metric that gold sets can't be.
- **Harden redaction** against name collisions and spelling variants (the known weak point), and move all API traffic to EU-resident zero-retention endpoints or fully self-hosted serving.

## Repo map

```text
src/carenote/   schema, providers, extract, validate, redact, transcribe, qm, pipeline
evals/          metrics, run_eval, mini_eval (CI), results/ (tables, charts, predictions)
data/           generate_notes (16 scripted edge cases), audio/, scripted/, gold/
demo/           Streamlit app: bilingual guide, pipeline page, results dashboard
tests/          58 unit tests, no live LLM needed
```

All resident names are invented; all audio is synthetic.
