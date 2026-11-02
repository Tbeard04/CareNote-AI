# CareNote AI

**German caregiver voice notes → validated, structured care documentation** (Strukturmodell: Berichteblatt entries + Risikomatrix flags mapped to the six SIS Themenfelder), with an eval harness comparing four models on accuracy, hallucination rate, latency and cost per note.

## Results

100 synthetic voice notes (short status lines to long rambling handovers) × 6 models × 2 conditions (clean script vs ASR transcript), scored field-by-field against hand-checked gold labels — 1,200 scored runs. Full tables and charts in [evals/results/results.md](evals/results/results.md); explore interactively on the demo's Results page.

| Model | Obs F1 (ASR) | Obs F1 (clean) | Themenfeld acc | Unsupported claims | Resident acc (clean) | p50 latency | $/1k notes |
|---|---|---|---|---|---|---|---|
| **claude-haiku-5-5** | 0.70 | 0.73 | **0.88** | **1.0%** | 0.91 | 5.4s | $1.04 |
| gemma3:12b (local) | **0.74** | **0.76** | 0.71 | 5.0% | 0.98 | 8.4s | **$0.00** |
| mistral-nemo:12b (local) | 0.72 | 0.73 | 0.64 | 13.4% | 0.36 | 5.3s | **$0.00** |
| qwen3:8b (local) | 0.71 | 0.73 | 0.54 | 2.4% | 0.98 | **4.9s** | **$0.00** |
| llama3.1:8b (local) | 0.69 | 0.69 | 0.52 | 6.0% | 0.98 | 5.2s | **$0.00** |
| gpt-4o-mini | 0.61 | 0.64 | 0.78 | 20.9% | 0.98 | **2.8s** | $0.37 |

What the numbers say:

- **The harder 100-note set compressed everyone and split "quality" into two axes.** On raw F1, the free local **gemma3:12b edges out Claude Haiku** (0.74 vs 0.70 on ASR). But Haiku stays the **trust leader**: near-zero unsupported claims (1% vs gemma's 5%) and the best Themenfeld mapping (0.88) — for a legally audited document, those axes matter as much as F1, and they cost about **€1 per 1,000 notes**.
- **Local model choice is now an informed trade-off, not a guess**: gemma3:12b for recall (0.90 — it misses the least), qwen3:8b for discipline (2% unsupported, highest precision among locals), and mistral-nemo despite its German pedigree disqualifies itself — it mangles resident references (0.36–0.40 accuracy) and fabricates quotes at 13%.
- **gpt-4o-mini is the cautionary tale**: last on F1, 21% paraphrased "quotes", and the **only model whose output ever failed schema validation** — all 5 repair-retries and both needs-human-review flags in 1,200 runs are its (the chain's cost when it fires is in the logs, measured).
- **The eval loop**: an eval-surfaced fix ("unremarkable status is not an observation") lifted the worst cell from 0.00 to 1.00 for every model — while an aggressive "extract every detail" rule was tried and *rejected* because the harness showed it cost the leading model ~0.1 F1 in over-extraction.
- **ASR costs little on clean synthetic speech** (mean WER 8.8%) but spikes on dialect and very short utterances (worst note: 54.5%) — precisely where real-world voice documentation gets hard.

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
- **Single labeller gold set.** 100 notes narrow the run-to-run noise that a 16-note pilot showed (±0.05–0.08 F1 there), but every label still reflects one person's judgment calls about what counts as an observation; differences of a couple of F1 points between adjacent models remain within arguing distance.
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
data/           note_scripts.json (100 scripted edge cases), audio/, scripted/, gold/
demo/           Streamlit app: bilingual guide, pipeline page, results dashboard
tests/          58 unit tests, no live LLM needed
```

All resident names are invented; all audio is synthetic.
