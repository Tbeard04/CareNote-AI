# Eval results

16 notes x 4 model columns x 2 conditions (clean script vs ASR transcript). Repair-retry enabled, cross-provider fallback disabled for fair per-model columns.

## Quality

| Provider | Model | Condition | Obs P | Obs R | Obs F1 | Themenfeld acc | Risk P | Risk R | Resident acc | Unsupported claims |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic | claude-haiku-5-5 | asr | 0.92 | 0.89 | **0.89** | 0.88 | 1.00 | 1.00 | 0.75 | 0.0% |
| anthropic | claude-haiku-5-5 | clean | 0.82 | 0.80 | **0.80** | 0.94 | 1.00 | 1.00 | 0.88 | 0.0% |
| ollama | llama3.1:8b | asr | 0.74 | 0.85 | **0.74** | 0.53 | 0.94 | 0.94 | 0.69 | 3.9% |
| ollama | llama3.1:8b | clean | 0.78 | 0.80 | **0.73** | 0.63 | 0.88 | 0.88 | 0.81 | 2.2% |
| ollama | qwen3:8b | asr | 0.90 | 0.72 | **0.76** | 0.62 | 0.88 | 0.88 | 0.69 | 2.6% |
| ollama | qwen3:8b | clean | 0.88 | 0.77 | **0.79** | 0.69 | 0.88 | 0.88 | 1.00 | 0.0% |
| openai | gpt-4o-mini | asr | 0.96 | 0.64 | **0.72** | 0.93 | 0.84 | 0.88 | 0.75 | 9.4% |
| openai | gpt-4o-mini | clean | 0.97 | 0.62 | **0.72** | 0.93 | 0.94 | 0.94 | 1.00 | 13.3% |

## Latency, reliability, cost

| Provider | Model | Condition | p50 latency | p95 latency | Retry rate | Human review | $/note | $/1k notes | €/1k notes |
|---|---|---|---|---|---|---|---|---|---|
| anthropic | claude-haiku-5-5 | asr | 5.9s | 7.7s | 0% | 0% | $0.00106 | $1.06 | €0.97 |
| anthropic | claude-haiku-5-5 | clean | 5.3s | 7.0s | 0% | 0% | $0.00103 | $1.03 | €0.95 |
| ollama | llama3.1:8b | asr | 5.6s | 6.6s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| ollama | llama3.1:8b | clean | 5.2s | 6.5s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| ollama | qwen3:8b | asr | 5.4s | 6.5s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| ollama | qwen3:8b | clean | 5.7s | 8.6s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| openai | gpt-4o-mini | asr | 3.1s | 4.6s | 0% | 0% | $0.00037 | $0.37 | €0.34 |
| openai | gpt-4o-mini | clean | 3.0s | 3.9s | 0% | 0% | $0.00037 | $0.37 | €0.34 |

## ASR quality (faster-whisper)

Mean WER across 16 notes: **7.2%** (worst: note_15 at 25.5%). Local models run at zero marginal cost; $/note covers API providers only.

![Observation F1 by provider](f1_by_provider.png)
