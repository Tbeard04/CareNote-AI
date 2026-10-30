# Eval results

16 notes x 4 model columns x 2 conditions (clean script vs ASR transcript). Repair-retry enabled, cross-provider fallback disabled for fair per-model columns.

## Quality

| Provider | Model | Condition | Obs P | Obs R | Obs F1 | Themenfeld acc | Risk P | Risk R | Resident acc | Unsupported claims |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic | claude-haiku-5-5 | asr | 0.90 | 0.75 | **0.80** | 0.93 | 0.94 | 1.00 | 0.69 | 0.0% |
| anthropic | claude-haiku-5-5 | clean | 0.98 | 0.77 | **0.84** | 0.97 | 1.00 | 1.00 | 1.00 | 0.0% |
| ollama | llama3.1:8b | asr | 0.91 | 0.81 | **0.83** | 0.66 | 0.88 | 0.88 | 0.69 | 7.3% |
| ollama | llama3.1:8b | clean | 0.88 | 0.73 | **0.77** | 0.66 | 0.94 | 0.94 | 0.94 | 2.5% |
| ollama | qwen3:8b | asr | 0.94 | 0.61 | **0.70** | 0.66 | 0.94 | 1.00 | 0.69 | 0.0% |
| ollama | qwen3:8b | clean | 0.95 | 0.61 | **0.70** | 0.74 | 1.00 | 1.00 | 1.00 | 0.0% |
| openai | gpt-4o-mini | asr | 0.94 | 0.53 | **0.65** | 0.93 | 0.88 | 0.94 | 0.75 | 10.3% |
| openai | gpt-4o-mini | clean | 0.94 | 0.54 | **0.66** | 0.89 | 0.91 | 0.94 | 1.00 | 14.8% |

## Latency, reliability, cost

| Provider | Model | Condition | p50 latency | p95 latency | Retry rate | Human review | $/note | $/1k notes | €/1k notes |
|---|---|---|---|---|---|---|---|---|---|
| anthropic | claude-haiku-5-5 | asr | 5.2s | 6.9s | 0% | 0% | $0.00107 | $1.07 | €0.98 |
| anthropic | claude-haiku-5-5 | clean | 5.9s | 7.1s | 0% | 0% | $0.00108 | $1.08 | €0.99 |
| ollama | llama3.1:8b | asr | 5.1s | 6.4s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| ollama | llama3.1:8b | clean | 5.0s | 7.7s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| ollama | qwen3:8b | asr | 5.0s | 6.3s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| ollama | qwen3:8b | clean | 4.9s | 9.3s | 0% | 0% | $0.00000 | $0.00 | €0.00 |
| openai | gpt-4o-mini | asr | 3.0s | 4.2s | 0% | 0% | $0.00037 | $0.37 | €0.34 |
| openai | gpt-4o-mini | clean | 3.1s | 4.3s | 0% | 0% | $0.00037 | $0.37 | €0.34 |

## ASR quality (faster-whisper)

Mean WER across 16 notes: **7.2%** (worst: note_15 at 25.5%). Local models run at zero marginal cost; $/note covers API providers only.

![Observation F1 by provider](f1_by_provider.png)
