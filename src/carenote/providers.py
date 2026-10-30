"""A deliberately small, hand-rolled provider interface.

One ``LLMProvider`` contract, three implementations: Anthropic, OpenAI and a
self-hosted Llama via Ollama. No LangChain, no LiteLLM — the pipeline needs
exactly one operation (a system+user completion that should return JSON), so
the interface is ~80 lines and every provider quirk is visible and testable.

Each provider uses its native structured-output / JSON mode:
- Anthropic: ``output_config.format`` with the JSON schema (guaranteed-valid
  JSON). Sampling parameters were removed from the current Claude API, so no
  temperature is sent.
- OpenAI: ``response_format={"type": "json_object"}`` at temperature 0 (strict
  schema mode rejects optional fields, so the schema travels in the prompt).
- Ollama: ``format=<schema>`` at temperature 0.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

# USD per 1M tokens (input, output). Ollama is local compute: 0 marginal cost.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass(frozen=True)
class LLMResult:
    """One completion plus the numbers the eval harness needs."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def cost_usd(self) -> float:
        prices = PRICING_USD_PER_MTOK.get(self.model)
        if prices is None:
            return 0.0
        price_in, price_out = prices
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class LLMProvider(ABC):
    """The one contract every backend fulfils."""

    name: str
    model: str

    def complete(self, system: str, user: str, json_schema: dict | None = None) -> LLMResult:
        """Run one completion, timed. ``json_schema`` requests native JSON output."""
        start = time.perf_counter()
        text, input_tokens, output_tokens = self._complete(system, user, json_schema)
        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=time.perf_counter() - start,
        )

    @abstractmethod
    def _complete(
        self, system: str, user: str, json_schema: dict | None
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self._client = anthropic.Anthropic()

    def _complete(
        self, system: str, user: str, json_schema: dict | None
    ) -> tuple[str, int, int]:
        kwargs: dict = {}
        if json_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text, response.usage.input_tokens, response.usage.output_tokens


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        import openai

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = openai.OpenAI()

    def _complete(
        self, system: str, user: str, json_schema: dict | None
    ) -> tuple[str, int, int]:
        kwargs: dict = {}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip(
            "/"
        )

    def _complete(
        self, system: str, user: str, json_schema: dict | None
    ) -> tuple[str, int, int]:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Thinking models (qwen3 etc.) reason for minutes on CPU before
            # emitting JSON; extraction needs the answer, not the monologue.
            # Non-thinking models ignore the flag.
            "think": False,
            "options": {"temperature": 0},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        timeout = float(os.getenv("OLLAMA_TIMEOUT_S", "600"))
        response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return (
            data["message"]["content"],
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Build a provider by name; default comes from CARENOTE_PROVIDER (ollama)."""
    load_dotenv()
    resolved = (name or os.getenv("CARENOTE_PROVIDER", "ollama")).lower()
    if resolved not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider '{resolved}' (known: {known})")
    return _PROVIDERS[resolved](model=model)
