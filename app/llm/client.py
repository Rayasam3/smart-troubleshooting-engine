"""Thin wrapper around any OpenAI-compatible chat API, with token + cost accounting."""
import time
from dataclasses import dataclass
from typing import List, Protocol

from app.config import LLMSettings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""


class LLMClient(Protocol):
    model: str

    def complete_json(self, system: str, messages: List[dict]) -> LLMResult: ...


class OpenAICompatClient:
    def __init__(self, settings: LLMSettings | None = None):
        from openai import OpenAI

        self.s = settings or LLMSettings()
        self.model = self.s.model
        self._client = OpenAI(api_key=self.s.api_key, base_url=self.s.base_url,timeout=self.s.timeout_s, max_retries=5)

    def complete_json(self, system: str, messages: List[dict]) -> LLMResult:
        kwargs = dict(model=self.s.model, temperature=self.s.temperature,
                      messages=[{"role": "system", "content": system}] + messages,
                      response_format={"type": "json_object"})
        if self.s.seed is not None:
            kwargs["seed"] = self.s.seed
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # some providers reject `seed`; retry once without it
            if "seed" in str(exc).lower() and "seed" in kwargs:
                kwargs.pop("seed")
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        usage = resp.usage
        p, c = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        cost = (p * self.s.price_in_per_m + c * self.s.price_out_per_m) / 1_000_000
        return LLMResult(text=resp.choices[0].message.content or "", prompt_tokens=p, completion_tokens=c,
                         cost_usd=round(cost, 6), latency_ms=round((time.perf_counter() - start) * 1000, 1),
                         model=self.s.model)