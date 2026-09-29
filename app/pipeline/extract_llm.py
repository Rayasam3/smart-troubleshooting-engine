"""LLM-based extractor with a repair loop: if the JSON is invalid, the exact error is sent back."""
import json
import re

from pydantic import ValidationError

from app.config import LLMSettings
from app.llm.client import LLMClient, LLMResult
from app.llm.prompts import EXTRACTION_SYSTEM, build_user_message, few_shot_messages
from app.pipeline.contracts import RawPlan


class ExtractionFailed(RuntimeError):
    pass


def _parse(text: str) -> RawPlan:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return RawPlan.model_validate(json.loads(text))


class LLMExtractor:
    name = "llm"

    def __init__(self, client: LLMClient, settings: LLMSettings | None = None):
        self.client = client
        self.settings = settings or LLMSettings()
        self.usage = LLMResult(text="", model=getattr(client, "model", ""))

    def _track(self, r: LLMResult):
        self.usage.prompt_tokens += r.prompt_tokens
        self.usage.completion_tokens += r.completion_tokens
        self.usage.cost_usd = round(self.usage.cost_usd + r.cost_usd, 6)
        self.usage.latency_ms += r.latency_ms

    def extract(self, query: str, title: str, content: str) -> RawPlan:
        self.usage = LLMResult(text="", model=getattr(self.client, "model", ""))
        messages = few_shot_messages() + [{"role": "user", "content": build_user_message(query, title, content)}]
        last_error = ""
        for _ in range(self.settings.max_repair_attempts + 1):
            result = self.client.complete_json(EXTRACTION_SYSTEM, messages)
            self._track(result)
            try:
                return _parse(result.text)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:800]
                messages = messages + [
                    {"role": "assistant", "content": result.text},
                    {"role": "user", "content": f"Your output was invalid: {last_error}\n"
                                                "Return ONLY the corrected JSON object."},
                ]
        raise ExtractionFailed(f"LLM output invalid after repairs: {last_error}")