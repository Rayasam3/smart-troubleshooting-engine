import json

import pytest

from app.data_loader import catalog_uris, paired_inputs
from app.llm.client import LLMResult
from app.pipeline.extract import extract
from app.pipeline.extract_llm import LLMExtractor
from app.pipeline.extract_rules import RuleExtractor
from app.pipeline.validators import audit_response

PHASE2_CODES = {"ORDER_CRITICAL_NOT_LAST", "AUTO_NO_DEEPLINK"}


@pytest.mark.parametrize("query,siis", paired_inputs())
def test_every_input_valid_offline(query, siis):
    assert siis is not None
    res = extract(query, siis, RuleExtractor())
    report = audit_response({"contexts": [res.goal.model_dump(mode="json")] if res.goal else []},
                            set(catalog_uris()))
    assert report["schema_valid"]
    assert [i for i in report["issues"] if i["severity"] == "error" and i["code"] not in PHASE2_CODES] == []


class FakeClient:
    model = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def complete_json(self, system, messages):
        self.calls += 1
        return LLMResult(text=self.replies.pop(0), prompt_tokens=100, completion_tokens=50, cost_usd=0.0001)


SIIS = {"siis_response": {"title": "Screen does not rotate", "content":
        "## Adjust Screen Orientation\nSwipe down from the top of the screen to open the Quick settings panel. "
        "Tap the Auto rotate icon to enable it."}}

BAD = {"relevance": 0.9, "topic": "screen rotation", "title": "Fix the screen rotation problem now",
       "query_variations": ["screen wont rotate", "screen wont rotate", "see https://x.com"],
       "actions": [{"actionName": "adjust screen orientation",
                    "description": "Lets you turn on automatic screen rotation on every app you use",
                    "steps": ["Swipe down from the top of the screen to open the Quick settings panel.",
                              "Visit https://techcorp.com/rotate for more information.",
                              "Download the Rotation Fixer app from the store."]}]}


def test_fake_llm_is_repaired_and_cleaned():
    client = FakeClient(["{not valid json", json.dumps(BAD)])
    res = extract("my screen won't rotate", SIIS, LLMExtractor(client))
    assert client.calls == 2                                    # repair loop worked
    steps = [s for a in res.goal.actions for g in a.stepGroups for s in g.steps]
    assert not any("http" in s or "Rotation Fixer" in s for s in steps)   # leak + invented step removed
    assert res.plan.query_variations == ["screen wont rotate"]           # duplicate + URL removed
    assert res.goal.title == "Fix screen rotation"


def test_llm_failure_never_crashes():
    res = extract("x", SIIS, LLMExtractor(FakeClient(["bad", "bad", "bad"])))
    assert res.goal is None and res.fallback == "extraction_error"