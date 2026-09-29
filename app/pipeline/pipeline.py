"""End-to-end engine: complaint + SIIS article -> final, validated troubleshooting plan."""
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from app.pipeline.deeplink_mapper import DeeplinkMapper, MappingDecision
from app.pipeline.extract import ExtractionResult, extract, get_extractor
from app.pipeline.ordering import order_actions
from app.schema import Goal


@dataclass
class PipelineResult:
    goal: Optional[Goal]
    extraction: ExtractionResult
    decisions: List[MappingDecision] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    @property
    def response(self) -> dict:
        return {"contexts": [self.goal.model_dump(mode="json")] if self.goal else []}

    def decisions_as_dicts(self) -> List[dict]:
        return [asdict(d) for d in self.decisions]


def finalize(goal: Goal, mapper: DeeplinkMapper):
    """Phase 2 on an extracted goal: attach deeplinks, fix categories, order actions."""
    goal, decisions = mapper.map_goal(goal)
    by_action = {id(a): d for a, d in zip(goal.actions, decisions)}
    goal = order_actions(goal)
    return goal, [by_action[id(a)] for a in goal.actions]   # keep decisions aligned with the new order


class TroubleshootPipeline:
    def __init__(self, extractor=None, mapper: Optional[DeeplinkMapper] = None):
        self.extractor = extractor or get_extractor()
        self.mapper = mapper or DeeplinkMapper()

    def run(self, query: str, siis: Optional[dict]) -> PipelineResult:
        t0 = time.perf_counter()
        ext = extract(query, siis, self.extractor)
        t1 = time.perf_counter()
        result = PipelineResult(goal=None, extraction=ext)
        if ext.goal is not None:
            result.goal, result.decisions = finalize(ext.goal, self.mapper)
        t2 = time.perf_counter()
        result.timings_ms = {"extract": round((t1 - t0) * 1000, 1), "map_order": round((t2 - t1) * 1000, 1)}
        return result