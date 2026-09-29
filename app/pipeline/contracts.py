"""Intermediate data passed between pipeline stages.

Extraction (Phase 1) produces a RawPlan: text only, no deeplinks, no categories.
Deeplink mapping (Phase 2) turns a RawPlan into the final schema.Goal.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RawAction(BaseModel):
    actionName: str
    description: str
    steps: List[str]


class RawPlan(BaseModel):
    no_match: bool = False
    relevance: float = Field(default=0.8, ge=0.0, le=1.0)
    topic: str = ""
    goal_kind: Literal["Troubleshooting", "Configuration"] = "Troubleshooting"
    title: str = ""
    normalized_query: Optional[str] = None
    query_variations: List[str] = []
    actions: List[RawAction] = []