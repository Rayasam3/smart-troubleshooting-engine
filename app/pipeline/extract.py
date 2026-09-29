"""Phase 1 entry point: SIIS article -> cleaned, grounded RawPlan -> provisional schema.Goal.
Deeplinks are attached in Phase 2; here every stepGroup has actionableDeeplink = None.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import RELEVANCE_MIN, RELEVANCE_MIN_LEXICAL, VARIATIONS_MAX, LLMSettings
from app.pipeline.categorize import categorize
from app.pipeline.contracts import RawAction, RawPlan
from app.pipeline.extract_rules import RuleExtractor
from app.pipeline.grounding import Grounder
from app.pipeline.scrub import scrub_text
from app.pipeline.siis_parser import clean_article
from app.pipeline.validators import (build_goal, clamp_score, clean_steps, fix_action_name,
                                     fix_description, fix_title)
from app.schema import Action, Goal, StepGroup


@dataclass
class ExtractionResult:
    goal: Optional[Goal]
    plan: Optional[RawPlan]
    extractor: str
    fallback: Optional[str] = None           # "no_match" | "no_siis_context" | "extraction_error"
    dropped_steps: List[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "rules"


def get_extractor(settings: LLMSettings | None = None):
    settings = settings or LLMSettings()
    if settings.use_offline:
        return RuleExtractor()
    from app.llm.client import OpenAICompatClient
    from app.pipeline.extract_llm import LLMExtractor

    return LLMExtractor(OpenAICompatClient(settings), settings)


def postprocess(plan: RawPlan, source_text: str) -> tuple[RawPlan, List[dict]]:
    """Deterministically enforce every text rule and drop ungrounded steps."""
    grounder = Grounder(source_text)
    dropped: List[dict] = []
    merged: dict[str, RawAction] = {}
    for action in plan.actions:
        name = fix_action_name(action.actionName)
        steps, lost = grounder.filter(clean_steps(action.steps))
        dropped += [{**d, "action": name} for d in lost]
        if not steps:
            continue
        if name in merged:  # same screen twice -> merge (one action = one screen)
            merged[name].steps += [s for s in steps if s not in merged[name].steps]
        else:
            merged[name] = RawAction(actionName=name, description=fix_description(action.description, name),
                                     steps=steps)

    topic = plan.topic or plan.title
    variations, seen = [], set()
    for v in plan.query_variations:
        v = scrub_text(v).strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            variations.append(v)

    cleaned = plan.model_copy(update={
        "actions": list(merged.values()),
        "title": fix_title(plan.title, topic),
        "relevance": clamp_score(plan.relevance),
        "normalized_query": scrub_text(plan.normalized_query) if plan.normalized_query else None,
        "query_variations": variations[:VARIATIONS_MAX],
    })
    if not cleaned.actions:
        cleaned.no_match = True
    return cleaned, dropped


def plan_to_goal(plan: RawPlan, score: float) -> Goal:
    actions = [Action(actionName=a.actionName, description=a.description,
                      stepGroups=[StepGroup(steps=a.steps)],
                      category=categorize(a.actionName, a.steps)) for a in plan.actions]
    return Goal(goal=build_goal(plan.topic or plan.title, plan.goal_kind), title=plan.title,
                actions=actions, score=clamp_score(score))


def extract(query: str, siis: dict | None, extractor=None) -> ExtractionResult:
    extractor = extractor or get_extractor()
    name = getattr(extractor, "name", "unknown")
    article = (siis or {}).get("siis_response") or siis or {}
    content = clean_article(article.get("content", "")) if isinstance(article, dict) else ""
    title = article.get("title", "") if isinstance(article, dict) else ""
    if not content.strip():
        return ExtractionResult(goal=None, plan=None, extractor=name, fallback="no_siis_context")

    def _usage(res: ExtractionResult) -> ExtractionResult:
        u = getattr(extractor, "usage", None)
        if u is not None:
            res.prompt_tokens, res.completion_tokens = u.prompt_tokens, u.completion_tokens
            res.cost_usd, res.model = u.cost_usd, u.model or name
        return res

    try:
        raw = extractor.extract(query, title, content)
    except Exception as exc:  # never crash: degrade to a fallback
        res = ExtractionResult(goal=None, plan=None, extractor=name, fallback="extraction_error")
        res.dropped_steps = [{"error": str(exc)[:300]}]
        return _usage(res)

    plan, dropped = postprocess(raw, title + "\n" + content)
    min_relevance = RELEVANCE_MIN if name == "llm" else RELEVANCE_MIN_LEXICAL
    if plan.no_match or plan.relevance < min_relevance:
        return _usage(ExtractionResult(goal=None, plan=plan, extractor=name, fallback="no_match",
                                       dropped_steps=dropped))

    score = plan.relevance if name == "llm" else 0.5 + 0.45 * plan.relevance
    return _usage(ExtractionResult(goal=plan_to_goal(plan, score), plan=plan, extractor=name,
                                   dropped_steps=dropped))
