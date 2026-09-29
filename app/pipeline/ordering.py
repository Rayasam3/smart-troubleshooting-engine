"""Plan hierarchy (PDF 6.2): settings toggles -> physical checks -> disruptive operations.
Stable sort, so the article's own order is kept inside each group."""
import re

from app.schema import Goal, actionCategory

_GROUP = {actionCategory.auto: 0, actionCategory.manual: 1, actionCategory.critical: 2}
_MANUAL_ESCALATION = re.compile(r"service cent|customer support|contact|repair|replace", re.I)
_CRITICAL_SEVERITY = [
    (re.compile(r"\bre-?start|\breboot|power (it |the device )?off and on", re.I), 0),
    (re.compile(r"safe mode|recovery (mode|menu)|wipe (the )?cache", re.I), 1),
    (re.compile(r"update|firmware", re.I), 2),
    (re.compile(r"reset (all |network )?settings", re.I), 3),
    (re.compile(r"factory|erase all", re.I), 4),
]


def _sub_rank(action) -> int:
    text = action.actionName + " " + " ".join(s for g in action.stepGroups for s in g.steps)
    if action.category == actionCategory.manual:
        return 1 if _MANUAL_ESCALATION.search(action.actionName) else 0   # visit service centre last
    if action.category == actionCategory.critical:
        name_hits = [rank for pat, rank in _CRITICAL_SEVERITY if pat.search(action.actionName)]
        text_hits = [rank for pat, rank in _CRITICAL_SEVERITY if pat.search(text)]
        return max(name_hits or text_hits or [2])
    return 0


def order_actions(goal: Goal) -> Goal:
    goal.actions = sorted(goal.actions, key=lambda a: (_GROUP[a.category], _sub_rank(a)))
    return goal