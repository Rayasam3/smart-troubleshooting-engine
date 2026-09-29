"""Programmatic enforcement of the output rules (PDF section 4).
  * fixers   (fix_*, to_*, clean_*) -> always return a rule-compliant value
  * checkers (audit_*)              -> report violations without changing anything
"""
import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

from pydantic import ValidationError

from app.config import DESC_MAX_WORDS, DESC_MIN_WORDS, GOAL_KINDS, TITLE_MAX_WORDS, TITLE_MIN_WORDS
from app.pipeline.scrub import contains_url, scrub_text
from app.schema import ContextDeeplinkResponse, Goal, actionCategory

SMALL_WORDS = {"a", "an", "the", "and", "or", "but", "nor", "for", "to", "of", "on", "in",
               "at", "by", "with", "via", "as", "from", "into", "vs"}
TRAILING_JUNK = SMALL_WORDS | {"your", "you", "this", "that", "its", "their", "my", "is", "are",
                               "be", "will", "can", "while", "when", "if", "so", "then"}
STEP_NOISE = re.compile(r"\b(learn more|for more information|more details|click here|"
                        r"website|web page|webpage|online at)\b", re.I)
_LEAD_FILLER = re.compile(r"^(?:please|first|firstly|next|then|now|finally|also|simply|"
                          r"alternatively|additionally|afterwards|lastly)\b[,:]?\s+", re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)]|step\s*\d+\s*[:.)-]?)\s*", re.I)
GOAL_RE = re.compile(r"^Follow these steps to perform this (?P<topic>.+) "
                     r"(?P<kind>Troubleshooting|Configuration)$")


# ------------------------------------------------------------------ helpers
def words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", (text or "").strip()) if w]


def _keeps_own_case(word: str) -> bool:
    """Acronyms and brand-style words (USB, QR, Wi-Fi, TechCorp) keep their casing."""
    core = re.sub(r"[^A-Za-z]", "", word)
    return len(core) > 1 and (core.isupper() or any(c.isupper() for c in core[1:]))


def _cap(word: str) -> str:
    return "-".join(p[:1].upper() + p[1:] for p in word.split("-"))


def to_title_case(text: str) -> str:
    tokens = words(re.sub(r"[\"'`]+", "", text))
    out = []
    for i, w in enumerate(tokens):
        if _keeps_own_case(w):
            out.append(w)
        elif 0 < i < len(tokens) - 1 and w.lower() in SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(_cap(w.lower()))
    return " ".join(out)


def to_sentence_case(text: str) -> str:
    tokens = words(text)
    out = []
    for i, w in enumerate(tokens):
        if _keeps_own_case(w):
            out.append(w)
        else:
            out.append(_cap(w.lower()) if i == 0 else w.lower())
    return " ".join(out)


def _strip_trailing_junk(tokens: List[str], keep_min: int = 1) -> List[str]:
    while len(tokens) > keep_min and tokens[-1].lower().strip(",.;:") in TRAILING_JUNK:
        tokens = tokens[:-1]
    return tokens


# ------------------------------------------------------------------ fixers
def clean_topic(topic: str) -> str:
    topic = scrub_text(topic or "")
    topic = re.sub(r"[^A-Za-z0-9\- ]+", " ", topic)
    topic = re.sub(r"\b(troubleshooting|configuration|guide|steps?)\b", "", topic, flags=re.I)
    tokens = _strip_trailing_junk(words(topic))[:4]
    return to_title_case(" ".join(tokens)) or "Device"


def build_goal(topic: str, kind: str = "Troubleshooting") -> str:
    kind = kind if kind in GOAL_KINDS else "Troubleshooting"
    return f"Follow these steps to perform this {clean_topic(topic)} {kind}"


def fix_title(title: str, fallback_topic: str = "") -> str:
    """2-3 words, sentence case."""
    tokens = words(re.sub(r"[^A-Za-z0-9\- ]+", " ", scrub_text(title or "")))
    if len(tokens) > TITLE_MAX_WORDS:
        content = [t for t in tokens if t.lower() not in TRAILING_JUNK]
        tokens = content[:TITLE_MAX_WORDS] if len(content) >= TITLE_MIN_WORDS else tokens[:TITLE_MAX_WORDS]
    tokens = _strip_trailing_junk(tokens)
    if len(tokens) < TITLE_MIN_WORDS:
        extra = [t for t in words(fallback_topic) if t.lower() not in {x.lower() for x in tokens}]
        tokens = (tokens + extra + ["issue", "fix"])[:TITLE_MAX_WORDS]
        tokens = tokens[:max(TITLE_MIN_WORDS, len(_strip_trailing_junk(tokens, TITLE_MIN_WORDS)))]
    return to_sentence_case(" ".join(tokens))


def fix_action_name(name: str) -> str:
    name = re.sub(r"^\s*(?:step\s*\d+\s*[:.)-]?|\d+[.)])\s*", "", scrub_text(name or ""), flags=re.I)
    name = re.sub(r"[.:;!?]+$", "", name).strip()
    return to_title_case(name) or "Follow Guided Steps"


def fix_description(desc: str, action_name: str = "") -> str:
    """Exactly 5-7 words starting with 'It will'."""
    body = scrub_text(desc or "")
    body = re.sub(r"[.!?;:]+$", "", body).strip()
    body = re.sub(r"^it\s+will\s+", "", body, flags=re.I)
    tokens = words(body)
    if not tokens:
        tokens = ["help", "you"] + words(action_name.lower())
    limit = DESC_MAX_WORDS - 2
    if len(tokens) > limit:
        tokens = _strip_trailing_junk(tokens[:limit])
    for pad in ("on", "your", "device"):
        if len(tokens) + 2 >= DESC_MIN_WORDS:
            break
        tokens.append(pad)
    tokens = [t if _keeps_own_case(t) else t.lower() for t in tokens]
    return "It will " + " ".join(tokens[:limit])


def split_step(step: str) -> List[str]:
    """One physical interaction per step: split multi-sentence strings."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", (step or "").strip())
    return [p for p in parts if p.strip()]


def clean_step(step: str) -> Optional[str]:
    s = scrub_text(step or "")
    s = _BULLET.sub("", s).strip()
    for _ in range(3):
        s = _LEAD_FILLER.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" -")
    if len(words(s)) < 2 or STEP_NOISE.search(s):
        return None
    s = s[0].upper() + s[1:]
    if not re.search(r"[.!?]$", s):
        s += "."
    return s


def clean_steps(steps: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for raw in steps:
        for part in split_step(raw):
            cleaned = clean_step(part)
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                out.append(cleaned)
    return out


def clamp_score(score) -> float:
    try:
        return round(min(1.0, max(0.0, float(score))), 2)
    except (TypeError, ValueError):
        return 0.5


# ------------------------------------------------------------------ checkers
@dataclass
class Issue:
    code: str
    severity: str  # "error" breaks a hard rule, "warning" is a quality signal
    path: str
    message: str

    def to_dict(self):
        return asdict(self)


def audit_goal(goal: Goal, catalog: Optional[set] = None, idx: int = 0) -> List[Issue]:
    issues: List[Issue] = []
    base = f"contexts[{idx}]"

    def add(code, sev, path, msg):
        issues.append(Issue(code, sev, path, msg))

    m = GOAL_RE.match(goal.goal or "")
    if not m:
        add("GOAL_FORMAT", "error", f"{base}.goal", f"Wrong goal syntax: {goal.goal!r}")
    n = len(words(goal.title))
    if not TITLE_MIN_WORDS <= n <= TITLE_MAX_WORDS:
        add("TITLE_WORDS", "error", f"{base}.title", f"Title has {n} words: {goal.title!r}")
    if goal.title and to_sentence_case(goal.title) != goal.title:
        add("TITLE_CASE", "error", f"{base}.title", f"Title not sentence case: {goal.title!r}")
    if not 0.0 <= goal.score <= 1.0:
        add("SCORE_RANGE", "error", f"{base}.score", f"Score {goal.score} outside 0-1")
    if not goal.actions:
        add("NO_ACTIONS", "error", f"{base}.actions", "Goal has no actions")

    seen_critical = False
    for a_i, action in enumerate(goal.actions):
        ap = f"{base}.actions[{a_i}]"
        if to_title_case(action.actionName) != action.actionName:
            add("ACTION_NAME_CASE", "error", f"{ap}.actionName", f"Not Title Case: {action.actionName!r}")
        d = action.description or ""
        if not d.startswith("It will"):
            add("DESC_PREFIX", "error", f"{ap}.description", f"Must start with 'It will': {d!r}")
        dn = len(words(d))
        if not DESC_MIN_WORDS <= dn <= DESC_MAX_WORDS:
            add("DESC_WORDS", "error", f"{ap}.description", f"Description has {dn} words: {d!r}")
        if action.category == actionCategory.critical:
            seen_critical = True
        elif seen_critical:
            add("ORDER_CRITICAL_NOT_LAST", "error", ap, "Non-critical action after a critical one")
        if not action.stepGroups:
            add("NO_STEPGROUPS", "error", ap, "Action has no stepGroups")

        texts = [("actionName", action.actionName), ("description", d)]
        for g_i, group in enumerate(action.stepGroups):
            gp = f"{ap}.stepGroups[{g_i}]"
            if not group.steps:
                add("EMPTY_STEPS", "error", gp, "Step group has no steps")
            texts += [(f"stepGroups[{g_i}].steps[{s_i}]", s) for s_i, s in enumerate(group.steps)]
            if group.actionableDeeplink:
                texts += [(f"stepGroups[{g_i}].actionableDeeplink.description", group.actionableDeeplink.description),
                          (f"stepGroups[{g_i}].actionableDeeplink.message", group.actionableDeeplink.message or "")]
            if action.category == actionCategory.manual and (group.actionableDeeplink or group.validationDeeplink):
                add("MANUAL_HAS_DEEPLINK", "error", gp, "Manual actions cannot carry deeplinks")
            if action.category == actionCategory.auto and not group.actionableDeeplink:
                add("AUTO_NO_DEEPLINK", "warning", gp, "Auto action without actionable deeplink")
            if catalog is not None:
                for kind, link in (("actionableDeeplink", group.actionableDeeplink),
                                   ("validationDeeplink", group.validationDeeplink)):
                    if link and link.deeplink not in catalog:
                        add("DEEPLINK_NOT_IN_CATALOG", "error", f"{gp}.{kind}", f"Unknown URI {link.deeplink!r}")
        for field_path, text in texts:
            if contains_url(text):
                add("URL_LEAK", "error", f"{ap}.{field_path}", f"URL/email in text: {text!r}")
    return issues


def audit_response(payload: dict, catalog: Optional[set] = None) -> dict:
    """Validate a raw `response` dict against schema.py AND the rulebook."""
    try:
        parsed = ContextDeeplinkResponse.model_validate(payload)
    except ValidationError as exc:
        return {"schema_valid": False, "schema_errors": [e["msg"] + f" @ {e['loc']}" for e in exc.errors()],
                "issues": []}
    issues = []
    for i, goal in enumerate(parsed.contexts):
        issues += audit_goal(goal, catalog, i)
    return {"schema_valid": True, "schema_errors": [], "issues": [x.to_dict() for x in issues]}