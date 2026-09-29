"""Attach the exact Settings-screen deeplink to every action (PDF 'Screen Resolution Accuracy').

Pipeline per action:
  1. extract the screen/option names the steps tell the user to reach ("Tap Navigation bar" -> "Navigation bar")
  2. pull candidates from the hybrid index
  3. accept a candidate ONLY if its setting name matches one of those names (defeats parent-menu matches)
  4. pick on/off/view variant from the step's verb ("turn off" -> offURL)
  5. no match but the action clearly opens Settings -> voiceassist://dummy_positive; otherwise no deeplink
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from app.config import DUMMY_DEEPLINK, MAP_MIN_NAME_MATCH
from app.pipeline.categorize import categorize
from app.pipeline.deeplink_index import Candidate, CatalogEntry, DeeplinkIndex
from app.pipeline.validators import words
from app.schema import Action, Deeplink, Goal, ValidationDeepLink, actionCategory

# "Tap on Navigation bar." / "toggle off Super steady mode" / "select Factory data reset"
_TARGET = re.compile(
    r"\b(?:tap|touch|select|choose|open|toggle|turn|switch|enable|disable|find|locate|go to|navigate to|"
    r"search for(?: and select)?|adjust|set|use)\s+(?:on\s+|off\s+|the\s+|to\s+)*"
    r"(?P<t>[A-Za-z0-9][\w\-/&' ]{1,60}?)"
    r"(?=\s+(?:to|and|then|again|icon|switch|option|options|menu|toggle|button|tab|setting|settings|screen|"
    r"panel|from|in|on|at|for|until|if|by|so|under|below|once|its|it|when|where|which|that|while|with|"
    r"into|after|before)\b|[.,;:()]|$)", re.I)
_QUOTED = re.compile(r"[\"“']([^\"”']{3,40})[\"”']")
_CHAIN = re.compile(r"settings\s*>\s*([^.]+)", re.I)
_GENERIC = {"settings", "setting", "it", "them", "this", "that", "the screen", "screen", "device", "phone",
            "tablet", "app", "apps", "restart", "power", "back", "home", "ok", "done", "search field",
            "the search field", "top", "the top", "down", "up", "home screen", "apps screen",
            # buttons and dialog answers are not screens
            "allow", "yes", "no", "cancel", "start", "start now", "confirm", "next", "continue", "accept",
            "agree", "save", "apply", "delete", "delete all", "reset", "more", "more options", "add", "edit",
            "mirror", "remove", "uninstall", "install", "update", "search"}
_OFF = re.compile(r"\b(turn(?:ing)? off|disable|switch off|toggle off|deactivate)\b", re.I)
_ON = re.compile(r"\b(turn(?:ing)? on|enable|switch on|toggle on|activate)\b", re.I)
_SETTINGS_SCREEN = re.compile(r"\bsettings\b|quick settings|\bmenu\b", re.I)
# critical actions only get a placeholder when they really navigate the Settings app
_OPENS_SETTINGS_APP = re.compile(r"\b(open|go to|navigate to|launch)\s+(and open\s+)?(the\s+)?Settings\b|"
                                 r"Settings\s*>", re.I)
_TRAILING = {"and", "or", "then", "to", "the", "a", "an", "it", "its", "once", "now", "next", "again"}
_NAME_STOP = {"the", "a", "an", "and", "or", "of", "to", "for", "on", "in", "with", "your", "my",
              "menu", "option", "options", "page"}


def _trim(t: str) -> str:
    toks = words(re.sub(r"^(?:the|your|a|an)\s+", "", t.strip(" .,'\""), flags=re.I))
    while toks and toks[-1].lower().strip(",") in _TRAILING:
        toks.pop()
    return " ".join(toks)


def extract_targets(steps: List[str], strict: bool = True) -> List[str]:
    """Screen/option names named by the steps, deepest (last) first.
    UI names in the articles are capitalised ("Auto rotate", "Factory data reset"), so a verb-pattern
    match that starts lowercase ("between Portrait", "it back") is treated as prose, not a screen."""
    found: List[str] = []
    for step in steps:
        for chain in _CHAIN.findall(step):
            found += [p for p in re.split(r"\s*>\s*|,", chain) if p.strip()]
        found += _QUOTED.findall(step)
        found += re.findall(r"Quick settings panel", step, re.I)
        found += [m.group("t") for m in _TARGET.finditer(step) if not strict or m.group("t")[:1].isupper()]
        found += re.findall(r"\bnext to (?:the )?([A-Za-z0-9][\w\-/&' ]{1,60}?)(?=\s+(?:on|off|switch|toggle)\b|[.,;:]|$)",step, re.I)
    clean: List[str] = []
    for t in map(_trim, found):
        if any(len(w) > 18 for w in words(t)):          # garbled source text ("enteryourcurrentpin")
            continue
        if len(t) > 2 and t.lower() not in _GENERIC and t.lower() not in {c.lower() for c in clean}:
            clean.append(t)
    return list(reversed(clean))


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _NAME_STOP]


def _dummy_target(targets: List[str], action_name: str) -> str:
    """The screen a placeholder should name: the target sharing most words with the action name."""
    generic = {"settings", "setting", "screen", "device", "phone", "app", "apps"}
    name_words = set(_content_words(action_name)) - generic
    scored = [(len((set(_content_words(t)) - generic) & name_words), -i, t) for i, t in enumerate(targets)]
    if scored and max(scored)[0] > 0:
        return max(scored)[2]
    return targets[0] if targets else re.sub(r"^(\w+)\s+", "", action_name)


def _covered(src: List[str], dst: List[str]) -> bool:
    """Every word of src appears in dst IN THE SAME ORDER (tolerating plurals / small typos).
    Order matters: "Screen lock" is not "Lock screen", "Device Connect" is not "Connected devices"."""
    pos = 0
    for w in src:
        while pos < len(dst) and not (w == dst[pos] or fuzz.ratio(w, dst[pos]) >= 85):
            pos += 1
        if pos == len(dst):
            return False
        pos += 1
    return True


def name_match(target: str, name: str) -> float:
    """0-1 similarity between a screen named in a step and a catalog setting name.
    Word coverage is required first, so look-alikes are rejected even when their letters are similar:
      "Super steady mode" vs "Easy mode"            -> 0
      "Factory data reset" vs "Auto factory reset"  -> 0
      "Navigation bar" vs "Hide status and navigation bars" -> 0
      "Display" vs "Privacy display"                -> 0
      "Screen lock and biometrics" vs "Lock screen" -> 0
    """
    a_w, b_w = _content_words(target), _content_words(name)
    if not a_w or not b_w:
        return 0.0
    if len(b_w) == 1:  # one-word setting names ("Notifications", "More options") must match exactly
        raw = lambda x: " ".join(re.findall(r"[a-z0-9]+", x.lower()))
        return 1.0 if raw(target) == raw(name) else 0.0
    name_in_target = _covered(b_w, a_w) and (len(a_w) - len(b_w) <= 1 or len(b_w) / len(a_w) >= 0.5)
    target_in_name = len(a_w) >= 2 and _covered(a_w, b_w) and len(b_w) - len(a_w) <= 1
    if not (name_in_target or target_in_name):
        return 0.0
    a, b = " ".join(a_w), " ".join(b_w)
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) / 100.0


def step_direction(text: str) -> Optional[str]:
    off, on = bool(_OFF.search(text)), bool(_ON.search(text))
    if off and not on:
        return "off"
    if on and not off:
        return "on"
    return None


@dataclass
class MappingDecision:
    action: str
    targets: List[str]
    chosen_id: Optional[str]
    chosen_name: Optional[str]
    kind: str                       # "catalog" | "dummy" | "none"
    name_score: float = 0.0
    alternatives: List[Tuple[str, str, float]] = field(default_factory=list)


def make_dummy(target: str) -> Deeplink:
    """Placeholder for a real Settings screen missing from the catalog: we write description + message."""
    t = words(target)[:3]
    while len(t) > 1 and t[-1].lower() in _TRAILING | {"or", "and", "with", "for", "in", "on"}:
        t.pop()
    return Deeplink(deeplink=DUMMY_DEEPLINK,
                    description=" ".join(["Open", *t, "screen", "in", "Settings"]),   # 5-7 words
                    message=" ".join(["Go", "to", *t, "in", "Settings"]),             # 5-7 words
                    originalType="placeholder")


def to_links(entry: CatalogEntry) -> Tuple[Deeplink, Optional[ValidationDeepLink]]:
    link = Deeplink(deeplink=entry.deeplink, description=entry.description, message=entry.message,
                    originalType=entry.original_type)
    if not entry.validation_link or not entry.key:
        return link, None
    if entry.original_type in ("onURL", "offURL"):
        val = ValidationDeepLink(deeplink=entry.validation_link, key=entry.key, resultType="boolean",
                                 condition="equal", value="True" if entry.original_type == "onURL" else "False")
    else:
        val = ValidationDeepLink(deeplink=entry.validation_link, key=entry.key)
    return link, val


class DeeplinkMapper:
    def __init__(self, index: Optional[DeeplinkIndex] = None):
        self.index = index or DeeplinkIndex()

    def best_entry(self, action: Action) -> Tuple[Optional[CatalogEntry], float, List[str], List[Candidate]]:
        steps = [s for g in action.stepGroups for s in g.steps]
        targets = extract_targets(steps, strict=False)
        query = " ".join([action.actionName, *targets, *steps])
        candidates = self.index.search(query)
        direction = step_direction(" ".join(steps) + " " + action.actionName)

        best, best_score = None, 0.0
        for rank, cand in enumerate(candidates):
            e = cand.entry
            per_target = [max((name_match(t, n) for n in e.names), default=0.0) for t in targets]
            ns = max(per_target, default=0.0)
            if ns < MAP_MIN_NAME_MATCH:
                continue                      # its name matches no screen in the steps -> never accept
            # targets are deepest-first: matching the deepest screen beats matching a parent menu
            first_hit = next(i for i, s in enumerate(per_target) if s >= MAP_MIN_NAME_MATCH)
            depth_bonus = 0.02 * (1 - first_hit / len(targets))
            dir_adj = 0.0
            if e.original_type in ("onURL", "offURL"):
                wanted = {"on": "onURL", "off": "offURL"}.get(direction)
                if wanted:
                    dir_adj = 0.05 if e.original_type == wanted else -0.2
                elif e.original_type == "offURL":
                    dir_adj = -0.03       # no verb: opening/enabling is the safer default than disabling
            total = ns + depth_bonus + dir_adj + 0.01 * cand.dense - 0.0005 * rank
            if total > best_score:
                best, best_score = e, total
        final_name = max((name_match(t, n) for t in targets for n in best.names), default=0.0) if best else 0.0
        return best, final_name, targets, candidates

    def map_action(self, action: Action) -> Tuple[Action, MappingDecision]:
        steps = [s for g in action.stepGroups for s in g.steps]
        entry, score, targets, cands = self.best_entry(action)
        alts = [(c.entry.id, c.entry.key or c.entry.message, round(c.dense, 3)) for c in cands[:3]]
        group = action.stepGroups[0]

        if action.category == actionCategory.manual:
            group.actionableDeeplink = group.validationDeeplink = None
            return action, MappingDecision(action.actionName, targets, None, None, "none", 0.0, alts)

        if entry:
            group.actionableDeeplink, group.validationDeeplink = to_links(entry)
            if action.category != actionCategory.critical:
                action.category = actionCategory.auto
            return action, MappingDecision(action.actionName, targets, entry.id, entry.key or entry.message,
                                           "catalog", round(score, 3), alts)

        text = " ".join(steps)
        wants_dummy = ((action.category == actionCategory.auto and _SETTINGS_SCREEN.search(text)) or
                       (action.category == actionCategory.critical and _OPENS_SETTINGS_APP.search(text)))
        if wants_dummy:
            group.actionableDeeplink = make_dummy(_dummy_target(extract_targets(steps), action.actionName))
            group.validationDeeplink = None
            return action, MappingDecision(action.actionName, targets, "DL-DUMMY", None, "dummy", 0.0, alts)

        group.actionableDeeplink = group.validationDeeplink = None
        if action.category == actionCategory.auto:          # no screen to open -> it is a manual task
            action.category = categorize(action.actionName, steps, has_deeplink=False)
            if action.category == actionCategory.auto:
                action.category = actionCategory.manual
        return action, MappingDecision(action.actionName, targets, None, None, "none", 0.0, alts)

    def map_goal(self, goal: Goal) -> Tuple[Goal, List[MappingDecision]]:
        decisions = []
        for i, action in enumerate(goal.actions):
            goal.actions[i], d = self.map_action(action)
            decisions.append(d)
        return goal, decisions