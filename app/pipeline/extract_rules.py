"""Offline, deterministic extractor (no LLM). Also the 'rules' baseline for the ablation study."""
import re
from typing import List, Optional

from app.pipeline.contracts import RawAction, RawPlan
from app.pipeline.grounding import content_tokens
from app.pipeline.siis_parser import split_sections, split_sentences
from app.pipeline.validators import words

IMPERATIVE_VERBS = set("""go tap touch open navigate select swipe press hold turn enable disable toggle
switch check verify ensure make remove insert connect disconnect plug unplug charge restart reboot
update install uninstall clear delete back contact visit schedule send try use adjust set change
increase decrease reduce locate find search enter type scan place inspect examine shine keep wait
confirm reset perform launch close drag sign log clean wipe replace reinsert reconnect test avoid
choose follow return access view review move run download activate deactivate rotate slide force
attempt configure manage transfer create customize exit mirror cast pair""".split())
_YOU_CAN = re.compile(r"^(?:you\s+(?:can|may|should|could|might|must|need to|will need to|may need to)"
                      r"(?:\s+also)?|let'?s(?:\s+try\s+to)?|we recommend that you|it is recommended to)\s+", re.I)
_FILLER = re.compile(r"^(?:please|first|next|then|now|finally|also|simply|alternatively|additionally)"
                     r"\b[,:]?\s+", re.I)
_LEADING_CLAUSE = re.compile(r"^(?:if|when|after|once|for|to|on|using|while|before|in)\b[^,]{0,90},\s*", re.I)
_TROUBLE = re.compile(r"\b(not|error|issue|problem|fail|flicker|black|blank|crack|damage|respond|"
                      r"broken|slow|bleeding|won't|doesn't)\w*", re.I)
# device names and filler that say nothing about WHICH problem the user has
QUERY_NOISE = set("""techcorp nexa fold ultra x1 a14 a15 a15g galaxy phone smartphone tablet device new
completely totally suddenly whenever always again still just really even never ever goes go went going
time short long work works working use using try tried happen happens thing things anything nothing
something other normal normally""".split())
_GENERIC_HEADINGS = re.compile(r"^(understanding|about|overview|introduction|note|what is|why)\b", re.I)
_META = re.compile(r"\b(these steps|following steps|troubleshooting steps|some steps|resolve this|see if we|"
                   r"go through|here's how|here is how|as follows)\b", re.I)


def _strip_front(s: str) -> str:
    for _ in range(3):
        s = _FILLER.sub("", s)
        s = _YOU_CAN.sub("", s)
    return s.strip()


def to_imperative(sentence: str) -> Optional[str]:
    s = _strip_front(sentence.strip())
    if ":" in s[:80]:  # "On devices with a Power button: Press and hold..."
        prefix, _, rest = s.partition(":")
        main = to_imperative(rest.strip()) if rest.strip() else None
        if main:
            if _META.search(prefix) or len(words(prefix)) > 10 or re.match(r"there (are|is)\b", prefix, re.I):
                return main
            return f"{prefix.strip()}, {main[0].lower() + main[1:]}"
    if _META.search(s):
        return None
    first = (words(s)[:1] or [""])[0].lower().strip(",.:")
    if first in IMPERATIVE_VERBS:
        return s
    m = _LEADING_CLAUSE.match(s)  # "If X, contact Y" / "Using two fingers, swipe down"
    if m:
        main = _strip_front(s[m.end():])
        head = (words(main)[:1] or [""])[0].lower().strip(",.:")
        if head in IMPERATIVE_VERBS:
            return s[: m.end()] + main
    return None


def _action_name(section_heading: str, steps: List[str]) -> str:
    if section_heading and not _GENERIC_HEADINGS.match(section_heading):
        return section_heading
    toks = [t.strip(",.") for t in words(steps[0])[:5]]
    while len(toks) > 2 and toks[-1].lower() in {"the", "a", "an", "and", "or", "to", "your", "of", "on", "for"}:
        toks.pop()
    return " ".join(toks[:4])


_DESC_DROP = {"and", "or", "for", "to", "of", "on", "in", "at", "by", "with", "via", "the", "a", "an", "your"}


def _phrase(action_name: str, limit: int) -> str:
    toks = words(re.sub(r"^(how to|ways to)\s+", "", action_name.lower()))
    if len(toks) > limit:
        toks = [t for t in toks if t not in _DESC_DROP]
    toks = toks[:limit]
    while len(toks) > 1 and toks[-1] in _DESC_DROP | {"but", "is", "are", "doesn't", "on"}:
        toks.pop()
    return " ".join(toks)


def _description(action_name: str) -> str:
    first = (words(action_name)[:1] or [""])[0].lower()
    if first in IMPERATIVE_VERBS:
        full = _phrase(action_name, 4)
        return f"It will help {full}" if len(words(full)) == 4 else f"It will help you {full}"
    return f"It will explain {_phrase(action_name, 4)}"


_TOPIC_DROP = {"when", "while", "if", "using", "with", "the", "a", "an", "your", "my", "on", "in", "for", "to"}
_AUX = {"does", "do", "did", "is", "are", "not", "won't", "doesn't", "can't", "or", "and"}


def _title_from_topic(topic: str, kind: str) -> str:
    toks = [t for t in words(topic) if t.lower() not in _AUX and t.lower() not in _TOPIC_DROP]
    if len(toks) > 3:
        toks = toks[:1] + toks[-2:]
    if len(toks) == 2 and kind == "Troubleshooting":
        toks.append("issue")
    return " ".join(toks)


def _topic(title: str) -> str:
    t = re.sub(r"\s+(on|for|with|to|in)\s+(a |an |your |the )?(techcorp )?(smartphone|tablet|phone|device|"
               r"galaxy)s?(\s+or\s+(a |an )?(smartphone|tablet|phone))?\b.*$", "", title, flags=re.I)
    t = " ".join(w for w in words(t or title) if w.lower() not in _TOPIC_DROP)
    return t or title


class RuleExtractor:
    name = "rules"

    def extract(self, query: str, title: str, content: str) -> RawPlan:
        actions: List[RawAction] = []
        for section in split_sections(content):
            steps = [st for st in (to_imperative(s) for s in split_sentences(section.body)) if st]
            if not steps:
                continue
            name = _action_name(section.heading, steps)
            actions.append(RawAction(actionName=name, description=_description(name), steps=steps))

        topic = _topic(title)
        q = content_tokens(query) - QUERY_NOISE
        src = content_tokens(title + " " + content)
        relevance = len(q & src) / len(q) if q else 0.0
        kind = "Troubleshooting" if _TROUBLE.search(title) else "Configuration"
        return RawPlan(no_match=not actions, relevance=round(relevance, 3), topic=topic, goal_kind=kind,
                       title=_title_from_topic(topic, kind), actions=actions)