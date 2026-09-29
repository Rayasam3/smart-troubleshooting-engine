"""A step is grounded when most of its content words appear in the article, or it fuzzy-matches it."""
import re
from typing import List, Tuple

from rapidfuzz import fuzz

from app.config import GROUNDING_THRESHOLD

STOPWORDS = set("""a an the and or but if then so to of on in at by for with from into onto over under
about as is are was were be been being it its this that these those you your yours my me we our they
them their he she his her i can could will would should may might must do does did done have has had
not no yes all any some each every more most other such only own same than too very just also
there here when where which who whom what why how until while again further once both few off out up
down after before above below between through during please""".split())
UI_WORDS = set("""tap touch open navigate go select choose press swipe settings setting screen device
phone tablet smartphone option options menu button icon app apps enter find locate check make sure
ensure try use using turn switch""".split())


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {_stem(t) for t in toks if t not in STOPWORDS and t not in UI_WORDS and len(t) > 1}


class Grounder:
    def __init__(self, source_text: str):
        self.source_lower = (source_text or "").lower()
        self.source_tokens = content_tokens(source_text)

    def score(self, step: str) -> float:
        toks = content_tokens(step)
        if not toks:
            return 1.0
        overlap = len(toks & self.source_tokens) / len(toks)
        fuzzy = fuzz.partial_ratio(step.lower().rstrip("."), self.source_lower) / 100.0
        return round(max(overlap, fuzzy), 3)

    def filter(self, steps: List[str]) -> Tuple[List[str], List[dict]]:
        kept, dropped = [], []
        for step in steps:
            s = self.score(step)
            if s >= GROUNDING_THRESHOLD:
                kept.append(step)
            else:
                dropped.append({"step": step, "grounding": s})
        return kept, dropped