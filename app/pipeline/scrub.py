"""Zero URL leak enforcement. Applied to free-text fields only, NEVER to deeplink fields."""
import re

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_SCHEME_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://\S+", re.I)   # http://, https://, voiceassist://
_WWW = re.compile(r"\bwww\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")
_TLDS = "com|net|org|io|co|in|us|uk|info|biz|app|dev|ly|me|gov|edu|ai|tv"
_DOMAIN = re.compile(rf"\b(?:[a-z0-9\-]+\.)+(?:{_TLDS})\b(?:/\S*)?", re.I)
# words left dangling once a link is removed, e.g. "Visit  for help."
_DANGLING = re.compile(r"\b(?:visit|see|at|go to|refer to|learn more at|available at)\s*(?=[.,;:!?]|$)", re.I)

_DETECTORS = (_MD_LINK, _SCHEME_URL, _WWW, _EMAIL, _DOMAIN)


def contains_url(text: str) -> bool:
    return bool(text) and any(p.search(text) for p in _DETECTORS)


def scrub_text(text: str) -> str:
    if not text:
        return text
    out = _MD_LINK.sub(r"\1", text)
    for pattern in (_SCHEME_URL, _WWW, _EMAIL, _DOMAIN):
        out = pattern.sub("", out)
    out = _DANGLING.sub("", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"([.,;:!?])\1+", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip(" ,;:")