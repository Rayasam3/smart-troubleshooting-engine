"""Turns a raw SIIS article into clean text and (heading, body) sections."""
import re
from dataclasses import dataclass, field
from typing import List

_HEADING = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$")
_STEP_PREFIX = re.compile(r"^(?:step\s*\d+\s*[:.)-]?|\d+[.)])\s*", re.I)


@dataclass
class Section:
    heading: str
    lines: List[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return " ".join(self.lines)


def strip_header(content: str) -> str:
    """Drop the 'Smartphone,Tablet Title ( Smartphone,Tablet): ' category prefix."""
    idx = content.find("): ")
    if 0 < idx < 400 and "(" in content[:idx]:
        return content[idx + 3:]
    return content


def clean_article(content: str) -> str:
    text = strip_header(content or "")
    text = text.replace('""', " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def split_sections(content: str) -> List[Section]:
    sections = [Section(heading="")]
    for raw in clean_article(content).splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            sections.append(Section(heading=_STEP_PREFIX.sub("", m.group(1)).strip()))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if s.lines or s.heading]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text or "")
    return [p.strip() for p in parts if p.strip()]