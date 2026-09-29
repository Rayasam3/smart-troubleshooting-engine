"""Loads the hackathon data files and pairs each input complaint with its SIIS article."""
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache

from app.config import DEEPLINKS_PATH, INPUT_PATH, SIIS_PATH


def _norm(text: str) -> str:
    text = re.sub(r"^\s*\d+\.\s*", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


@lru_cache(maxsize=1)
def load_deeplinks() -> list[dict]:
    with open(DEEPLINKS_PATH, encoding="utf-8") as f:
        return json.load(f)["deeplinks"]


@lru_cache(maxsize=1)
def catalog_uris() -> frozenset[str]:
    """Every URI that may legally appear in an output (actionable + validation)."""
    uris = set()
    for entry in load_deeplinks():
        uris.add(entry["deeplink"])
        if entry.get("validation"):
            uris.add(entry["validation"]["deeplink"])
    return frozenset(uris)


@lru_cache(maxsize=1)
def load_siis() -> list[dict]:
    with open(SIIS_PATH, encoding="utf-8") as f:
        return json.load(f)["responses"]


def load_inputs() -> list[str]:
    with open(INPUT_PATH, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_siis_for_query(query: str) -> dict | None:
    """Match a complaint to its SIIS row by text (row ids skip numbers, so never match by index)."""
    target = _norm(query)
    for row in load_siis():
        if _norm(row["original_query"]) == target:
            return row
    best, best_ratio = None, 0.0
    for row in load_siis():
        ratio = SequenceMatcher(None, target, _norm(row["original_query"])).ratio()
        if ratio > best_ratio:
            best, best_ratio = row, ratio
    return best if best_ratio >= 0.9 else None


def paired_inputs() -> list[tuple[str, dict | None]]:
    return [(q, find_siis_for_query(q)) for q in load_inputs()]