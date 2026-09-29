"""Hybrid retrieval over the deeplink catalog: BM25 (exact words) + dense embeddings (meaning),
merged with Reciprocal Rank Fusion. Matching uses ONLY descriptive fields, never the masked URI."""
import hashlib
import json
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import INDEX_DIR, RETRIEVAL_TOP_K, RRF_K
from app.data_loader import load_deeplinks
from app.retrieval.embedder import Embedder

USABLE_TYPES = {"onURL", "offURL", "onClickURL", "updateURL"}
# appliances / TV entries can never be the target of a phone troubleshooting step
NOT_PHONE = re.compile(r"\bTV Settings\b|refrigerator|air conditioner|washer|dryer|oven", re.I)
BOILERPLATE = re.compile(r"\b(via|in) device settings on the device\b|\bsettings page\b|\bon the device\b|"
                         r"\bin device settings\b|\bdevice settings\b", re.I)
_VERB_PREFIX = re.compile(r"^(enable|disable|view|adjust|increase|decrease|set|check|open|turn on|turn off)\s+", re.I)
_STOP = set("a an the and or of to for on in at by with via your you it its this that is are be from".split())


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 1]


@dataclass
class CatalogEntry:
    id: str
    deeplink: str
    description: str
    message: str
    original_type: str
    qna: str
    key: str                     # the setting's real name, e.g. "Touch sensitivity"
    validation_link: Optional[str]

    @property
    def names(self) -> List[str]:
        """Names of the setting this entry opens. The validation key is the real setting name; messages are
        sometimes misleading (e.g. "View Reset Options" opens a TalkBack preset), so they are used only
        when an entry has no key."""
        out: List[str] = []
        sources = [self.key] if self.key else [_VERB_PREFIX.sub("", self.message or "").strip()]
        for raw in sources:
            for n in (raw, re.sub(r"\s*\([^)]*\)", "", raw or "").strip()):  # "Back up data (TechCorp Cloud)"
                if n and n.lower() not in {o.lower() for o in out}:
                    out.append(n)
        return out

    @property
    def doc_text(self) -> str:
        desc = BOILERPLATE.sub("", self.description or "")
        return f"{self.key}. {self.message}. {desc}. {self.qna}"


def load_entries() -> List[CatalogEntry]:
    entries = []
    for raw in load_deeplinks():
        if raw.get("originalType") not in USABLE_TYPES:
            continue
        text = f"{raw.get('description', '')} {raw.get('message', '')}"
        if NOT_PHONE.search(text):
            continue
        val = raw.get("validation") or {}
        entries.append(CatalogEntry(
            id=raw["id"], deeplink=raw["deeplink"], description=raw.get("description", ""),
            message=raw.get("message", ""), original_type=raw["originalType"],
            qna=raw.get("qna_description") or "", key=val.get("key") or "",
            validation_link=val.get("deeplink")))
    return entries


@dataclass
class Candidate:
    entry: CatalogEntry
    fused: float        # RRF score (rank-based, comparable across queries)
    dense: float        # cosine similarity query <-> entry
    bm25: float         # raw BM25 score


class DeeplinkIndex:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.entries = load_entries()
        self.embedder = embedder or Embedder()
        docs = [e.doc_text for e in self.entries]
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.embedder.fit(docs)
        self.vectors = self._load_or_build_vectors(docs)

    def _load_or_build_vectors(self, docs: List[str]) -> np.ndarray:
        """Embeddings are cached on disk so startup is instant after the first run."""
        digest = hashlib.sha1((self.embedder.name + json.dumps(docs)).encode()).hexdigest()[:12]
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", self.embedder.name)
        path = INDEX_DIR / f"catalog_{safe}_{digest}.npy"
        if self.embedder.backend == "st" and path.exists():
            return np.load(path)
        vecs = self.embedder.encode(docs)
        if self.embedder.backend == "st":
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, vecs)
        return vecs

    def search(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Candidate]:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        dense_scores = self.vectors @ self.embedder.encode([query])[0]
        bm25_rank = np.argsort(-bm25_scores)
        dense_rank = np.argsort(-dense_scores)
        fused = np.zeros(len(self.entries))
        for rank, idx in enumerate(bm25_rank[: k * 4]):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dense_rank[: k * 4]):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
        top = np.argsort(-fused)[:k]
        return [Candidate(self.entries[i], float(fused[i]), float(dense_scores[i]), float(bm25_scores[i]))
                for i in top if fused[i] > 0]