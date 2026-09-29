"""Fast-path semantic cache (PDF Phase 3).

Every validated plan is stored once in SQLite together with several "doors" into it:
the original query, its normalized form and the LLM's 8-10 paraphrases. Lookup is two layers:
  1. exact   : normalized text -> plan              (~1 ms)
  2. semantic: embed the query, compare with every stored door in one matrix product;
               hit only if similarity >= threshold AND it beats the best *different* plan by a margin
If the caller supplies an SIIS article, only plans built from that same article can be returned.
"""
import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import CACHE_DB_PATH, CACHE_MARGIN, CACHE_THRESHOLD


def normalize_query(text: str) -> str:
    text = re.sub(r"^\s*\d+\.\s*", "", (text or "").lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def plan_id_for(query: str) -> str:
    return hashlib.sha1(normalize_query(query).encode()).hexdigest()[:16]


def siis_hash(siis: Optional[dict]) -> Optional[str]:
    if not siis:
        return None
    body = (siis.get("title", "") + "\n" + siis.get("content", "")).strip()
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16] if body else None


@dataclass
class CacheHit:
    plan_id: str
    response: dict
    query_variations: List[str]
    match: str              # "exact" | "semantic"
    similarity: float
    margin: float
    matched_text: str


@dataclass
class Scored:
    plan_id: str
    similarity: float
    margin: float
    matched_text: str


class SemanticCache:
    def __init__(self, embedder, path: Path = CACHE_DB_PATH,
                 threshold: float = CACHE_THRESHOLD, margin: float = CACHE_MARGIN):
        self.embedder = embedder
        self.threshold, self.margin = threshold, margin
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS plans(plan_id TEXT PRIMARY KEY, query TEXT, response TEXT,
                variations TEXT, siis_hash TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS doors(plan_id TEXT, kind TEXT, text TEXT, norm TEXT, vec BLOB);
            CREATE TABLE IF NOT EXISTS info(k TEXT PRIMARY KEY, v TEXT);""")
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self):
        row = self.db.execute("SELECT v FROM info WHERE k='embedder'").fetchone()
        rows = self.db.execute("SELECT plan_id, kind, text, norm, vec FROM doors").fetchall()
        if rows and (row is None or row[0] != self.embedder.name):
            # embedder changed -> old vectors are meaningless: re-embed every stored door once
            vecs = self.embedder.encode([r[2] for r in rows])
            with self.db:
                self.db.execute("DELETE FROM doors")
                self.db.executemany("INSERT INTO doors VALUES (?,?,?,?,?)",
                                    [(r[0], r[1], r[2], r[3], v.astype(np.float32).tobytes())
                                     for r, v in zip(rows, vecs)])
            rows = self.db.execute("SELECT plan_id, kind, text, norm, vec FROM doors").fetchall()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO info VALUES ('embedder', ?)", (self.embedder.name,))
        self.door_plan = [r[0] for r in rows]
        self.door_text = [r[2] for r in rows]
        self.exact = {r[3]: r[0] for r in rows}
        self.matrix = (np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
                       if rows else np.zeros((0, 1), dtype=np.float32))
        self.plan_siis = dict(self.db.execute("SELECT plan_id, siis_hash FROM plans").fetchall())

    def __len__(self):
        return len(self.plan_siis)

    # ------------------------------------------------------------------ lookup
    def _allowed(self, plan_id: str, article: Optional[str]) -> bool:
        return article is None or self.plan_siis.get(plan_id) in (None, article)

    def score(self, query: str, article: Optional[str] = None) -> Optional[Scored]:
        """Best semantic candidate (no threshold applied) - also used by the benchmark sweep."""
        if not len(self.door_plan):
            return None
        sims = self.matrix @ self.embedder.encode([query])[0]
        best_per_plan: dict = {}
        for i, pid in enumerate(self.door_plan):
            if self._allowed(pid, article) and sims[i] > best_per_plan.get(pid, (-1.0, 0))[0]:
                best_per_plan[pid] = (float(sims[i]), i)
        if not best_per_plan:
            return None
        ranked = sorted(best_per_plan.items(), key=lambda kv: -kv[1][0])
        top_pid, (top_sim, top_i) = ranked[0]
        second = ranked[1][1][0] if len(ranked) > 1 else 0.0
        return Scored(top_pid, round(top_sim, 4), round(top_sim - second, 4), self.door_text[top_i])

    def lookup(self, query: str, article: Optional[str] = None) -> Optional[CacheHit]:
        pid = self.exact.get(normalize_query(query))
        if pid and self._allowed(pid, article):
            return self._hit(pid, "exact", 1.0, 1.0, query)
        s = self.score(query, article)
        if s and s.similarity >= self.threshold and s.margin >= self.margin:
            return self._hit(s.plan_id, "semantic", s.similarity, s.margin, s.matched_text)
        return None

    def _hit(self, pid, match, sim, margin, text) -> CacheHit:
        with self._lock:                              # one SQLite connection is shared by API threads
            response, variations = self.db.execute(
                "SELECT response, variations FROM plans WHERE plan_id=?", (pid,)).fetchone()
        return CacheHit(pid, json.loads(response), json.loads(variations), match, sim, margin, text)

    # ------------------------------------------------------------------ writing
    def put(self, query: str, response: dict, query_variations: List[str],
            article: Optional[str] = None, normalized_query: Optional[str] = None) -> str:
        pid = plan_id_for(query)
        doors = [("query", query)] + ([("normalized", normalized_query)] if normalized_query else []) + \
                [("variation", v) for v in query_variations]
        seen, unique = set(), []
        for kind, text in doors:
            n = normalize_query(text)
            if n and n not in seen:
                seen.add(n)
                unique.append((kind, text, n))
        vecs = self.embedder.encode([t for _, t, _ in unique])
        with self._lock, self.db:
            self.db.execute("DELETE FROM doors WHERE plan_id=?", (pid,))
            self.db.execute("INSERT OR REPLACE INTO plans VALUES (?,?,?,?,?,?)",
                            (pid, query, json.dumps(response), json.dumps(query_variations), article, time.time()))
            self.db.executemany("INSERT INTO doors VALUES (?,?,?,?,?)",
                                [(pid, k, t, n, v.astype(np.float32).tobytes())
                                 for (k, t, n), v in zip(unique, vecs)])
            self._load()
        return pid

    def clear(self):
        with self._lock, self.db:
            self.db.execute("DELETE FROM plans")
            self.db.execute("DELETE FROM doors")
            self._load()