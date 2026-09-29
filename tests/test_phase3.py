"""Phase 3: semantic cache + engine."""
import zlib

import numpy as np
import pytest

from app.cache.semantic_cache import SemanticCache, normalize_query, plan_id_for
from app.data_loader import find_siis_for_query, load_inputs
from app.engine import TroubleshootEngine
from app.pipeline.deeplink_index import DeeplinkIndex
from app.pipeline.deeplink_mapper import DeeplinkMapper
from app.pipeline.extract_rules import RuleExtractor
from app.retrieval.embedder import Embedder


class BagOfWords:
    name = "test:bow"

    def encode(self, texts):
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in normalize_query(t).split():
                out[i, zlib.crc32(w.encode()) % 512] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


PLAN_A = {"contexts": [{"goal": "a", "title": "Screen crack", "score": 0.9, "actions": []}]}
PLAN_B = {"contexts": [{"goal": "b", "title": "Touch lag", "score": 0.9, "actions": []}]}


@pytest.fixture
def cache(tmp_path):
    c = SemanticCache(BagOfWords(), tmp_path / "c.db", threshold=0.6, margin=0.05)
    c.put("my phone screen is cracked", PLAN_A, ["cracked phone display", "screen broken glass cracked"], "art1")
    c.put("touch screen is slow and laggy", PLAN_B, ["touch input lag", "slow touch response"], "art2")
    return c


def test_exact_hit_ignores_case_and_punctuation(cache):
    hit = cache.lookup("My phone screen is CRACKED!!")
    assert hit.match == "exact" and hit.response == PLAN_A


def test_paraphrase_hits_via_stored_variation(cache):
    hit = cache.lookup("broken cracked screen glass")
    assert hit and hit.match == "semantic" and hit.response == PLAN_A


def test_unrelated_query_misses(cache):
    assert cache.lookup("battery drains overnight") is None


def test_different_article_never_returned(cache):
    assert cache.lookup("my phone screen is cracked", article="art2") is None
    assert cache.lookup("my phone screen is cracked", article="art1") is not None


def test_near_tie_between_plans_is_a_miss(tmp_path):
    c = SemanticCache(BagOfWords(), tmp_path / "t.db", threshold=0.3, margin=0.2)
    c.put("screen flickers", PLAN_A, [])
    c.put("screen goes black", PLAN_B, [])
    assert c.lookup("screen") is None           # equally close to both -> refuse to guess


def test_cache_survives_restart(tmp_path):
    path = tmp_path / "p.db"
    SemanticCache(BagOfWords(), path).put("screen is cracked", PLAN_A, ["broken display"])
    reopened = SemanticCache(BagOfWords(), path, threshold=0.5)
    assert reopened.lookup("screen is cracked").response == PLAN_A
    assert plan_id_for("Screen is cracked.") in reopened.plan_siis


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    emb = Embedder("tfidf")
    mapper = DeeplinkMapper(DeeplinkIndex(emb))
    cache = SemanticCache(emb, tmp_path_factory.mktemp("e") / "e.db")
    return TroubleshootEngine(embedder=emb, extractor=RuleExtractor(), cache=cache, mapper=mapper)


def test_engine_miss_then_hit_is_identical_and_free(engine):
    query = load_inputs()[12]                    # cracked screen
    siis = find_siis_for_query(query)
    first = engine.handle(query, siis)
    second = engine.handle(query)                # no SIIS needed once cached
    assert first["meta"]["cache_hit"] is False and second["meta"]["cache_hit"] is True
    assert second["response"] == first["response"]
    assert second["meta"]["cost_usd"] == 0.0


def test_engine_accepts_raw_text_siis(engine):
    query = load_inputs()[19]
    raw = find_siis_for_query(query)["siis_response"]["content"]
    r = engine.handle(query, raw)
    assert r["response"]["contexts"]


def test_unknown_query_without_siis_falls_back(engine):
    r = engine.handle("my smartwatch strap snapped")
    assert r["response"] == {"contexts": []} and r["meta"]["fallback"] == "no_siis_context"