"""Phase 4: REST API contract."""
import time

import pytest
from fastapi.testclient import TestClient

from app.cache.semantic_cache import SemanticCache
from app.data_loader import find_siis_for_query, load_inputs
from app.engine import TroubleshootEngine
from app.main import create_app
from app.pipeline.deeplink_index import DeeplinkIndex
from app.pipeline.deeplink_mapper import DeeplinkMapper
from app.pipeline.extract_rules import RuleExtractor
from app.retrieval.embedder import Embedder
from app.schema import ContextDeeplinkResponse


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    def factory():
        emb = Embedder("tfidf")
        return TroubleshootEngine(embedder=emb, extractor=RuleExtractor(), mapper=DeeplinkMapper(DeeplinkIndex(emb)),
                                  cache=SemanticCache(emb, tmp_path_factory.mktemp("api") / "c.db"))
    with TestClient(create_app(factory, background=False)) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_troubleshoot_returns_pure_valid_json_then_hits_cache(client):
    q = load_inputs()[12]
    siis = find_siis_for_query(q)["siis_response"]["content"]
    r1 = client.post("/v1/troubleshoot", json={"query": q, "siis_response": siis})
    assert r1.status_code == 200 and r1.headers["content-type"].startswith("application/json")
    body = r1.json()
    ContextDeeplinkResponse.model_validate(body["response"])
    assert set(body) == {"query", "query_variations", "response", "meta"}
    assert {"latency_ms", "cache_hit", "model", "cost_usd"} <= set(body["meta"])
    r2 = client.post("/v1/troubleshoot", json={"query": q}).json()
    assert r2["meta"]["cache_hit"] is True and r2["response"] == body["response"]


def test_unknown_without_siis_is_graceful(client):
    body = client.post("/v1/troubleshoot", json={"query": "my smartwatch strap snapped"}).json()
    assert body["response"] == {"contexts": []} and body["meta"]["fallback"] == "no_siis_context"


def test_bad_request_is_json_422(client):
    r = client.post("/v1/troubleshoot", json={"query": ""})
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/json")


def test_demo_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Troubleshooting" in r.text


def test_health_is_503_while_starting():
    def slow():
        time.sleep(2)
        raise RuntimeError("never ready")
    with TestClient(create_app(slow, background=True)) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/v1/troubleshoot", json={"query": "x"}).status_code == 503


def test_engine_crash_never_leaks_stack_trace():
    class Broken:
        def handle(self, *a):
            raise RuntimeError("secret internals")
    with TestClient(create_app(lambda: Broken(), background=False)) as c:
        r = c.post("/v1/troubleshoot", json={"query": "screen black"})
        assert r.status_code == 500 and "secret" not in r.text
        assert r.json()["meta"]["fallback"] == "internal_error"