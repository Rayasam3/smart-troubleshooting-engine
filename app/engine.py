"""The engine behind the API: semantic cache first, full pipeline on a miss, write-through on success."""
import time
from typing import Optional, Union

from app.cache.semantic_cache import SemanticCache, siis_hash
from app.data_loader import catalog_uris
from app.pipeline.deeplink_index import DeeplinkIndex
from app.pipeline.deeplink_mapper import DeeplinkMapper
from app.pipeline.pipeline import TroubleshootPipeline
from app.pipeline.validators import audit_response
from app.retrieval.embedder import Embedder


def to_article(siis: Union[None, str, dict]) -> Optional[dict]:
    """Accept the raw text the API receives, a {title, content} dict, or a full siis_responses.json row."""
    if not siis:
        return None
    if isinstance(siis, str):
        return {"title": "", "content": siis}
    if "siis_response" in siis:
        return siis["siis_response"]
    return siis


class TroubleshootEngine:
    def __init__(self, embedder: Optional[Embedder] = None, extractor=None,
                 cache: Optional[SemanticCache] = None, mapper: Optional[DeeplinkMapper] = None):
        # explicit "is None" checks: an EMPTY cache has len() == 0, so `cache or ...` would wrongly replace it
        self.embedder = embedder if embedder is not None else Embedder()   # loaded ONCE, shared by index + cache
        self.mapper = mapper if mapper is not None else DeeplinkMapper(DeeplinkIndex(self.embedder))
        self.cache = cache if cache is not None else SemanticCache(self.embedder)
        self.pipeline = TroubleshootPipeline(extractor, self.mapper)
        self.catalog = set(catalog_uris())

    def handle(self, query: str, siis_response: Union[None, str, dict] = None) -> dict:
        t0 = time.perf_counter()
        article = to_article(siis_response)
        art_hash = siis_hash(article)

        hit = self.cache.lookup(query, art_hash)
        if hit:
            return self._reply(query, hit.query_variations, hit.response, t0, cache_hit=True, model="cache",
                               cost=0.0, extra={"cache_match": hit.match, "similarity": hit.similarity})
        if article is None:
            return self._reply(query, [], {"contexts": []}, t0, cache_hit=False, model="none", cost=0.0,
                               extra={"fallback": "no_siis_context"})

        res = self.pipeline.run(query, article)
        ext = res.extraction
        variations = ext.plan.query_variations if ext.plan else []
        response = res.response
        extra = {"fallback": ext.fallback} if ext.fallback else {}
        if res.goal is not None:
            audit = audit_response(response, self.catalog)
            if audit["schema_valid"] and not [i for i in audit["issues"] if i["severity"] == "error"]:
                self.cache.put(query, response, variations, art_hash,
                               ext.plan.normalized_query if ext.plan else None)   # only validated plans
        return self._reply(query, variations, response, t0, cache_hit=False, model=ext.model,
                           cost=ext.cost_usd, extra=extra)

    @staticmethod
    def _reply(query, variations, response, t0, cache_hit, model, cost, extra) -> dict:
        return {"query": query, "query_variations": variations, "response": response,
                "meta": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "cache_hit": cache_hit,
                         "model": model, "cost_usd": round(cost, 6), **extra}}