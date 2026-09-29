"""Fill the semantic cache with validated plans from outputs/phase2_results.jsonl (no LLM calls).

  python -m scripts.warm_cache            # add / refresh plans
  python -m scripts.warm_cache --rebuild  # empty the cache first
"""
import json
import sys

from app.cache.semantic_cache import SemanticCache, siis_hash
from app.config import OUTPUT_DIR
from app.data_loader import catalog_uris, find_siis_for_query
from app.pipeline.validators import audit_response


def warm(cache: SemanticCache, rebuild: bool = False) -> int:
    path = OUTPUT_DIR / "phase2_results.jsonl"
    if not path.exists():
        sys.exit("outputs/phase2_results.jsonl not found - run `python -m scripts.run_phase2` first")
    if rebuild:
        cache.clear()
    catalog, stored = set(catalog_uris()), 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r["response"]["contexts"]:
            continue                                   # fallbacks are never cached
        audit = audit_response(r["response"], catalog)
        if not audit["schema_valid"] or any(i["severity"] == "error" for i in audit["issues"]):
            continue                                   # only validated plans are cached
        row = find_siis_for_query(r["query"])
        cache.put(r["query"], r["response"], r.get("query_variations", []),
                  siis_hash(row["siis_response"]) if row else None)
        stored += 1
    return stored


if __name__ == "__main__":
    from app.engine import TroubleshootEngine

    c = TroubleshootEngine().cache          # same embedder setup the API uses
    n = warm(c, rebuild="--rebuild" in sys.argv)
    doors = len(c.door_plan)
    print(f"cached {n} plans ({doors} stored phrasings, {doors / max(n, 1):.1f} per plan), embedder={c.embedder.name}")