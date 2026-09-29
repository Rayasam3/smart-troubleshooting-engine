"""Final deliverable: outputs/results.jsonl - one API-shaped JSON object per complaint in input.txt.

  python -m scripts.run_results          # uses the cache (plans built earlier are reused)
  python -m scripts.run_results --cold   # empty temporary cache: every complaint runs the full pipeline (LLM)
"""
import json
import sys
import tempfile
from pathlib import Path

from app.cache.semantic_cache import SemanticCache
from app.config import OUTPUT_DIR
from app.data_loader import catalog_uris, paired_inputs
from app.engine import TroubleshootEngine
from app.pipeline.scrub import contains_url
from app.pipeline.validators import audit_response


def main():
    engine = TroubleshootEngine()
    if "--cold" in sys.argv:
        engine.cache = SemanticCache(engine.embedder, Path(tempfile.mkdtemp()) / "cold.db")
    catalog = set(catalog_uris())
    out = OUTPUT_DIR / "results.jsonl"
    n = valid = clean = hits = leaks = 0
    cost = 0.0
    with open(out, "w", encoding="utf-8") as f:
        for query, siis in paired_inputs():
            r = engine.handle(query, siis)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            audit = audit_response(r["response"], catalog)
            errors = [i for i in audit["issues"] if i["severity"] == "error"]
            n += 1
            valid += audit["schema_valid"]
            clean += audit["schema_valid"] and not errors
            hits += r["meta"]["cache_hit"]
            leaks += contains_url(json.dumps(r["response"]).replace("voiceassist://", ""))
            cost += r["meta"]["cost_usd"]
            print(f"{n:>2}. hit={str(r['meta']['cache_hit']):<5} {r['meta']['latency_ms']:>8} ms  "
                  f"actions={len(r['response']['contexts'][0]['actions']) if r['response']['contexts'] else 0:<2} "
                  f"{r['meta'].get('fallback', ''):<10} {query[:55]}")
    print(f"\nschema-valid {valid}/{n} | zero rule errors {clean}/{n} | URL leaks {leaks} | "
          f"cache hits {hits}/{n} | LLM cost ${cost:.4f}\nwritten to {out}")


if __name__ == "__main__":
    main()