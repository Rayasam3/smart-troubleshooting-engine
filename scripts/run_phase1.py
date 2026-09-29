"""Run Phase 1 on every complaint in input.txt.
  python -m scripts.run_phase1            # LLM if LLM_API_KEY is set, else offline
  python -m scripts.run_phase1 --offline  # force rule-based extractor
"""
import json
import sys
import time

from app.config import OUTPUT_DIR, LLMSettings
from app.data_loader import catalog_uris, paired_inputs
from app.pipeline.extract import extract, get_extractor
from app.pipeline.validators import audit_response


def main():
    settings = LLMSettings()
    if "--offline" in sys.argv:
        settings.provider = "offline"
    extractor = get_extractor(settings)
    catalog = set(catalog_uris())
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase1_results.jsonl"

    rows, total_cost = [], 0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (query, siis) in enumerate(paired_inputs(), 1):
            t0 = time.perf_counter()
            res = extract(query, siis, extractor)
            latency = round((time.perf_counter() - t0) * 1000, 1)
            response = {"contexts": [res.goal.model_dump(mode="json")] if res.goal else []}
            audit = audit_response(response, catalog)
            errors = [x for x in audit["issues"] if x["severity"] == "error"]
            record = {
                "query": query,
                "siis_id": siis["id"] if siis else None,
                "query_variations": res.plan.query_variations if res.plan else [],
                "response": response,
                "meta": {"latency_ms": latency, "cache_hit": False, "model": res.model,
                         "cost_usd": res.cost_usd, "extractor": res.extractor,
                         "relevance": res.plan.relevance if res.plan else None,
                         "llm_no_match": res.plan.no_match if res.plan else None,
                         **({"fallback": res.fallback} if res.fallback else {})},
                "debug": {"dropped_steps": res.dropped_steps, "issues": audit["issues"]},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_cost += res.cost_usd
            rows.append((audit["schema_valid"], len(errors), res.fallback))
            n_actions = len(res.goal.actions) if res.goal else 0
            print(f"{i:>2}. {record['siis_id'] or '-':<7} actions={n_actions:<2} "
                  f"fallback={res.fallback or '-':<16} errors={len(errors):<2} {query[:50]}")

    n = len(rows)
    produced = [r for r in rows if r[2] is None]
    print("\n==== Phase 1 summary ====")
    print(f"extractor               : {getattr(extractor, 'name', '?')}")
    print(f"schema-valid outputs    : {sum(r[0] for r in rows)}/{n}")
    print(f"plans produced          : {len(produced)}/{n}")
    print(f"plans with 0 rule errors: {sum(1 for r in produced if r[1] == 0)}/{len(produced)}")
    print(f"total LLM cost (USD)    : {total_cost:.5f}")
    print(f"written to              : {out_path}")


if __name__ == "__main__":
    main()