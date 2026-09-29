"""Phase 2: attach deeplinks + order actions.
  python -m scripts.run_phase2           # reuses outputs/phase1_results.jsonl (no LLM cost)
  python -m scripts.run_phase2 --live    # runs the full pipeline, calling the LLM again
  add --show to print every mapping decision
"""
import json
import sys
import time
from collections import Counter

from app.config import OUTPUT_DIR
from app.data_loader import catalog_uris, paired_inputs
from app.pipeline.deeplink_mapper import DeeplinkMapper
from app.pipeline.pipeline import TroubleshootPipeline, finalize
from app.pipeline.validators import audit_response
from app.schema import Goal, actionCategory


def iter_phase1():
    path = OUTPUT_DIR / "phase1_results.jsonl"
    if not path.exists():
        sys.exit("outputs/phase1_results.jsonl not found - run `python -m scripts.run_phase1` first")
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    live, show = "--live" in sys.argv, "--show" in sys.argv
    t0 = time.perf_counter()
    mapper = DeeplinkMapper()
    print(f"index ready: {len(mapper.index.entries)} usable deeplinks, embedder={mapper.index.embedder.name}, "
          f"{(time.perf_counter() - t0):.1f}s\n")
    catalog = set(catalog_uris())
    out_path = OUTPUT_DIR / "phase2_results.jsonl"
    stats, codes = Counter(), Counter()

    if live:
        pipe = TroubleshootPipeline(mapper=mapper)
        source = ((q, s, None) for q, s in paired_inputs())
    else:
        source = ((r["query"], None, r) for r in iter_phase1())

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (query, siis, p1) in enumerate(source, 1):
            t = time.perf_counter()
            if live:
                res = pipe.run(query, siis)
                goal, decisions = res.goal, res.decisions_as_dicts()
                meta = {"model": res.extraction.model, "cost_usd": res.extraction.cost_usd,
                        **({"fallback": res.extraction.fallback} if res.extraction.fallback else {})}
                variations = res.extraction.plan.query_variations if res.extraction.plan else []
            else:
                ctx = p1["response"]["contexts"]
                goal, decisions = (None, [])
                if ctx:
                    goal, ds = finalize(Goal.model_validate(ctx[0]), mapper)
                    decisions = [d.__dict__ for d in ds]
                meta = {k: v for k, v in p1["meta"].items() if k in ("model", "cost_usd", "fallback")}
                variations = p1.get("query_variations", [])
            meta["map_latency_ms"] = round((time.perf_counter() - t) * 1000, 1)

            response = {"contexts": [goal.model_dump(mode="json")] if goal else []}
            audit = audit_response(response, catalog)
            errors = [x for x in audit["issues"] if x["severity"] == "error"]
            codes.update(x["code"] for x in errors)
            stats["rows"] += 1
            stats["schema_valid"] += audit["schema_valid"]
            if goal:
                stats["plans"] += 1
                stats["clean_plans"] += not errors
                for a in goal.actions:
                    stats[f"cat_{a.category.value}"] += 1
                    dl = a.stepGroups[0].actionableDeeplink
                    if a.category == actionCategory.auto:
                        stats["auto"] += 1
                        stats["auto_with_link"] += dl is not None
                    if dl is not None:
                        stats["dummy" if dl.originalType == "placeholder" else "catalog"] += 1

            f.write(json.dumps({"query": query, "query_variations": variations, "response": response,
                                "meta": meta, "debug": {"mapping": decisions, "issues": audit["issues"]}},
                               ensure_ascii=False) + "\n")
            n_act = len(goal.actions) if goal else 0
            print(f"{i:>2}. actions={n_act:<2} errors={len(errors):<2} {meta.get('fallback', '') or '':<9} {query[:60]}")
            if show and goal:
                for a, d in zip(goal.actions, decisions):
                    dl = a.stepGroups[0].actionableDeeplink
                    print(f"      [{a.category.value:<8}] {a.actionName:<40} -> {d['kind']:<7} {dl.message if dl else '-'}")

    print("\n==== Phase 2 summary ====")
    print(f"schema-valid outputs          : {stats['schema_valid']}/{stats['rows']}")
    print(f"plans with 0 rule errors      : {stats['clean_plans']}/{stats['plans']}")
    print(f"remaining error codes         : {dict(codes) or 'none'}")
    print(f"actions auto/manual/critical  : {stats['cat_auto']}/{stats['cat_manual']}/{stats['cat_critical']}")
    pct = 100 * stats['auto_with_link'] / stats['auto'] if stats['auto'] else 100
    print(f"auto actions with deeplink    : {stats['auto_with_link']}/{stats['auto']} ({pct:.0f}%)  target >= 90%")
    print(f"deeplinks catalog / dummy     : {stats['catalog']} / {stats['dummy']}")
    print(f"written to                    : {out_path}")


if __name__ == "__main__":
    main()