"""Architectural ablation (metrics.md section 5). No LLM calls.

  A  = ours: LLM extraction + hybrid BM25/dense retrieval + exact-name gate   (from results.jsonl)
  A' = same plans, but take the hybrid retriever's TOP-1 entry with no name gate
  B  = pure rules: offline rule extractor + the same mapper
Writes outputs/ablation.json.
"""
import json
import time

import numpy as np

from app.config import OUTPUT_DIR
from app.data_loader import catalog_uris, paired_inputs
from app.engine import TroubleshootEngine
from app.pipeline.deeplink_mapper import extract_targets, name_match
from app.pipeline.extract import extract
from app.pipeline.extract_rules import RuleExtractor
from app.pipeline.pipeline import finalize
from app.pipeline.validators import audit_response
from app.schema import Goal, actionCategory


def compliance(responses, catalog):
    ok = sum(1 for r in responses if audit_response(r, catalog)["schema_valid"] and
             not any(i["severity"] == "error" for i in audit_response(r, catalog)["issues"]))
    return round(ok / max(len(responses), 1), 3)


def main():
    engine = TroubleshootEngine()
    catalog = set(catalog_uris())
    results = [json.loads(l) for l in open(OUTPUT_DIR / "results.jsonl", encoding="utf-8")]

    # ---- A' : top-1 retrieval without the name gate, on the same auto actions as A
    agree = decoy = total = 0
    for r in results:
        for ctx in r["response"]["contexts"]:
            for a in Goal.model_validate(ctx).actions:
                if a.category != actionCategory.auto:
                    continue
                steps = [s for g in a.stepGroups for s in g.steps]
                cands = engine.mapper.index.search(" ".join([a.actionName, *steps]))
                if not cands:
                    continue
                total += 1
                top = cands[0].entry
                ours = a.stepGroups[0].actionableDeeplink
                agree += bool(ours) and ours.deeplink == top.deeplink
                targets = extract_targets(steps, strict=False)
                decoy += max((name_match(t, n) for t in targets for n in top.names), default=0) < 0.6

    # ---- B : pure rules
    lat, rule_resps = [], []
    for q, siis in paired_inputs():
        t = time.perf_counter()
        ext = extract(q, siis, RuleExtractor())
        goal = finalize(ext.goal, engine.mapper)[0] if ext.goal else None
        lat.append((time.perf_counter() - t) * 1000)
        rule_resps.append({"contexts": [goal.model_dump(mode="json")] if goal else []})

    cold = {}
    if (OUTPUT_DIR / "cold_benchmark.json").exists():
        cold = json.loads((OUTPUT_DIR / "cold_benchmark.json").read_text())
    report = {
        "A_ours": {"rule_compliance": compliance([r["response"] for r in results], catalog),
                   "p95_ms_cold": cold.get("p95_ms"), "cost_per_query": cold.get("avg_cost_usd")},
        "A_prime_top1_no_gate": {"auto_actions": total, "same_link_as_ours": agree,
                                 "top1_name_matches_no_screen_in_steps": decoy},
        "B_rules": {"rule_compliance": compliance(rule_resps, catalog),
                    "p95_ms": round(float(np.percentile(lat, 95)), 1), "cost_per_query": 0.0,
                    "plans": sum(1 for r in rule_resps if r["contexts"])},
    }
    (OUTPUT_DIR / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()