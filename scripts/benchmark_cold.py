"""Cold-path latency: the full pipeline (LLM extraction + deeplink mapping), cache bypassed.

  python -m scripts.benchmark_cold                 # 20 requests, 30 s apart (Groq free tier: 8k tokens/min)
  python -m scripts.benchmark_cold --rounds 2      # 40 requests (PDF asks for N >= 30; uses ~120k tokens)
  python -m scripts.benchmark_cold --pace 5        # faster, if your plan has higher rate limits

Latency is measured per request only; the pause between requests is NOT counted.
Requests that fail at the API (e.g. rate limit) are reported separately and NOT counted as latency samples.
"""
import argparse
import json
import time

import numpy as np

from app.config import OUTPUT_DIR
from app.data_loader import catalog_uris, paired_inputs
from app.engine import TroubleshootEngine
from app.pipeline.validators import audit_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--pace", type=float, default=30.0)
    args = ap.parse_args()

    engine = TroubleshootEngine()
    catalog = set(catalog_uris())
    lat, costs, tokens = [], [], []
    errors = fallbacks = failed = 0
    model_name = "unknown"
    pairs = paired_inputs()
    total = len(pairs) * args.rounds

    for i in range(total):
        query, siis = pairs[i % len(pairs)]
        if i:
            time.sleep(args.pace)
        t0 = time.perf_counter()
        res = engine.pipeline.run(query, siis)                  # bypasses the cache on purpose
        elapsed = (time.perf_counter() - t0) * 1000
        ext = res.extraction

        if ext.fallback == "extraction_error":                  # API failure (e.g. rate limit): not a latency sample
            failed += 1
            print(f"{i + 1:>2}/{total} FAILED, not counted: {str(ext.dropped_steps[:1])[:100]}")
            continue

        lat.append(elapsed)
        costs.append(ext.cost_usd)
        tokens.append(ext.prompt_tokens + ext.completion_tokens)
        model_name = ext.model
        fallbacks += ext.fallback is not None
        audit = audit_response(res.response, catalog)
        errors += (not audit["schema_valid"]) or any(x["severity"] == "error" for x in audit["issues"])
        print(f"{i + 1:>2}/{total} {elapsed:>8.0f} ms  tokens={tokens[-1]:<6} ${costs[-1]:.5f} "
              f"{ext.fallback or '':<16} {query[:45]}")

    if not lat:
        raise SystemExit("Every request failed - check the error above (rate limit?) and try again later.")

    summary = {
        "n": len(lat),
        "failed_requests": failed,
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "avg_cost_usd": round(float(np.mean(costs)), 6),
        "avg_tokens": round(float(np.mean(tokens)), 1),
        "fallbacks": fallbacks,
        "plans_with_rule_errors": errors,
        "model": model_name,
    }
    (OUTPUT_DIR / "cold_benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\ncold path  P50 {summary['p50_ms']} ms | P95 {summary['p95_ms']} ms (target <= 8000) | "
          f"avg cost ${summary['avg_cost_usd']} | avg tokens {summary['avg_tokens']}")
    print(f"successful requests {summary['n']} | failed (not counted) {failed} | model {model_name}")
    if failed:
        print("WARNING: some requests failed - rerun when your rate limit resets for a complete benchmark.")
    print("written to outputs/cold_benchmark.json")


if __name__ == "__main__":
    main()