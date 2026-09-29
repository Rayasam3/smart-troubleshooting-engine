"""Phase 3 benchmark: cache latency (P50/P95), determinism, and hit rate on UNSEEN paraphrases.
  python -m scripts.benchmark_cache
Writes outputs/cache_benchmark.json (used for metrics.md in Phase 4).
"""
import json
import time

import numpy as np

from app.cache.semantic_cache import plan_id_for
from app.config import CACHE_MARGIN, CACHE_THRESHOLD, HELDOUT_PATH, OUTPUT_DIR
from app.data_loader import load_inputs
from app.engine import TroubleshootEngine
from scripts.warm_cache import warm


def pct(values, p):
    return round(float(np.percentile(values, p)), 1) if values else None


def main():
    t0 = time.perf_counter()
    engine = TroubleshootEngine()
    print(f"engine ready in {time.perf_counter() - t0:.1f}s (embedder={engine.embedder.name})")
    if len(engine.cache) == 0:
        print(f"cache empty -> warming from phase2 results: {warm(engine.cache)} plans")
    cache, inputs = engine.cache, load_inputs()
    cached = [q for q in inputs if plan_id_for(q) in cache.plan_siis]
    engine.handle(cached[0])                                     # warm-up call (first encode is slower)

    # ---- 1. exact repeats: latency + determinism (N >= 30)
    exact_lat, identical = [], True
    first = {}
    while len(exact_lat) < 40:
        for q in cached:
            r = engine.handle(q)
            exact_lat.append(r["meta"]["latency_ms"])
            key = json.dumps(r["response"], sort_keys=True)
            identical &= first.setdefault(q, key) == key
            if len(exact_lat) >= 40:
                break

    # ---- 2. unseen paraphrases
    items = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))["items"]
    para_lat, rows = [], []
    for item in items:
        source = inputs[item["input_line"] - 1]
        expected = plan_id_for(source)
        for p in item["paraphrases"]:
            r = engine.handle(p)
            para_lat.append(r["meta"]["latency_ms"])
            s = cache.score(p)
            rows.append({"expected": expected, "in_cache": expected in cache.plan_siis,
                         "got": s.plan_id if s else None, "sim": s.similarity if s else 0.0,
                         "margin": s.margin if s else 0.0, "text": p})

    def rates(thr, margin):
        pos = [x for x in rows if x["in_cache"]]
        neg = [x for x in rows if not x["in_cache"]]
        hit = lambda x: x["sim"] >= thr and x["margin"] >= margin
        right = sum(1 for x in pos if hit(x) and x["got"] == x["expected"])
        same_article = sum(1 for x in pos if hit(x) and x["got"] != x["expected"] and
                           cache.plan_siis.get(x["got"]) == cache.plan_siis.get(x["expected"]))
        wrong = sum(1 for x in pos if hit(x) and x["got"] != x["expected"]) - same_article
        false_hits = sum(1 for x in neg if hit(x))
        return {"threshold": thr, "hit_rate": round(right / max(len(pos), 1), 3),
                "same_article_hits": same_article, "wrong_plan_hits": wrong,
                "false_hits_on_uncached": false_hits, "n_cached": len(pos), "n_uncached": len(neg)}

    sweep = [rates(round(t, 3), CACHE_MARGIN) for t in np.arange(0.70, 0.951, 0.025)]
    current = rates(CACHE_THRESHOLD, CACHE_MARGIN)

    print(f"\n==== Phase 3 benchmark (threshold={CACHE_THRESHOLD}, margin={CACHE_MARGIN}) ====")
    print(f"plans in cache                 : {len(cache)} ({len(cache.door_plan)} phrasings)")
    print(f"exact repeats   P50 / P95      : {pct(exact_lat, 50)} / {pct(exact_lat, 95)} ms  (N={len(exact_lat)}, target P95 <= 300)")
    print(f"paraphrases     P50 / P95      : {pct(para_lat, 50)} / {pct(para_lat, 95)} ms  (N={len(para_lat)})")
    print(f"deterministic repeats          : {identical}")
    print(f"unseen paraphrase hit rate     : {current['hit_rate'] * 100:.0f}%  (target >= 80%)")
    print(f"hits on a sibling plan (same article) : {current['same_article_hits']}")
    print(f"hits on a WRONG plan           : {current['wrong_plan_hits']}")
    print(f"false hits (source not cached) : {current['false_hits_on_uncached']} of {current['n_uncached']}")
    print("\nthreshold sweep  (pick the lowest threshold with 0 wrong-plan hits)")
    print("  thr    hit%   sibling  wrong  false")
    for r in sweep:
        print(f"  {r['threshold']:<6} {r['hit_rate'] * 100:>4.0f}%   {r['same_article_hits']:>5}   "
              f"{r['wrong_plan_hits']:>5}  {r['false_hits_on_uncached']:>5}")
    misses = [x for x in rows if x["in_cache"] and not (x["sim"] >= CACHE_THRESHOLD and x["margin"] >= CACHE_MARGIN)]
    if misses:
        print("\nmissed paraphrases (lowest similarity first):")
        for x in sorted(misses, key=lambda x: x["sim"])[:8]:
            print(f"  sim={x['sim']:.3f} margin={x['margin']:.3f}  {x['text'][:80]}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "cache_benchmark.json").write_text(json.dumps({
        "embedder": engine.embedder.name, "threshold": CACHE_THRESHOLD, "margin": CACHE_MARGIN,
        "exact": {"n": len(exact_lat), "p50_ms": pct(exact_lat, 50), "p95_ms": pct(exact_lat, 95)},
        "paraphrase": {"n": len(para_lat), "p50_ms": pct(para_lat, 50), "p95_ms": pct(para_lat, 95), **current},
        "deterministic": identical, "sweep": sweep}, indent=2), encoding="utf-8")
    print("\nwritten to outputs/cache_benchmark.json")


if __name__ == "__main__":
    main()