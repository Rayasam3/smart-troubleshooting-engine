"""End-to-end stress test against the RUNNING API (PDF Phase 4).

  uvicorn app.main:app --port 8000          (in one terminal)
  python -m scripts.stress_test             (in another)

Checks: cold-start readiness, cache-hit latency (exact + unseen paraphrases), concurrency,
pure-JSON delivery, schema compliance and zero URL leaks on EVERY response.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from app.config import HELDOUT_PATH, OUTPUT_DIR
from app.data_loader import catalog_uris, load_inputs
from app.pipeline.validators import audit_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    client = httpx.Client(base_url=args.url, timeout=60)
    catalog = set(catalog_uris())

    t0 = time.perf_counter()
    while True:
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.perf_counter() - t0 > 180:
            raise SystemExit("API did not become healthy within 180 s")
        time.sleep(0.5)
    ready_s = round(time.perf_counter() - t0, 1)
    info = client.get("/v1/info").json()
    print(f"healthy after {ready_s}s | {info}")

    problems = {"non_json": 0, "http_errors": 0, "schema_invalid": 0, "rule_errors": 0, "url_leaks": 0}
    lock = threading.Lock()

    def flag(key, n=1):
        with lock:                                   # counters are shared by the worker threads
            problems[key] += n

    def call(query):
        t = time.perf_counter()
        r = client.post("/v1/troubleshoot", json={"query": query})
        wall = (time.perf_counter() - t) * 1000
        if r.status_code != 200:
            flag("http_errors")
        if not r.headers.get("content-type", "").startswith("application/json"):
            flag("non_json")
            return wall, None
        body = r.json()
        audit = audit_response(body.get("response", {}), catalog)
        errs = [i for i in audit["issues"] if i["severity"] == "error"]
        flag("schema_invalid", int(not audit["schema_valid"]))
        flag("url_leaks", sum(i["code"] == "URL_LEAK" for i in errs))
        flag("rule_errors", int(bool(errs)))
        return wall, body

    inputs = load_inputs()
    exact = [call(inputs[i % len(inputs)]) for i in range(40)]
    exact_hits = [w for w, b in exact if b and b["meta"]["cache_hit"]]
    paras = [p for item in json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))["items"] for p in item["paraphrases"]]
    para = [call(p) for p in paras]
    para_hits = [w for w, b in para if b and b["meta"]["cache_hit"]]

    burst = (inputs + paras) * 2
    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        conc = list(pool.map(call, burst))
    elapsed = time.perf_counter() - t

    pct = lambda xs, p: round(float(np.percentile(xs, p)), 1) if xs else None
    report = {
        "ready_after_s": ready_s, "engine": info,
        "exact": {"n": len(exact), "hits": len(exact_hits), "p50_ms": pct(exact_hits, 50), "p95_ms": pct(exact_hits, 95)},
        "paraphrase": {"n": len(para), "hits": len(para_hits), "hit_rate": round(len(para_hits) / len(para), 3),
                       "p50_ms": pct(para_hits, 50), "p95_ms": pct(para_hits, 95)},
        "concurrency": {"requests": len(burst), "workers": args.workers,
                        "throughput_rps": round(len(burst) / elapsed, 1),
                        "p95_ms": pct([w for w, _ in conc], 95)},
        "problems": problems,
    }
    (OUTPUT_DIR / "stress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nexact hits       {report['exact']['hits']}/40  P50 {report['exact']['p50_ms']} ms  P95 {report['exact']['p95_ms']} ms (HTTP, end to end)")
    print(f"paraphrase hits  {report['paraphrase']['hits']}/60  P50 {report['paraphrase']['p50_ms']} ms  P95 {report['paraphrase']['p95_ms']} ms")
    print(f"concurrency      {len(burst)} requests x {args.workers} workers -> {report['concurrency']['throughput_rps']} req/s, "
          f"P95 {report['concurrency']['p95_ms']} ms")
    print(f"problems         {problems}")
    print("written to outputs/stress.json")


if __name__ == "__main__":
    main()