"""Generate metrics.md (PDF Appendix C) from the benchmark outputs.

  python -m scripts.make_metrics
Needs: outputs/results.jsonl, cache_benchmark.json; optional: cold_benchmark.json, stress.json, ablation.json,
and data/eval/manual_review.json for the human-scored accuracy rows (a template is created if missing).
"""
import json
import os
import platform
from collections import Counter

from app.config import DATA_DIR, EMBEDDING_BACKEND, EMBEDDING_MODEL, OUTPUT_DIR, ROOT, LLMSettings
from app.data_loader import catalog_uris
from app.pipeline.scrub import contains_url
from app.pipeline.validators import audit_response
from app.schema import Goal, actionCategory

REVIEW = DATA_DIR / "eval" / "manual_review.json"
SYNTAX = {"GOAL_FORMAT", "TITLE_WORDS", "TITLE_CASE", "ACTION_NAME_CASE", "DESC_PREFIX", "DESC_WORDS"}


def load(name):
    p = OUTPUT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def main():
    catalog = set(catalog_uris())
    results = [json.loads(l) for l in open(OUTPUT_DIR / "results.jsonl", encoding="utf-8")]
    valid = syntax_ok = catalog_ok = links = auto = auto_linked = leaks = 0
    codes = Counter()
    for r in results:
        a = audit_response(r["response"], catalog)
        valid += a["schema_valid"]
        errs = [i for i in a["issues"] if i["severity"] == "error"]
        codes.update(i["code"] for i in errs)
        syntax_ok += not any(i["code"] in SYNTAX for i in errs)
        leaks += contains_url(json.dumps(r["response"]).replace("voiceassist://", ""))
        for ctx in r["response"]["contexts"]:
            for act in Goal.model_validate(ctx).actions:
                for g in act.stepGroups:
                    for dl in (g.actionableDeeplink, g.validationDeeplink):
                        if dl:
                            links += 1
                            catalog_ok += dl.deeplink in catalog
                if act.category == actionCategory.auto:
                    auto += 1
                    auto_linked += act.stepGroups[0].actionableDeeplink is not None
    n = len(results)

    if not REVIEW.exists():
        REVIEW.write_text(json.dumps({"_readme": "Score each plan by hand. step_accuracy 0-3 (completeness, correctness, "
                                                 "ordering); deeplink_relevance 0-2 (exact screen=2, parent menu=1, wrong=0).",
                                      "scores": [{"input_line": i + 1, "step_accuracy": None, "deeplink_relevance": None}
                                                 for i in range(n)]}, indent=2), encoding="utf-8")
    scores = json.loads(REVIEW.read_text(encoding="utf-8"))["scores"]
    avg = lambda k: (lambda v: f"{sum(v) / len(v):.2f} (n={len(v)})" if v else "_score in data/eval/manual_review.json_")(
        [s[k] for s in scores if s.get(k) is not None])

    cache, cold, stress, abl = load("cache_benchmark.json"), load("cold_benchmark.json"), load("stress.json"), load("ablation.json")
    ex, pa = cache.get("exact", {}), cache.get("paraphrase", {})
    s = LLMSettings()
    emb = f"{EMBEDDING_MODEL} (sentence-transformers)" if EMBEDDING_BACKEND == "st" else "char n-gram TF-IDF"
    A, Ap, B = abl.get("A_ours", {}), abl.get("A_prime_top1_no_gate", {}), abl.get("B_rules", {})

    md = f"""# System Performance Metrics & Evaluation Report
**Model(s):** {s.model} via {s.base_url or 'OpenAI API'} (temperature {s.temperature})
**Embeddings:** {emb}
**Environment:** {os.cpu_count()} vCPU / RAM: _fill in_ / {platform.system()} {platform.release()}, Python {platform.python_version()}

---

## 1. Schema & Rule Compliance
Evaluated on the {n} complaints in `input.txt` (`outputs/results.jsonl`).

| Metric | Target | Measured Value |
| :--- | :--- | :--- |
| Schema-valid output lines | >= 99% | {pct(valid / n)} |
| Rule compliance (Goal / Title / Description syntax) | >= 95% | {pct(syntax_ok / n)} |
| Absolute URL leaks | 0 | {leaks} |
| Deeplink catalog validity (exact URI match) | 100% | {pct(catalog_ok / links) if links else 'n/a'} ({catalog_ok}/{links}) |
| Auto actions carrying valid actionable deeplink | >= 90% | {pct(auto_linked / auto) if auto else 'n/a'} ({auto_linked}/{auto}) |

Remaining rule errors: {dict(codes) or 'none'}.

---

## 2. Accuracy Benchmarks
Human review of each plan against its SIIS article (`data/eval/manual_review.json`).

| Evaluation Metric | Scale / Anchor | Score |
| :--- | :--- | :--- |
| Step accuracy (completeness, correctness, ordering) | 0.0 - 3.0 | {avg('step_accuracy')} |
| Deeplink relevance (exact target screen vs. parent menu) | 0.0 - 2.0 | {avg('deeplink_relevance')} |

---

## 3. Latency Benchmarks (N >= 30 requests per path)

| Execution Path | Target (P95) | P50 (ms) | P95 (ms) |
| :--- | :--- | :--- | :--- |
| Cache hit - exact query match (N={ex.get('n', '-')}) | <= 300 ms | {ex.get('p50_ms', '-')} | {ex.get('p95_ms', '-')} |
| Cache hit - unseen semantic paraphrase (N={pa.get('n', '-')}) | <= 300 ms | {pa.get('p50_ms', '-')} | {pa.get('p95_ms', '-')} |
| Cold query - full pipeline extraction & mapping (N={cold.get('n', '-')}) | <= 8000 ms | {cold.get('p50_ms', '-')} | {cold.get('p95_ms', '-')} |
"""
    if stress:
        c = stress["concurrency"]
        md += (f"\nEnd-to-end over HTTP (`scripts/stress_test.py`): exact-hit P95 {stress['exact']['p95_ms']} ms, "
               f"paraphrase-hit P95 {stress['paraphrase']['p95_ms']} ms, {c['requests']} concurrent requests with "
               f"{c['workers']} workers at {c['throughput_rps']} req/s (P95 {c['p95_ms']} ms); "
               f"problems: {stress['problems']}; API healthy {stress['ready_after_s']} s after launch.\n")
    md += f"""
---

## 4. Operational Cost & Cache Efficacy

| Metric Item | Target | Measured Value |
| :--- | :--- | :--- |
| Cold query average inference cost | Tracked | ${cold.get('avg_cost_usd', 0):.5f} ({cold.get('avg_tokens', '-')} tokens) |
| Cache hit inference cost | $0.00 | $0.00 |
| Semantic cache hit rate (on unseen paraphrases) | >= 80% | {pct(pa.get('hit_rate'))} (wrong-plan hits: {pa.get('wrong_plan_hits', '-')}, false hits: {pa.get('false_hits_on_uncached', '-')}) |
| Cost derivation method | - | (prompt tokens x ${s.price_in_per_m}/M + completion tokens x ${s.price_out_per_m}/M) |

Cache threshold {cache.get('threshold')} / margin {cache.get('margin')} chosen from the sweep in `outputs/cache_benchmark.json`
as the lowest threshold with zero wrong-plan and zero false hits.

---

## 5. Architectural Ablation Analysis

| Architecture Variant | Step Accuracy | Latency (P95) | Cost / Query | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| Baseline: Full LLM Deeplink Mapping | not run | - | - | Deliberately avoided: URIs are masked tokens, so letting the LLM pick them risks invented or altered URIs (rule 4.2.2). |
| Variant A: Hybrid BM25 + Dense Retrieval (ours, + exact-name gate) | {avg('step_accuracy')} | {A.get('p95_ms_cold', '-')} ms cold | ${A.get('cost_per_query') or 0:.5f} | Rule compliance {pct(A.get('rule_compliance'))}. Retrieval proposes candidates; a word-order-aware name gate accepts only the exact screen named in the steps. |
| Variant A': hybrid retrieval top-1, no name gate | - | same | same | Top-1 agreed with ours on {Ap.get('same_link_as_ours', '-')}/{Ap.get('auto_actions', '-')} auto actions; in {Ap.get('top1_name_matches_no_screen_in_steps', '-')} cases top-1 names no screen in the steps (decoys like "Easy mode" for "Super steady mode"). |
| Variant B: Pure Rules-Based (offline extractor) | - | {B.get('p95_ms', '-')} ms | $0.00 | Rule compliance {pct(B.get('rule_compliance'))}, {B.get('plans', '-')} plans. Free and deterministic but coarse: section headings become actions. |

---

## 6. Known Edge Cases & System Limitations
* **Official samples break the rulebook:** `sample_output.json` has 9- and 12-word descriptions (limit 7); Appendix B uses `bixby://` while the catalog uses `voiceassist://`. Our validators flag both.
* **Mismatched complaint/article pairs:** e.g. "screen stays small" is paired with a TV screen-mirroring article, "Fold screen flickers when opened" with a camera-flicker article. The engine answers `no_match` instead of inventing a plan.
* **LLM run-to-run variation:** gpt-oss is a reasoning model and varies slightly even at temperature 0; the semantic cache makes repeated and paraphrased questions fully deterministic.
* **Catalog decoys:** many near-identical setting names ("Navigation bar" vs "Hide status and navigation bars", "Factory data reset" vs "Auto factory reset"); an order-aware word-coverage gate rejects them.
* **Misleading catalog messages:** some `message` fields name a different feature than the entry opens ("View Reset Options" opens a TalkBack preset), so matching uses the validation key.
* **Non-phone catalog entries** (TV, refrigerator, air conditioner) and status-only entries are excluded from mapping.
* **Garbled source text** (spaces missing in one SIIS article) and an embedded email address are scrubbed; ungrounded LLM steps (e.g. an unstated "Release the buttons.") are dropped by the grounding check.
* **Near-tie paraphrases** between sibling black-screen plans are refused by the cache margin rule and fall back to the full pipeline.
* **Rate limits:** Groq's free tier (8k tokens/min) forces pacing between cold requests; latency is measured per request, excluding the pause.
"""
    (ROOT / "metrics.md").write_text(md, encoding="utf-8")
    print(f"metrics.md written ({n} results). Fill RAM and the manual scores in {REVIEW.relative_to(ROOT)}, then rerun.")


if __name__ == "__main__":
    main()