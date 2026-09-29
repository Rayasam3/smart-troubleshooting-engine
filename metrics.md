# System Performance Metrics & Evaluation Report
**Model(s):** openai/gpt-oss-120b via https://api.groq.com/openai/v1 (temperature 0.0)
**Embeddings:** BAAI/bge-small-en-v1.5 (sentence-transformers)
**Environment:** 12 vCPU / RAM: _fill in_ / Windows 10, Python 3.11.9

---

## 1. Schema & Rule Compliance
Evaluated on the 20 complaints in `input.txt` (`outputs/results.jsonl`).

| Metric | Target | Measured Value |
| :--- | :--- | :--- |
| Schema-valid output lines | >= 99% | 100.0% |
| Rule compliance (Goal / Title / Description syntax) | >= 95% | 100.0% |
| Absolute URL leaks | 0 | 0 |
| Deeplink catalog validity (exact URI match) | 100% | 100.0% (11/11) |
| Auto actions carrying valid actionable deeplink | >= 90% | 100.0% (7/7) |

Remaining rule errors: none.

---

## 2. Accuracy Benchmarks
Human review of each plan against its SIIS article (`data/eval/manual_review.json`).

| Evaluation Metric | Scale / Anchor | Score |
| :--- | :--- | :--- |
| Step accuracy (completeness, correctness, ordering) | 0.0 - 3.0 | _score in data/eval/manual_review.json_ |
| Deeplink relevance (exact target screen vs. parent menu) | 0.0 - 2.0 | _score in data/eval/manual_review.json_ |

---

## 3. Latency Benchmarks (N >= 30 requests per path)

| Execution Path | Target (P95) | P50 (ms) | P95 (ms) |
| :--- | :--- | :--- | :--- |
| Cache hit - exact query match (N=40) | <= 300 ms | 0.1 | 0.1 |
| Cache hit - unseen semantic paraphrase (N=60) | <= 300 ms | 32.8 | 39.3 |
| Cold query - full pipeline extraction & mapping (N=20) | <= 8000 ms | 184.7 | 5110.5 |

End-to-end over HTTP (`scripts/stress_test.py`): exact-hit P95 5.6 ms, paraphrase-hit P95 45.9 ms, 160 concurrent requests with 8 workers at 29.6 req/s (P95 393.3 ms); problems: {'non_json': 0, 'http_errors': 0, 'schema_invalid': 0, 'rule_errors': 0, 'url_leaks': 0}; API healthy 5.1 s after launch.

---

## 4. Operational Cost & Cache Efficacy

| Metric Item | Target | Measured Value |
| :--- | :--- | :--- |
| Cold query average inference cost | Tracked | $0.00015 (444.6 tokens) |
| Cache hit inference cost | $0.00 | $0.00 |
| Semantic cache hit rate (on unseen paraphrases) | >= 80% | 86.3% (wrong-plan hits: 0, false hits: 0) |
| Cost derivation method | - | (prompt tokens x $0.15/M + completion tokens x $0.6/M) |

Cache threshold 0.85 / margin 0.02 chosen from the sweep in `outputs/cache_benchmark.json`
as the lowest threshold with zero wrong-plan and zero false hits.

---

## 5. Architectural Ablation Analysis

| Architecture Variant | Step Accuracy | Latency (P95) | Cost / Query | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| Baseline: Full LLM Deeplink Mapping | not run | - | - | Deliberately avoided: URIs are masked tokens, so letting the LLM pick them risks invented or altered URIs (rule 4.2.2). |
| Variant A: Hybrid BM25 + Dense Retrieval (ours, + exact-name gate) | _score in data/eval/manual_review.json_ | 5110.5 ms cold | $0.00015 | Rule compliance 100.0%. Retrieval proposes candidates; a word-order-aware name gate accepts only the exact screen named in the steps. |
| Variant A': hybrid retrieval top-1, no name gate | - | same | same | Top-1 agreed with ours on 2/7 auto actions; in 5 cases top-1 names no screen in the steps (decoys like "Easy mode" for "Super steady mode"). |
| Variant B: Pure Rules-Based (offline extractor) | - | 708.6 ms | $0.00 | Rule compliance 100.0%, 20 plans. Free and deterministic but coarse: section headings become actions. |

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
