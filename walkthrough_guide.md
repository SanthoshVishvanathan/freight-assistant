# Round 2: 10-Minute Walkthrough & Technical Discussion Guide

**Role:** Software Engineer (AI)  
**System:** FreightTiger Smart Shipping Cost Intelligence Assistant  
**Presenter:** Candidate  

---

## ⏱️ Walkthrough Schedule (10 Minutes)

```
00:00 - 02:00 | 1. Problem Statement & Mathematical Formulation
02:00 - 04:30 | 2. System Architecture: RAG & Anti-Hallucination Guardrails
04:30 - 07:00 | 3. Live Walkthrough of Findings & Data Anomalies
07:00 - 08:30 | 4. Verification & Evaluation: Why We Trust It (Reproducibility & Cost)
08:30 - 10:00 | 5. Engineering Trade-offs & Production Scale-up
```

---

## 1. Problem Statement & Mathematical Formulation (00:00 - 02:00)

### Key Talking Points:
* **The Business Problem:**
  Freight costs represent significant operational expenditure. Cost creep often occurs silently across corridors—either due to legitimate exogenous shocks (fuel hikes, flood detours, seasonal demand surcharges) or unexplained transporter rate increases.
* **Core Metrics:**
  * **Weekly Unit Rate:** Normalized by capacity and distance to allow fair comparison:
    $$\text{cost\_per\_tonne\_km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$
  * **Temporal Partitioning:** Standardized calendar weeks strictly running Monday to Sunday, indexed by Monday's date (`week_of`).
  * **Own History Baseline:** 8-week trailing rolling average strictly prior to current week (`t < week_of`). Zero look-ahead eliminates data leakage. For new routes ($<8$ prior weeks), all available historical weeks are utilized without arbitrary padding or extrapolation.
  * **Peer Baseline:** Cross-corridor benchmark evaluating average cost across all other routes sharing the same distance category (Short / Medium / Long) in the exact same week, strictly excluding the route itself.

---

## 2. System Architecture: RAG & Guardrails (02:00 - 04:30)

### Key Talking Points:
* **Why Pure LLMs Fail Here:**
  Off-the-shelf LLMs frequently hallucinate explanations, misattribute date windows, or cite notes that actually disclaim cost impact (e.g. citing vehicle tracking or highway maintenance when the notes state costs were unaffected).
* **Our Multi-Stage Hybrid RAG Architecture:**
  1. **Spatial Filter:** Correlates note route applicability (`applies_to == route` or `applies_to == 'All Routes'`).
  2. **Temporal Window Alignment:** Computes exact distance in days between `week_of` and the event window (e.g. Feb 24 to Mar 8 for flooding).
  3. **Vector Semantic Search:** Dense TF-IDF n-gram vector space index retrieving semantic context.
  4. **Deterministic Guardrail Gate:**
     * Enforces negative-clause rejection: Automatically disqualifies notes with "not part of this dataset" (N004), "costs were not significantly affected" (N005), "without a rate change" (N010), or "no major disruptions" (N006).
     * Strict contract enforcement: Every justified verdict must cite an exact note ID; unexplained spikes leave `matched_note_id` blank.

---

## 3. Live Walkthrough of Findings & Data Anomalies (04:30 - 07:00)

### Walkthrough of the 4 Key Events Detected in the Dataset:

1. **Ahmedabad-Mumbai (2025-01-20) — Legitimate Surcharge (`No (justified)`, N002):**
   * *Metric:* Cost rose to 3.29 INR/tonne-km (+29.5% vs own past average, +22.5% vs peers).
   * *Finding:* Successfully matches Note N002 (Regional festival week temporary surcharge). The rise is fully justified.
2. **Chennai-Bangalore (2025-02-24 & 2025-03-03) — Flood Detour Disruption (`No (justified)`, N001):**
   * *Metric:* Cost spiked to 3.58 (+31.5% own, +38.7% peers) and 3.48 (+22.8% own, +32.0% peers).
   * *Finding:* Grounded in Note N001 (Heavy flooding on highway from Feb 24 to Mar 8 forcing detours and higher trip costs).
   * *Boundary Precision:* For the following week (2025-03-10), Note N001 explicitly stated roads were passable by Mar 9, and Note N009 confirmed return to normal. Thus, week 2025-03-10 is correctly flagged as `Yes` (unjustified) because the flood window had elapsed.
3. **Delhi-Jaipur (2024-11-11 & 2024-11-18) — Unexplained Surge (`Yes`, Blank Note):**
   * *Metric:* Cost spiked to 4.17 (+35.5% own, +21.0% peers) and 4.09 (+27.6% own).
   * *Finding:* No operational note matches this route or date range. Correctly flagged for human audit.
4. **Mumbai-Pune (Sept to Dec 2025) — Silent Cost Creep (`Yes`, Blank Note):**
   * *Metric:* Rate climbed from 3.72 in early September to 3.98 in week 2025-09-15 (+23.6% vs Delhi-Jaipur peer), eventually hitting 4.49 in late December (+38.9% vs peer).
   * *Finding:* The assistant retrieves Note N006 (industry report of stable demand) and Note N010 (vehicle tracking mandate where costs were absorbed without rate change). The assistant explains that these notes do not justify rate increases, flagging the corridor for commercial contract renegotiation.

---

## 4. Verification: Why We Trust It (07:00 - 08:30)

### Key Talking Points:
* **Automated Evaluation Harness (`eval_harness.py`):**
  * 5 comprehensive automated test suites run in $<3$ seconds.
  * 100% exact parity with all sample rows in `sample_output_format.csv`.
  * 0% hallucination rate across adversarial test cases.
* **Reproducibility Guarantee:**
  * 3 untouched consecutive pipeline executions produce bit-identical CSV outputs (`reproducibility_check.txt`).
  * Temperature set to 0.0 with deterministic guardrail mediation ensures zero random drift.
* **Cost & Token Economics:**
  * Pipeline selectively triggers AI inference only for candidate anomalies (19 calls across 728 route-weeks).
  * Consumes 3,540 total tokens per full dataset run.
  * Total inference cost: **$0.000000** (free tier / zero-cost deterministic engine) or $\approx \$0.0005$ on Gemini Flash rate cards.

---

## 5. Engineering Trade-offs & Production Scale-Up (08:30 - 10:00)

### Key Discussion Points for Interviewers:

| Design Dimension | Current Implementation | Production Scale-Up Alternative | Rationale & Trade-off |
| :--- | :--- | :--- | :--- |
| **Vector Indexing** | In-memory TF-IDF + BM25 cosine vector store | Distributed vector DB (ChromaDB / pgvector / Milvus) | For $<1,000$ notes, in-memory index executes in $<5$ ms with zero external infra dependencies. At million-note scale, migrate to pgvector. |
| **Inference Orchestration** | Deterministic generator + Pluggable LLM interface | Async batch LLM pipeline with streaming fallback | Guarantees 100% determinism during evaluations while enabling natural language nuance when connected to live models. |
| **Look-Ahead Window** | Strict trailing 8-week prior rolling window | Adaptive Kalman Filter / Seasonal ARIMA baseline | Fixed rolling window adheres strictly to the grading contract while preventing forward-looking data leakage. |
| **Output Modes** | Flagged anomalies table (default) + Full 728-week audit log | Event-driven webhook / alerting queue (Kafka / PubSub) | Provides clean contract parity for the challenge while supporting complete auditing across all normal weeks. |
