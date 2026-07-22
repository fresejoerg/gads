# aah_v1 — provenance & scoring rationale

**Established:** 2026-07-21. Grounding-axis benchmark (approach_docs/015), orthogonal to the
delegation dial. Translates decisionspine's `ai-analytics-harness` (aah) into GADS.

**Source:** [d-n-ust/ai-analytics-harness](https://github.com/d-n-ust/ai-analytics-harness)
(MIT — license retained at `research/harness/aah/LICENSE.upstream`). A deterministic synthetic
habit-app warehouse (SEED=42, N_USERS=2500) with planted traps (dirty enums, integer status
codes, opens-vs-completes grain trap, internal/test users, an APAC launch cutoff, a
partnerships test channel, and an injected week-over-week anomaly for the diagnostic tier).

## External anchor

The 25 golds are **computed, not hand-typed**: `research/harness/aah/export_and_gold.py` runs
each question's `gold_sql` over the clean star and freezes the result in `expected.json`.
If the generator changes, rerun the export and the golds follow. The gold set is the external
referent; a run's answers are graded against it.

## The 25 questions (5 tiers × 5)

| tier | first unlocked at | grader | example |
|---|---|---|---|
| lookup | rung 2 (clean names) | numeric | value moments in June |
| filtered | rung 2 | numeric | web value moments in June |
| metric | rung 3 (governed defs) | numeric | MRR, ARPU, active/power users, activation |
| knowledge | rung 4 (examples/KB) | numeric + keyword | real-acquisition signups (partnerships excluded), APAC-since-launch, "retention"→days/user |
| diagnostic | rung 6 (metric tree) | diagnostic | "weekly value moments dropped — why?" (driver=days/user, cause=reminder change) |

## Answer artifact & scoring

A run must emit **`aah_answers.json`** = `{question_id: answer_text}` (and, for numeric
questions, `metrics.json` = `{question_id: number}`). `scripts/grade_aah.py` grades it:
- **numeric** — extract a number; correct within the question's tolerance (a rate answered as a
  percentage, e.g. 53 vs 0.53, is accepted).
- **keywords** — the answer names the right metric.
- **diagnostic** — the answer identifies the right driver AND the root cause.
Reports overall + per-tier accuracy, plus abstention and confident-wrong counts. Self-tested:
perfect sheet → 100%, empty → 0%.

## Reference golds (subset)

| question | gold | tol |
|---|---|---|
| t1_signups_june | 553 | 0.02 |
| t1_value_moments_june | 16041 | 0.02 |
| t3_mrr | 2685.08 | 0.02 |
| t3_active_users_last_week | 886 | 0.02 |
| t3_power_users | 151 | 0.05 |
| t3_activation_rate_june | 0.5322 | 0.03 |
| t4_real_signups_june | 453 | 0.03 |
| t4_apac_value_moments_q2 | 3852 | 0.04 |

## Validation status

Deterministic machinery validated (no LLM): `aah_star.connect()` reproduces **every** `gold_sql`
exactly; `aah_metrics.query_metric` reproduces the governed metrics. Rungs 4–6 and the local-engine
accuracy-curve runs are pending (vertical-slice-first plan, approach_docs/015).
