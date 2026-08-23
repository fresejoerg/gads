---
id: analytics.answer_suite.star
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [business_analytics]
  data_modality: [tabular]
  pin_only: true          # research instrument — reachable only by an explicit spec pin
  anti_signals:
    - routing: "never match — aah grounding-axis instrument (rung 2 (star schema)); selected only by an explicit spec pin"

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: con
      kind: duckdb.DuckDBPyConnection
  capabilities: [duckdb]

# ——— DAG TEMPLATE ———
dag:
  - id: setup_grounding
    intent: >
      Build the clean star: `from aah_star import connect; con = connect('.')`. This gives DuckDB views dim_users, dim_habits, fct_value_moments, fct_reminders, fct_subscriptions, fct_marketing_spend, fct_referrals. Print their columns.
      Also load aah_questions.json so the 25 question ids are available. Do NOT answer yet.
    worker_tier: T2
    produces: [con]
    postconditions:
      - "con is not None"

  - id: answer_all_questions
    intent: >
      Load aah_questions.json ({question_id: text}) and answer every question BY ITS ID. The
      output dicts MUST be keyed by those exact ids (t1_signups_june, t3_mrr, …) — never
      renumber to q1/q2/…. Fixed clock: today is 2026-07-16; data is complete through
      2026-07-12; "last week" = 2026-07-06..2026-07-12. Read each question's period/segment and
      apply it (June -> 2026-06-01..2026-06-30; Q2 -> 2026-04-01..2026-06-30; last week ->
      that week; a platform/region/channel -> filter on it).
      GROUNDING: a clean star schema via `from aah_star import connect; con = connect('.')`.
      Query the dim_/fct_ views (never the raw tables): a value moment is a row in
      fct_value_moments; use fct_subscriptions.is_active and .plan; dim_users has normalized
      platform/region/channel and is_internal. The star fixes names/enums/types and the
      opens-vs-completes grain — but holds NO metric definitions or business rules; compute what
      each question literally asks with SQL over the views.
      ROBUSTNESS: answer each question INDEPENDENTLY inside its own try/except; on error store
      the error string as that id's answer and CONTINUE. NEVER answer "data not available" — every
      numeric question is computable. Build the dict incrementally and, in a finally block, write
      aah_answers.json ({id: "<answer incl. the number and one-line why>"} for all 25 ids) and
      metrics.json ({id: <number>} for the numeric ones). Emit a business_analytics insight.
    depends_on: [setup_grounding]
    worker_tier: T2
    produces: [aah_answers_path, metrics_path]
    postconditions:
      - "output_type == 'file'"
    required_metrics: []

# ——— GLOBAL INVARIANTS ———
invariants:
  - "STAR ONLY: query the dim_/fct_ views via aah_star.connect(), never the raw tables. No governed metric definitions — compute each question literally."
  - "EXACT IDS: key aah_answers.json and metrics.json by the exact ids from aah_questions.json (t1_signups_june, t3_mrr, …). Never renumber to q1/q2 — the benchmark is scored by id (scripts/grade_aah.py)."
  - "NEVER PUNT: every numeric question is computable with the grounding you have. Do not answer 'data not available'/'no specific metric' — a punt is scored wrong."
  - "APPLY THE PERIOD: read each question's period and pass it — June = 2026-06-01..2026-06-30, Q2 = 2026-04-01..2026-06-30, 'last week' = 2026-07-06..2026-07-12. Ignoring the period returns the all-time value and is wrong."
  - "ROBUST HARNESS: answer each question in its own try/except; always write aah_answers.json and metrics.json in a finally block, even if some questions error."
  - "FIXED CLOCK: today is 2026-07-16, data complete through 2026-07-12; do not use the system date."
---

# Answer-Suite — grounding rung 2 (star schema)

## Rationale
Rung-2 grounding (star schema): the constant harness + the clean star (aah_star_schema), no metric definitions. The middle of the aah grounding curve (approach_docs/015).
The harness (load questions → per-question try/except loop → always-write, exact-id keys) is
IDENTICAL across the answer_suite.{raw,star,metrics} family; only the grounding block differs,
so accuracy differences are attributable to grounding alone (the aah replication design).

## When to use
Only for aah grounding-axis runs, pinned by a spec. Never Router-matched.
