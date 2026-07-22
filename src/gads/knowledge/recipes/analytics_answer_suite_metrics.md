---
id: analytics.answer_suite.metrics
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [business_analytics]
  data_modality: [tabular]
  anti_signals:
    - routing: "never match — aah grounding-axis instrument (rung 3 (governed semantic layer)); selected only by an explicit spec pin"

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: con
      kind: aah_metrics.SemanticLayer
  capabilities: [duckdb]

# ——— DAG TEMPLATE ———
dag:
  - id: setup_grounding
    intent: >
      Set up the governed layer: `from aah_metrics import query_metric, query_rows, list_metrics; print(list_metrics())`. The layer builds automatically on first use; no layer object to thread.
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
      GROUNDING: a governed semantic layer via `from aah_metrics import query_metric`. Map every
      question to a governed metric: value moments->value_moments; signups->new_signups;
      spend->marketing_spend; annual subs->active_subscriptions(filters={{"plan":"annual"}});
      MRR/ARPU->mrr/arpu; active users->active_users; power users->power_users; activation->
      activation_rate; retention/frequency->days_per_user; depth->moments_per_day. Call e.g.
      query_metric("value_moments", start="2026-06-01", end="2026-06-30"),
      query_metric("active_users", period="last_week"),
      query_metric("value_moments", filters={{"platform":"ios"}}, start="2026-06-01", end="2026-06-30").
      The metric bakes in the correct grain/population/definition — never re-derive it or add your
      own internal/test filter. Only if no metric fits, fall back to SQL via aah_star.connect().
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
  - "GOVERNED METRICS: answer via query_metric from aah_metrics; never hand-write a metric's SQL or add a population filter it already applies. This rung provides metric DEFINITIONS only — not world-facts (APAC cutoff, partnerships), so knowledge/diagnostic questions may still miss."
  - "EXACT IDS: key aah_answers.json and metrics.json by the exact ids from aah_questions.json (t1_signups_june, t3_mrr, …). Never renumber to q1/q2 — the benchmark is scored by id (scripts/grade_aah.py)."
  - "NEVER PUNT: every numeric question is computable with the grounding you have. Do not answer 'data not available'/'no specific metric' — a punt is scored wrong."
  - "APPLY THE PERIOD: read each question's period and pass it — June = 2026-06-01..2026-06-30, Q2 = 2026-04-01..2026-06-30, 'last week' = 2026-07-06..2026-07-12. Ignoring the period returns the all-time value and is wrong."
  - "ROBUST HARNESS: answer each question in its own try/except; always write aah_answers.json and metrics.json in a finally block, even if some questions error."
  - "FIXED CLOCK: today is 2026-07-16, data complete through 2026-07-12; do not use the system date."
---

# Answer-Suite — grounding rung 3 (governed semantic layer)

## Rationale
Rung-3 grounding (governed semantic layer): the constant harness + governed metric definitions. Unlocks the metric tier; withholds world-facts and the driver graph (rungs 4-6). Top of the built slice (approach_docs/015).
The harness (load questions → per-question try/except loop → always-write, exact-id keys) is
IDENTICAL across the answer_suite.{raw,star,metrics} family; only the grounding block differs,
so accuracy differences are attributable to grounding alone (the aah replication design).

## When to use
Only for aah grounding-axis runs, pinned by a spec. Never Router-matched.
