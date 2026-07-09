# External DS-Agent Benchmarks: Landscape & GADS Adaptation Plan

**Date:** 2026-07-09 · **Status:** survey + phased adaptation proposal
**Purpose:** identify existing data-science benchmarks whose tasks and ground truths can be
converted into GADS benchmark specs (`research/benchmarks/`), so run quality across routing
modes is measured against *externally validated* expectations rather than only our own
reference runs.

## What "fits GADS" means

A candidate benchmark adapts well if it satisfies most of:

1. **Project-shaped tasks** — an objective over one or more data files, requiring a multi-step
   workflow (not single-line code completion). GADS runs specs, not exercises.
2. **Closed-form or decision-level ground truth** — either numeric expected results
   (→ `expected.json.metrics`) or expert-verified *analysis decisions*
   (→ `expected.json.methodology`), matching our two research metrics.
3. **Sandbox-compatible** — CSV/Parquet tabular or text data; solvable with the fixed package
   set (pandas/sklearn/AutoGluon/statsmodels/dowhy/pymc…); no internet at runtime; dataset
   sizes that fit `GADS_DATASETS_ROOT` and the row-cap budget.
4. **Local-tier feasible** — the latent plan is codifiable as a recipe, so weak engines are
   measured on realization, not on surviving open-ended planning (unless deliberately testing
   lane C).
5. **Redistributable data/answers** (license) — we copy datasets into workspaces.

## Survey

| Benchmark | Shape & ground truth | GADS fit | Notes |
|---|---|---|---|
| **AMLB / OpenML AutoML Benchmark** ([JMLR](https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf), [docs](https://openml.github.io/automlbenchmark/docs/extending/benchmark/)) | 104 tabular tasks (71 classification, 33 regression), 10-fold CV, **published per-framework reference results incl. AutoGluon** | ★★★★★ | The natural extension of `fraud_autogluon_v1`. OpenML datasets are freely downloadable; AMLB's AutoGluon scores give externally-validated expected metrics (with documented CV-vs-holdout caveat). Recipe already exists. |
| **BLADE** ([EMNLP'24](https://aclanthology.org/2024.findings-emnlp.815/), [repo](https://github.com/behavioral-data/BLADE), [site](https://blade-bench.github.io/)) | 12 datasets + research questions from real papers; **~500 expert-verified analysis decisions** (variables, transforms, model choices) from independent expert analyses | ★★★★★ | The *only* benchmark whose ground truth is literally our "methodological appropriateness" metric — the latent correct plan, empirically enumerated as the space of justifiable decisions. Decisions → `methodology.required/forbidden_patterns` (regex subset now; decision-matching scorer later). May need recipe extensions (mixed models via statsmodels). |
| **QRData** ([used with](https://arxiv.org/pdf/2602.20571)) | Quantitative-reasoning questions over data files incl. **causal estimation with numeric gold answers** | ★★★★ | Direct fit for the causal recipes (`causal_effect.observational.dowhy` etc.): gold ATE-style numerics → `metrics` with tolerances. Textbook-clean identification, good first causal benchmark tier. |
| **StatQA** ([used with](https://arxiv.org/pdf/2602.20571)) | 11,623 tasks over 5 categories: descriptive stats, correlation, contingency tests, distribution tests, variance tests; gold = correct test + answer | ★★★★ | Ground truth includes **which statistical test is appropriate** — appropriateness of *method selection*, scoreable. Wants the hypothesis-testing recipe (approach doc 005) built out; then a generator can mass-produce specs. |
| **InfiAgent-DABench / DAEval** ([paper](https://arxiv.org/abs/2401.05507), [site](https://infiagent.github.io/)) | 311 public closed-form questions over 55 CSVs (validation split public by design) | ★★★ | Cheap breadth: closed-form answers → exact `metrics`. But single-question tasks are shallower than GADS projects — best used as a smoke-coverage pack (bundle several questions per CSV into one spec). Verify license before ingesting. |
| **DSBench** ([survey ref](https://arxiv.org/pdf/2508.02744)) | 540 tasks from ModelOff + Kaggle: analysis + modeling with ground truth | ★★★ | Modeling subset fits; ModelOff analysis tasks are Excel-workbook-shaped — poor sandbox fit. Cherry-pick the tabular modeling tasks. |
| **MLE-bench** ([paper](https://arxiv.org/pdf/2410.07095)) | 75 full Kaggle competitions, graded vs. human leaderboards | ★★ | Prestige benchmark but heavy: large datasets, deep-learning-dominated, long budgets — hostile to the local tier and the sandbox. Consider 2–3 tabular "Lite" competitions later, mostly for cloud-mode headroom measurement. |
| **DABstep** ([HF blog](https://huggingface.co/blog/dabstep), [paper](https://arxiv.org/html/2506.23719v1)) | 450 real-world financial-analytics tasks requiring **cross-referencing heterogeneous documentation** | ★★ | Multi-step reasoning fit is good, but tasks depend on doc-retrieval over policy manuals — GADS has no document-context stage. Revisit if/when a knowledge-docs stage exists. |
| **DiscoveryBench** ([paper](https://arxiv.org/html/2407.01725v1)) | Open-ended discovery tasks from published work with ground-truth hypotheses | ★★ | Aligns with the latent-plan philosophy but open-ended answer matching makes automatic scoring hard; better as a later, LLM-judged tier. |
| **DataSciBench** ([THUDM](https://github.com/THUDM/DataSciBench), [paper](https://arxiv.org/pdf/2502.13897)) / **AgenticDataBench** ([2026](https://letsdatascience.com/news/researchers-release-agenticdatabench-for-llm-data-agents-f3a2fd61)) / **DSAEval** ([2026](https://arxiv.org/html/2601.13591v2)) | Workflow-level agent benchmarks with multidimensional metrics | ★★★ | Newer, workflow-shaped; worth mining for *metric definitions* (DataSciBench's 25 metrics) and task sourcing once repos are vetted. |
| **AIRepr** ([paper](https://arxiv.org/pdf/2502.16395)) | Analyst–inspector *framework* for evaluating LLM reproducibility in DS | n/a (method, not tasks) | Borrow methodology: their reproducibility operationalization is directly citable for our metric definitions in write-ups. |

## Phased adaptation plan

**Phase 1 — extend what already works (low effort, this sprint-ish):**
1. `amlb_<task>_v1` × 3–5: pick OpenML tasks spanning binary / multiclass / regression /
   severe-imbalance (e.g. from the AMLB 104), download once into `GADS_DATASETS_ROOT`,
   generate specs pinning `tabular_automl.autogluon.standard`. Expected metrics from our own
   reference runs, with AMLB's published AutoGluon results recorded in `notes.md` as the
   external sanity anchor (holdout-vs-CV difference documented).
   *Prerequisite from journal 2026-07-09: recipe v2 with fixed model portfolio (no wall-clock
   `time_limit`) so `test_score` can be `exact: true`.*
2. `qrdata_causal_<n>_v1` × 5: QRData causal questions with numeric gold answers → specs
   pinning the dowhy recipe; gold values → `metrics` with stated tolerances. Exercises the
   causal stack that currently has no benchmark coverage.

**Phase 2 — the methodological-appropriateness flagship:**
3. `blade_<dataset>_v1` × 2–3: convert BLADE research questions to specs; encode the
   expert-decision ground truth as `methodology` checks (regex tier now: required variable
   names, transformation calls, model families; forbidden shortcuts). Add a
   `decision-matching` scorer mode later that parses `workflow_execution.py` AST against
   BLADE's decision schema. This is the benchmark that tests whether GADS recipes encode
   *defensible* methodology, not just reproducible methodology.

**Phase 3 — breadth and generators:**
4. DAEval smoke pack (license permitting): bundle 3–5 closed-form questions per CSV into one
   spec; exact-match metrics. Cheap mode-comparison fodder.
5. StatQA → requires building the hypothesis-testing recipe first (approach doc 005); then a
   spec generator over its 5 categories, scoring *test selection* as methodology.

**Explicitly deferred:** MLE-bench (compute-hostile), DABstep (needs doc-retrieval stage),
DiscoveryBench (open-ended scoring).

## Design notes for adapters

- **One benchmark = one frozen folder** per `research/benchmarks/README.md`; external
  provenance (source benchmark, task id, license) goes in `notes.md`.
- **External gold vs. GADS canonical:** record both. `metrics.value` = our verified reference
  run (what reproducibility is measured against); `notes.md` carries the external benchmark's
  expected value (what appropriateness is anchored to). Divergence between the two is itself a
  journal-worthy observation.
- **Lane discipline:** Phase 1–2 benchmarks pin recipes (measure realization). For boundary
  experiments, the *same* spec can be run with `disable_recipes` to measure lane-C planning —
  the scorer doesn't care how the plan was made.
- **Licensing:** OpenML (mostly CC/public domain per-dataset — check each), BLADE (MIT repo;
  datasets from published papers — verify per-dataset), QRData/StatQA/DAEval (verify repo
  licenses before committing data; store datasets in `GADS_DATASETS_ROOT`, never in git).
