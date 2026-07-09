# GADS Research Journal

**Mission.** GADS is a research platform for investigating the **efficiency boundary of agentic
data science as a function of the power of the LLM engine**. The two target metrics are:

1. **Reproducibility** — consecutive runs of the exact same spec must produce identical results.
2. **Methodological appropriateness** — for any GADS project there exists a latent, correct plan
   (whether codified in a recipe or not); appropriateness is therefore, at least in theory,
   verifiable against that latent plan.

This journal collects observations that speak to these objectives, for later use in report
write-ups. Companion assets: `research/benchmarks/` (specs with expected results for
quantifying run quality across modes) and `approach_docs/011` (techniques report).

**Entry conventions.** Newest first. Tags: `[repro]` reproducibility, `[method]` methodological
appropriateness, `[boundary]` capability-boundary evidence, `[harness]` instrumentation that
enables measurement. Every entry cites evidence: project UUIDs, commits, file paths. Claims
without run evidence are marked *(hypothesis)*.

---

## 2026-07-09 — `[repro]` `[method]` Wall-clock AutoML budgets break reproducibility with *identical* generated code

Same spec, same recipe, **same generated code shape** (verified by diffing
`workflow_execution.py`: identical `sample(50000, random_state=42)`, identical stratified
`train_test_split(..., random_state=42)`, identical `fit(time_limit=120, presets='good_quality')`),
yet:

| Run | Date | Coder model | test_score (ROC-AUC) |
|---|---|---|---|
| `faa3870e` (cloud) | 07-07 | claude-sonnet-4.6 | **0.9692712906057946** |
| `e9ecf496` (local) | 07-08 | gemma-4-12b | 0.9570593538427091 |
| `3fcb048a` (hybrid) | 07-08 | gemma-4-12b | 0.9570593538427091 |
| `151a323d` (cloud_pinned) | 07-08 | claude-haiku-4.5 | 0.9570593538427091 |

The divergent run's leaderboard reached `WeightedEnsemble_L3`; the others plausibly stopped at a
shallower stack. `time_limit=120` is a **wall-clock** budget: how many models AutoGluon trains
within it depends on machine load (the 07-08 runs shared the box with LM Studio inference) and
hardware state. The nondeterminism was injected by the *recipe's own invariant*, not by the LLM.

**Implications.** (a) Reproducibility audits must separate LLM-induced variance from
*methodology-induced* variance — a recipe can be methodologically reasonable yet
reproducibility-hostile. (b) For benchmark recipes, replace time budgets with fixed model
portfolios / explicit `hyperparameters` dicts, or record the leaderboard composition as an
expected artifact. (c) The benchmark scorer uses a tolerance band for `test_score` until the
recipe is fixed; `naive_baseline` (pure data function) is expected to match exactly.

## 2026-07-08 — `[repro]` Bitwise cross-model reproducibility under recipe constraint

Three runs with **three different coder configurations** — gemma-4-12b (local), gemma-4-12b
(hybrid), claude-haiku-4.5 (cloud_pinned) — produced bitwise-identical
`test_score = 0.9570593538427091` and `naive_baseline = 0.99834` (runs `e9ecf496`, `3fcb048a`,
`151a323d`). When the recipe pins the code patterns (seeded sampling, seeded split, fixed fit
config) and generation temperature is low (0.1 local), *different models converge to the same
realization*: the code differs cosmetically but is functionally identical.

**Implication.** Recipe-constrained code generation is the reproducibility mechanism — the
stronger the recipe's grip on the methodology, the less the engine's identity matters for the
*result* (engine power then only affects success *rate* and *cost*, which is precisely the
efficiency-boundary framing). Reproducibility of a GADS run is inherited from the recipe, not
from the model.

## 2026-07-08 — `[repro]` Failures are reproducible too (and that makes them benchmarkable)

- claude-haiku-4.5 generated the **identical** `Length mismatch: Expected axis has 7 elements,
  new values have 2 elements` bug in the Extract task on workflow attempts 2 and 3 of run
  `151a323d` (it had *passed* the same task on attempt 1 — the leaderboard DataFrame it
  mangled only exists with certain column counts, i.e. state-dependent determinism).
- gemma-4-12b at temperature 0.1 produced near-identical generations across retries: in runs
  #3–#5 the same fence/truncation/empty-code failures recurred across all 3 inner retries and
  across workflow attempts.

**Implication.** At low temperature, a model×prompt×state triple has an approximately
deterministic outcome — so benchmark runs measure a *property of the configuration*, not a dice
roll. Aggregate success rates across n runs remain worth recording, but n can be small.

## 2026-07-08 — `[method]` `[repro]` Verifier-triggered replans on compiled plans: pure re-execution, artifact-overwrite hazard

In run `151a323d` all three tasks passed on attempt 1, yet the CompletenessVerifier flagged gaps
(it could not see the feature-importance evidence in the task summaries) and triggered a replan.
On a compiled plan a replan **recompiles the identical plan**, so attempts 2–3 re-executed the
same tasks: wasted compute, and each re-run **overwrites attempt 1's artifacts**
(`model.joblib`, `metrics.json` re-saved; figures survived only because the re-run Extract
failed). With a nondeterministic recipe step, a *correct* attempt-1 result could be silently
replaced by a different attempt-2 result — a reproducibility hazard inside a single project.

**Fix directions:** skip verifier-triggered replans for compiled plans (mirror the PlanCritique
skip), and/or feed emitted insights into verifier evidence; version artifacts per attempt.

## 2026-07-08 — `[boundary]` The local capability boundary is planning, not realization

Six-run campaign, one spec (fraud/AutoGluon), gemma-4-12b: with LLM front-end (Router+Planner)
the run died at those stages (runs `68ce5e0b`, `f5e332e8` — degeneration/truncation of
structured JSON). With the deterministic front-end + raw-code Coder, the same model executed
all three tasks first-shot with zero escalations (`e9ecf496`). Detail in
`approach_docs/011` §5. The boundary statement: **a 12B-class engine is a competent code
realizer and a hopeless project manager** — open-ended decomposition, classification with
calibrated confidence, and large structured output sit beyond the boundary; one-shot realization
of a precise intent sits within it.

**Efficiency-boundary corollary.** Engine power buys the right to *delegate more decisions to
runtime*. Weak engines need the latent plan compiled a-priori (recipes); strong engines can
recover it at runtime (drafted plans). The interesting measurable quantity is the *gap* between
a mode's output and the recipe's latent plan, as a function of engine tier — the benchmark repo
exists to quantify exactly this.

## 2026-07-07 — `[method]` Confident misclassification: method selection cannot be delegated on confidence

The Router matched the fraud objective (explicitly: supervised AutoML, held-out ROC-AUC) to the
*unsupervised* Isolation Forest recipe at **0.95 confidence** with fluent, wrong reasoning
(cloud run, pre-fix; the local Router was mid-way to the same choice before dying). The Recipe
Enforcer then re-imposed the wrong methodology on every replan — a methodological error made
*deterministic* by the harness. Fixed by spec recipe pinning (`0f0d181`): the spec is the
methodological ground truth; LLM classification is advisory.

**Implication for verification.** "Methodological appropriateness" cannot be self-assessed by
the executing system at any confidence level; it needs an external referent — the pinned recipe
today, the benchmark's expected plan tomorrow.

## 2026-07-07/08 — `[harness]` Instrumentation that makes the boundary measurable

- Every generation carries `completion_path` (`first_shot | stream_repair | manual_fallback |
  raw_code`) — first-shot success vs. repaired success is the engine-power signal, invisible in
  outcome-only metrics (plan 010).
- Traces are tagged `routing_mode:{cloud|local|hybrid|cloud_pinned}` + `recipe:{id}` +
  `outcome:{...}` — run quality is joinable to engine configuration.
- `metrics.json` is the canonical scalar ground truth per run; contracts check the live kernel,
  never model claims.
- Known gap: metric-contract failures burn a workflow attempt without reaching the Coder as
  retry feedback (board P1) — this conflates "engine can't do it" with "engine was never told",
  which pollutes boundary measurements.

---

*Add entries above this line. Keep the evidence discipline: UUID or it didn't happen.*
