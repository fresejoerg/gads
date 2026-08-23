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

## 2026-07-15 — `[boundary]` `[repro]` `[method]` QRData suite on the local engine: 4/5, and where engine variance actually lives

Full local-mode (gemma-4-12b) sweep of the five QRData causal benchmarks — the first
qrdata × local cells. Grid after the batch: **D4×local 6/7 (86%), D4×cloud 5/5, D1×cloud 1/1.**

| Benchmark | run | local ATE | cloud ref | agreement |
|---|---|---|---|---|
| ihdp_0 | `6e4591f6` | 3.896514 | 3.896514 | ~1e-15 (same top-10 set) |
| ihdp_1 | `3fbfec1a` | 3.899352 | 3.852051 | 1.2% (different sets) |
| collections_email | `fc101698` | 4.430356 | 4.430356 | ~1e-13 (same set; mediator trap held) |
| online_classroom | `c7b0678e` | **FAIL 11/14** | -4.212521 | — |
| hospital_treatment | `94f77f10` | -7.591172 | -7.591172 | **bitwise** (same set) |

**Where engine variance enters is now precisely located.** The D5 native node contributed
*zero* variance: given the same adjustment set, engines agree to at worst the last ulp
(bitwise on hospital_treatment; sandbox counterfactual on ihdp_0: top-10 → 3.8965142215032174,
all-25 → 3.9286717508727147). All observed variance lives in the D4 role-assignment task:
on ihdp_1 the local engine passed all 25 covariates, skipping the "at most 10" cap it had
correctly applied on ihdp_0 in the same session — within-engine, run-to-run variance in
instruction-following, invisible to outcome metrics when tolerances absorb it.

**The mediator trap held at the local tier** (`fc101698`): the 12B engine excluded
`opened`/`agreement` as post-treatment and selected exactly `['credit_limit','risk_score']`,
landing ~1e-13 from the cloud reference. The re-layered recipe's role contract works at
both engine tiers.

**The one failure is a clean boundary observation** (`c7b0678e`, online_classroom): the
dataset has NaNs in 76/323 rows of exactly the demographic-dummy confounders; statsmodels
raises `exog contains inf or nans` inside the native node. Cloud had preemptively cleaned
(`replace([inf,-inf],nan).dropna(...)`); the local engine emitted byte-identical code across
all three workflow attempts *with the error in its retry context*. Error → diagnosis →
insert-a-cleaning-step is beyond the 12B realizer; one-shot realization of the happy path
is not. Fix direction (deliberate, not applied): harden `gads_causal_estimate_ate` to drop
NaN/inf rows over used columns internally — matches cloud behavior, leaves NaN-free
canonicals unchanged. Applying it moves the step from D4-recoverable to D5-mechanized —
the delegation dial turned one notch by evidence.

**Scorer blind spots, confirmed twice:** (1) the failed run still passed the methodology
regex tier 8/8 — "right code that never produced numbers" is indistinguishable from a
working run; (2) the ihdp_1 adjustment-set divergence passes unnoticed inside the 2%
tolerance. Both are exactly what the Phase-2 decision-matching scorer (BLADE) is for.

**Ledger hygiene:** a first launch of ihdp_0 (`5808672b`) died to the sandbox
workspace-mount regression (WSL UNC bind; /execute 500 PermissionError) — annotated
`excluded: true` in the ledger (infrastructure, not engine evidence; `dial_heatmap.py`
now skips excluded records). Also recurred today: GADS tables vanished from Postgres on
stack restart (recreated via `init_db`).

## 2026-07-14 — `[method]` `[repro]` QRData causal benchmarks: externally-anchored golds, and the mediator trap held

Five QRData questions are now GADS benchmarks (`qrdata_*_v1`) — the first with **externally
defined** gold answers (published ATEs), closing the self-referentiality gap of the AMLB
family. All five cloud runs (recipe `causal_effect.observational.dowhy` v2.1.0, rung D4)
passed 16/16, with refuters clean (placebo ≈ 0, subset ≈ ATE):

| Benchmark | run | GADS ATE | gold | Δ |
|---|---|---|---|---|
| ihdp_0 | `3f6f83d5` | 3.8965 | 4.02 | −0.12 |
| ihdp_1 | `7d07a611` | 3.8521 | 4.05 | −0.20 |
| collections_email | `1e601199` | **4.4304** | 4.43 | **+0.0004** |
| online_classroom | `31891ed3` | −4.2125 | −4.91 | +0.70 |
| hospital_treatment | `21d21c3f` | **−7.5912** | −7.59 | **−0.0012** |

**The mediator trap held.** collections_email contains two post-treatment variables
(`opened`, `agreement`) that the pre-refactor confounder heuristic would have adjusted
for, biasing the answer. Under the re-layered recipe, the role-assignment task explicitly
selected `['credit_limit', 'risk_score']` and named the exclusions as post-treatment (run
`1e601199` stdout) — and landed within 0.0004 of the gold. The 2026-07-13 recipe fix is
validated by exactly the failure it was designed to prevent.

**The one visible gap** — online_classroom (Δ 0.70, ~14%) — is an adjustment-set
difference, not a malfunction: the gold comes from OLS with the full demographic dummy
set; the recipe's fallback caps/selects confounders. Direction and rough magnitude agree;
the benchmark records our canonical value with the gold as anchor. *(hypothesis: exact
match achievable if the objective enumerates the adjustment set, as collections_email
does — untested.)*

**Ledger-rule amendment:** run `7d07a611`'s only failure was the fail-open
CompletenessVerifier crashing — advisory by design, so it no longer counts against the
dial outcome (rule amended in server.py; the run's record backfilled with a note).

**First drafted-lane cell (D1 × cloud):** amlb_segment with recipes disabled, run
`6beeea9f` — the LLM Planner decomposed into **8 tasks** (vs the recipe's 3), chose
AutoGluon on its own, and scored macro-F1 **0.9719**, *above* the deterministic recipe's
0.9281 — while emitting a differently-named metric (`test_macro_f1` vs `test_score`) and
a protocol that is non-reproducible by construction (time-budgeted). The efficiency
boundary in one observation: at D1 a frontier engine completes and can even outperform
the pinned methodology, but its output is contract-incomparable and unrepeatable —
higher score, lower evidential value. Grid after the batch: D4×local 2/2, D4×cloud 5/5,
D1×cloud 1/1.

## 2026-07-14 — `[harness]` `[boundary]` The delegation dial is now measured: rung × engine evidence grid

The efficiency-boundary question is now an accumulating dataset. Every run is placed on
the **delegation ladder** (approach_docs/013): D0 full delegation → D1 framed (spec
hints) → D3 directed (compiled plan + invariants) → D4 patterned (+ curated skills) →
D5 mechanized (native step functions). Project rung = **minimum over task rungs**
(the weakest link gets the most freedom). D2 "advised" is reserved but operationally
unreachable — a Router-matched recipe compiles exactly like a pinned one, so selection
provenance (`pinned|routed|drafted`) is recorded as a separate field.

Instrumentation (`core/dial.py`): rung computed at plan construction, logged and stored
on the project; at run end a record lands in **`research/dial_ledger.jsonl`** (rung,
task rungs, selection, routing_mode, outcome — pass requires approved synthesis AND
zero failed tasks). `scripts/dial_rung.py` determines any spec's rung statically;
`scripts/dial_heatmap.py` renders the rung × engine grid. Current spec library: all
pinned specs are D4 projects (the causal spec's estimate node is individually D5);
unpinned test specs are D0/D1.

Verified with runs `594fa3cb` and `42d3b901` (amlb_segment, local): both PASS, ledger
records correct, and both reproduced `test_score = 0.9281148405938683` — the third and
fourth consecutive bitwise reproductions of that benchmark on the local engine. Grid
after seeding: **D4 × local = 2/2**.

Found and fixed along the way: kernel-state snapshots were **full-replacing**
`project.last_state_json` after every completed task, clobbering project metadata —
including the pre-existing `spec_filename` (the retroactive spec-matching hack in
`list_projects` existed because of this). Snapshots now merge, preserving metadata keys.

**Interpretation discipline:** cells are observational, not randomized — specs are not
randomly assigned to rungs or engines. The grid accumulates evidence for D\*(E) (the
lowest rung where engine E stays within tolerance); it does not estimate causal effects
of rung on outcome. Known coarseness: skills on drafted-plan tasks do not lift the rung
above D1 (methodology freedom dominates realization guidance in v1).

## 2026-07-13 — `[harness]` Semantic skill retrieval (embedding-based), gated to uncurated tasks

Skills can now be found semantically, not just by keyword triggers: fastembed (ONNX,
CPU-local, deterministic) runs `all-MiniLM-L6-v2` — the same model family the sandbox uses —
over lean skill cards (id + description + triggers). Wired into the compiler fallback and the
executor (`core/skill_semantics.py`; debug endpoint `GET /skills/match?query=…`).

**Reproducibility gate:** semantic discovery fires ONLY for tasks with no curated
`attached_skills` (drafted-lane tasks, recipe nodes that declare none). Curated tasks —
including every frozen-benchmark node — keep byte-stable prompts.

**Calibration findings (2026-07-13):**
- Absolute cosine values are not comparable across queries with this model (a true match
  scored 0.27 on one query, 0.54 on another; a false positive hit 0.31). Selection therefore
  uses a per-query z-score (skill must be an outlier ≥2.0 among all 15 skill scores) AND an
  absolute floor (0.25), capped at top-3.
- Card content matters: including section headings in the embedded text *degraded* ranking
  (top-3 containment 2/5 → 5/5 after dropping them — keyword-dense soup dilutes the vector).
- Technical Planner-style phrasings select cleanly, including two cases keyword triggers miss
  entirely; colloquial paraphrases may return [] — the safe failure (keyword matching still
  applies). Precision was chosen over recall: a wrongly attached skill costs local-model
  context budget on every retry.

## 2026-07-13 — `[method]` `[repro]` `[harness]` Recipe/skill re-layering: bitwise reproducibility survives decision-level prompts

The recipe library was refactored from verbatim code dictation to a three-layer split
(`approach_docs/012`): methodology decisions stay in recipes, reusable code-hints move to
skills (injected deterministically via `attached_skills`), environment facts live in the
force-loaded `sandbox_environment` skill. All 14 recipes rewritten; dataset residue removed
(Kaggle section, Titanic, `'<=50K'` examples); the DoWhy/Bayesian confounder rule now treats
objective-named roles as authoritative and **excludes post-treatment variables** (the old
all-numeric heuristic would have conditioned on mediators — e.g. QRData collections_email's
`opened`/`agreement` — guaranteeing a biased ATE on that upcoming benchmark).

**Verification (amlb_segment, local/gemma-4-12b):** run `523bd7eb` — 3/3 tasks, 0 escalations,
scored **PASS 20/20** with `test_score = 0.9281148405938683`, bitwise-identical to the cloud
(`11f012e1`) and pre-refactor local (`f4e64460`) reference runs, under `exact: true`.
**The determinism never lived in the verbatim intents** — it lives in the invariants (fixed
portfolio, seeded split) plus skill patterns; a 12B model realizes decision-level intents into
metric-identical code.

**The two failed attempts on the way were both harness, not model:**
1. Run 1 (invalid): a stale backend predating the rewrite still held the old registry in
   memory — my restart had silently died on the occupied port, and WSL clock skew made the
   zombie's `lstart` look fresh. In-memory registry state is invisible; compiled task text is
   the only trustworthy witness of what the workflow actually saw.
2. Runs `835cf36c`/`590c6055`: the Extract task failed with `SyntaxError: keyword argument
   repeated: num_shuffle_sets` — **not** gemma's doing. The executor's feature_importance
   timeout sanitizer checked only for `subsample_size` before appending
   `subsample_size=1000, num_shuffle_sets=1`, corrupting the skill's already-guarded call.
   Gemma had followed the new pattern correctly in every attempt. Fixed in `executor.py`
   (inject only when the call is completely unguarded).

**Lesson for boundary attribution:** before charging a failure to the engine, rule out the
harness — here, two consecutive "local-model failures" were a stale process and a sanitizer
regex. Layered guidance also needs cross-auditing: deterministic code-mutating barriers
(sanitizers) must be checked against the skill patterns they may collide with.

## 2026-07-10 — `[method]` First new benchmark immediately exposed a latent recipe defect (label-dtype)

The very first run of `amlb_adult` (project `52406af9`, cloud) failed its Extract task with
`Labels in y_true and y_pred should be of the same type: y_true=['<=50K' '>50K'], y_pred=[0 1]`.
Root cause: **two layers** of the AutoGluon recipes assumed binary targets are 0/1 ints —
the recipe's calibration pattern (`(y_prob.iloc[:,1] >= t).astype(int)`) *and* the
`gads_calibrate_threshold` native helper (int preds vs. raw y_true inside f1_score). The fraud
benchmark never caught it because `Class` is already 0/1; adult's string labels did, on the
first attempt, deterministically (the recipe mandates the exact pattern — all replans would
have failed identically; run cancelled after attempt 1 confirmed it).

Fixes: recipe pattern now maps thresholded booleans back to `y_prob.columns` (the class
labels); native helper binarizes non-{0,1} targets against the lexicographically-last class
(matching AutoGluon's proba column order). Both recipe variants patched.

**Implication.** A recipe verified on one dataset encodes that dataset's incidental properties
as silent assumptions. Benchmark *diversity* is what converts "methodologically appropriate on
fraud" into "methodologically appropriate for the recipe's declared applicability domain" —
one new dataset falsified an invariant assumption within minutes. Corollary for the flywheel:
distilled invariants must state their preconditions, or diverse benchmarks will break them.

## 2026-07-10 — `[harness]` Benchmark infrastructure stood up; engine fleet change (gpt-5.6)

- **Benchmark repo + scorer live** (`be0079d`): `research/benchmarks/` format (metrics with
  exact/tolerance semantics, artifact manifest, methodology regexes) + `scripts/score_benchmark.py`.
  First benchmark `fraud_autogluon_v1` seeded from the four verified reference runs; scorer
  validated (local reference 18/18 PASS; divergent cloud reference passes within the documented
  `time_limit` tolerance).
- **External landscape surveyed** (`47e9012`, `research/benchmark_landscape.md`): AMLB and BLADE
  are the priority adaptations (external reference results / expert-verified analysis decisions
  respectively); QRData → causal recipes; StatQA → hypothesis-testing recipe. Phase 1 = AMLB
  slice + QRData causal.
- **Engine fleet change:** OpenAI GPT-5.6 suite (sol/terra/luna) went GA 2026-07-08/09; this
  account's rollout hadn't landed as of 07-10 (IDs 404 upstream). Ladder now carries 5.6 as
  primary OpenAI slots with 5.4/5.5 same-tier fallbacks — escalation absorbs the interim 404s
  and self-heals. **Comparability rule this implies:** benchmark records must capture the
  *actual serving model* per generation (they do — `model_used` per task + LiteLLM logs), not
  the tier label; cross-time comparisons on "T2" are meaningless across fleet changes.

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

## 2026-07-16/20 — `[method]` The local delegation-dial column: a non-monotonic valley at D3

The QRData causal suite (5 benchmarks) was swept across the delegation dial on the local engine
(gemma-4-12b-qat, `routing_mode:local`), completing a 20-run column. Pass = approved synthesis
AND zero failed tasks (note: this criterion does **not** audit methodology — see the D0 caveat
below). Result by rung:

| Rung | Realization channel | QRData causal pass |
|---|---|---|
| **D5** mechanized | native `gads_causal_estimate_ate` node | **4/5** (only `online_classroom` fails) |
| **D4** curated skill | worked-example DoWhy skill | ~4/5 (`ihdp_0` flaky: `6e4591f6` pass, `5808672b` fail) |
| **D3** directed (dowhy) | model writes DoWhy code, no skill | **0/5** |
| **D1** framed drafted | LLM-drafted plan + spec hints | **1/5** (only `ihdp_1` `f1efdcfa`) |
| **D0** full delegation | LLM-drafted plan, bare objective | **1/5** (only `ihdp_0` `2a5e7e1b`) |

**The signature is non-monotonic in structure.** The *most*-structured free-code rung, D3-directed,
is the single **worst** performer (0/5) — worse than the *less*-structured D0/D1 drafted lanes.
The reason is specific and uniform: D3 fixes the methodology to the **DoWhy** four-step API, and
gemma cannot write it — every D3 run burned all 3 attempts in ~5 min on hallucinated modules
(`dowhy.ifm`, `dowhy.models`) and wrong signatures. This is an **API-knowledge wall, not a
planning or reasoning failure**: the plan is compiled at D3, and the same engine interprets D4/D5
results acceptably. **D\*(gemma-4-12b) = D4** for this task family — the lowest rung it clears is
the one that hands it worked code patterns.

**D0 "passes by doing less" — the pass criterion is methodology-blind.** `ihdp_0` D0 (`2a5e7e1b`)
passes while its more-guided D1 (`1464af0b`) fails. The D0 plan emitted a bare ATE with no
refuters; the ledger's pass gate (synthesis approved + zero failed tasks) never checks that the
refutation checks the recipe mandates were actually run. A pass that gets there by omitting the
hard part is the inverse of methodological appropriateness — the "external referent" problem from
the 2026-07-07 entry, now visible from below. Consequence for 014: grade ATE fidelity and
failure-class, never pass/fail alone.

**`online_classroom` is a rung-invariant covariate, not signal.** It fails at D5/D4/D3/D1/D0 — a
NaN data-cleaning defect independent of the dial. Effective ceiling for the suite is 4/5; treat
its cell as N/A when reading the column.

**Harness confound caught and fixed — Router token starvation.** Two `collections_email` D1 runs
(`c49ade32`, then `9037eea4`) halted fatally with `[HALTED] Fatal system error`. Root cause was
**not** the engine: `router.py` capped `max_tokens` at 1024, but gemma-4-12b-qat is a reasoning
model whose `reasoning_content` counts against the completion budget — on a dense drafted-lane
objective it burned the whole 1024 on CoT (`reasoning_tokens=1021`) and emitted an empty answer,
`finish_reason='length'`. Raised the cap to 4096 (`router.py:~35`; fits reasoning + the ~200-token
JSON while still failing fast on a true runaway). **Monotonically safe:** it cannot change any run
whose reasoning already fit under 1024, so the two already-ledgered D1 passes stay valid. Relaunch
`666701bd` cleared the halt and produced a genuine fail. This is the 014 threat-to-validity in the
flesh: harness confounds masquerade as engine failures — grade the failure class before reading any
cell.

**Operational note — DB loss on stack restart.** The MyLocalStack restart between 07-17 and 07-20
dropped the GADS tables (`project`/`task`/… vanish, ledger JSONL survives as a file), taking the
first `online_classroom` D0 attempt (`f1f15912`, unledgered) with it. Recovered via `init_db()` +
clean backend restart; re-ran as `036fbc68` (fail, genuine — real code errors: `'DataFrame' object
has no attribute 'schema'`, unclosed brackets). Final D0 run `hospital_treatment` (`4114bb77`, fail,
genuine syntax errors) closed the column.

**Reading.** The dial has a **valley**: high at D5/D4 (structure supplies the API), collapses at
D3 (niche API, no scaffold), recovers only marginally at D0/D1 (drafted lane occasionally passes
by simplifying the task away). The valley's floor is an API-familiarity artifact, which is exactly
the hypothesis agenda `approach_docs/014` isolates — same estimand, same golds, same D3 rung, but
swapping DoWhy → statsmodels → sklearn (Phase A recipes/specs built, not yet run). External
corroboration: decisionspine's `ai-analytics-harness` (2026-07-20 review) independently found
(a) a small model converging with a flagship once the right structure is present, and (b) prose
guidance that never improves accuracy while pattern-matchable *examples* do — both directly
predict the 014 outcome (worked examples, not bare signatures, are the active ingredient of D4).

## 2026-07-22 — `[method]` The grounding axis: replicating aah inside GADS (slice validated)

Translated decisionspine's `ai-analytics-harness` (aah) — a 25-question business-analytics
benchmark over a deterministic warehouse — into GADS as the **grounding axis**, orthogonal to the
delegation dial (approach_docs/015). aah varies *grounding of the inputs* (raw → star → semantic →
examples → KB → tree) with the agent fixed; GADS varies *delegation of the method*. The 2-D grid
crosses them.

**Design correction mid-build.** First attempt put rungs 1–2 on the drafted lane (dial-D0) — but the
12B couldn't even write the 25-question answer-harness there (it assumed `aah_questions.json` handed
it pre-written SQL: `q_data.get('sql')` on a string → uniform failure). That's a *delegation-floor*
confound, not a grounding effect. Fix (with Jörg): a **constant harness** — one recipe family
`analytics.answer_suite.{raw,star,metrics}` with identical loop scaffolding (load questions →
per-question try/except → always-write, exact-id keys), all pinned at D3, grounding the only
variable. This is aah's actual design (fixed agent, grounding varies).

**Result (gemma-4-12b-qat, local, one run/rung):**

| rung | grounding | accuracy | metric tier |
|---|---|---|---|
| 1 | messy raw | **0%** | 0/5 |
| 2 | + star | **8%** | 0/5 |
| 3 | + governed metrics | **28%** | **5/5 (100%)** |

**The shape replicates aah** (flagship 41→46→66): monotone rise, the **semantic layer is the single
biggest step** (+20 pts, aah's 46→66 signature), and the **metric tier hits 100% only with governed
definitions** — deterministically (MRR 2685.08, active-users-last-week 886, power-users 151,
activation 0.53, all validated pre-run without an LLM). The *absolute* gap is the local-model story:
gemma scores **0% on raw data** (vs 41% flagship / 22% mini) — it can't wrangle the messy schema at
all, degenerating to "2500 records" (the raw user-table count) for most questions. Grounding matters
*more* for a weak engine, but the raw floor is unusable.

**Iteration trail (each ~5 min on local):** import ergonomics (`from aah_metrics import layer`) →
point-in-time crash (a period on ARPU aborting the whole task) → per-question robustness → answer
re-keying (q1..q25 vs benchmark ids) → mapping/punting. Each was a real recipe/module defect the
live 12B surfaced; the module was made forgiving (auto-built layer, lenient periods) and the recipe
directive (exact ids, never-punt, apply-period, mapping table). The metric-tier 100% is the robust
signal; lookup/filtered/knowledge tiers vary run-to-run (single-seed; aah uses 5 reps/cell).

Artifacts: `research/harness/aah/` (MIT provenance), `research/benchmarks/aah_v1/` (25 golds +
results.jsonl + scorer `scripts/grade_aah.py`), recipe family + `aah_star_schema` skill, 3 rung
specs, datasets exported to `$GADS_DATASETS_ROOT/aah/`. Rungs 4–6 and reps pending.

## 2026-07-30 — `[survey]` External corroboration: Stripe's Kai knowledge-AI platform

Read Stripe's "Meet Kai" post (stripe.dev/blog/meet-stripes-knowledge-ai-platform). Enterprise
knowledge-work agent (GTM/sales/compliance), not DS; product announcement, thin on mechanism
(defers skill selection to a follow-up). No run, no UUID — a reading note, not evidence. Nothing
that changes the roadmap, but useful external validation of directions GADS already took:

- **Active vs "extended" context (S3/vfs), 932-turn sessions without context degradation** →
  validates GADS's context-distillation bet (sliding-window 2+1, `orchestrator_summary`, live
  kernel snapshot as ground truth) over context-stuffing.
- **"A monolithic agent can't encode domain complexity → decentralized domain ownership, avoid
  micro-agent proliferation"** → validates the recipe/skill KnowledgeRegistry design (modular
  `.md` knowledge, not an agent zoo).
- **"Knowledge work needs task-specific tools/data/outputs unlike coding's uniform workflow"** →
  already encoded in GADS's faceted taxonomy + per-`task_type` recipe DAGs.
- **"Context locked into individual sessions" named as a pain point** → the cross-run error ledger
  (`research/error_ledger.jsonl`) is exactly the un-locking of session-local context; proof point
  for the error-learning stack.

**One adoptable/reinforcing concept:** Stripe lists **"reflection-based self-improvement loops for
skills"** as a *future* priority — verbatim GADS's open follow-up (auto-propose recipe/skill edits
from the `error_ledger_report.py` hardening report as a GOD task). A team at their scale prioritizing
what GADS already has the substrate for is a nudge that closing that loop is high-value. Cheap
companion: a per-recipe/skill quality tab in Studio (analogue of their AgentStudio "quality signals")
to make the loop observable.

**Where GADS is ahead:** the post is silent on evaluation methodology and provenance/citations,
substituting business metrics (deals, revenue) for any rigor story. That rigor — reproducibility +
methodological-appropriateness as first-class metrics, the skore audit gate, UUID/commit citation —
is exactly GADS's differentiator on the dimension that matters for a research platform.

## 2026-08-04 — `[survey]` Ellf (Explosion AI): recipes-as-code, and the precondition gap

Read the Ellf beta docs (beta.ellf.ai/docs — landing, `recipe-development`, `tasks-actions-agents`).
Explosion AI's new NLP platform, from the spaCy/Prodigy team. A reading note, no run, no UUID.

**The instructive difference — recipes-as-code vs recipes-as-specification.** An Ellf recipe is a
decorated Python function (`@task_recipe`, `@action_recipe`, `@agent_recipe`, `@service_recipe`)
that *executes directly*, with typed inputs (`Dataset[Literal["text"]]`), `field_props` driving
generated UI/CLI, and per-recipe dependencies shipped as a container image. A GADS recipe is
declarative YAML+Markdown compiled into a plan an LLM writes code *for*. So in GADS terms **Ellf
recipes are closer to native nodes than to recipes** — the same prefer-a-lookup-to-a-generation
instinct, drawn at a different line: their LLM is confined to *agents* (autonomous annotators),
while "actions execute deterministic code". Worth noting they reached the same instinct from the
annotation/data-development side rather than the autonomous-DS side.

Also: the docs contain **no "skills" concept** — recipes are the only composable unit. The
recipe/skill split remains a GADS-specific decomposition.

**The one transferable idea → issue #31.** Ellf declares inputs with *kind constraints* plus
`Validation` objects carrying operator, value, message and **severity** (`error` blocks submission;
`warning`/`info`/`success` do not). GADS's `applies_when` matches task_type/modality but never asks
whether the data is structurally *fit* for the method — and this cycle paid for that: five
end-to-end runs to establish that Amazon Fashion cannot support collaborative filtering (1.10
interactions/user, empty 3-core). That is a property the DataAnalyzer already measures **before the
Planner runs**. A declarative `preconditions` block evaluated at launch would have failed it once,
with the true reason, instead of failing several nodes deep after each fix.

The **severity grading** is the independent win: GADS today has only advisory `postconditions`
(deliberately not runtime-evaluated) and hard-fail `required_metrics`. There is no way to say "this
run is legal but distrust the absolute numbers" — exactly the state adaptive k-core relaxation
produces (bdf7a4d), and exactly what a `warning` level expresses.

**Where GADS is ahead:** no visible system-level evaluation instrumentation — no delegation dial,
no `pass@model`, no cross-run error ledger. Their quality story is per-project (holdout F1, training
curves, "rule-of-10" test sizing); GADS measures *its own scaffolding*. Their ADR project-plan
methodology is nonetheless a good source of NLP `invariants` if that recipe family is ever built:
validate learnability before scaling, training curves before spending annotation budget, reject
composite labels in favour of NER + role assignment.

## 2026-08-13 — `[method]` Three confounds between us and the D3 conditions arm

Starting issue #26 (run the API-familiarity arm of `approach_docs/014`) surfaced three defects
that would each, independently, have made the arm measure nothing. None is an engine result.
Recording them before any cell is read, per the standing discipline: **grade the failure class
before reading any cell.** Four pilot runs on `qrdata_ihdp_0_d3ols`, all graded
harness/design-confound and excluded from H1: `df1fe72f`, `cd57bf6f`, `56917ff0` (cancelled),
`4ee77379`.

**1 — Chain-of-thought returned as the program** (fixed, `1d721f8`). `get_code_completion`'s
unfenced fallback returned the longer output channel as-is. gemma-4-12b routes thinking into
`reasoning_content`; when it never opened a fence, the accumulated *deliberation prose* became
"the program", failed `ast.parse`, and surfaced as `ValidationError: syntax error (unterminated
string literal) at line n` — a false diagnosis the model cannot act on, so every retry re-burned
on it. `df1fe72f`: three attempts at `define_causal_question`, all prose (20344 / 2161 / 6317
chars, opening `*   Goal:` and `*    Wait, the prompt says:`), `exec_nodes: 0`. Note attempts 2
and 3 were far under the 8192 budget — truncation is a contributing case, not the cause. Fix:
gate the unfenced fallback on `ast.parse`; raise a typed `CodeGenerationError` carrying a
`truncated` flag from `finish_reason`, caught *inside* the retry loop (the generic handler
returned immediately, aborting the task with no retry) and converted into feedback that states
the remedy.

**2 — Correct code destroyed by a uniform indent** (fixed, `2e8f4c1`). After fix 1, `cd57bf6f`
still hit `exec_nodes: 0` — but the specimens show the model writing **correct statsmodels code**,
emitted with every line after the first at column 4. `_repair_stray_indent` compared each line
against the *original* predecessor, so in a uniformly-indented block only line 2 ever qualified;
the partial repair then failed to parse and was discarded wholesale. Comparing against the
*repaired* predecessor cascades down the block (`opens_block` still protects real nesting, the
`ast.parse` acceptance gate is unchanged). Replayed on all six real specimens: the two genuine
programs now parse (6 and 19 lines repaired), the four prose blobs route to the actionable path.
Also moved the honesty gate to the choke point — `ast.parse` after `_sanitize_code`, before
dispatch — which closes the fenced-path hole (prose after an unclosed fence) and saves a sandbox
round-trip.

**Both are the same class**, and it is the *third* instance: reasoning-model output-channel
failures masquerading as engine incapacity. Router token starvation (2026-07-20) was the first.
The signature to watch: a failure whose diagnosis names a *syntactic* defect the model never
committed.

**3 — Skill discovery silently defeated the manipulation** (fixed, `85db4ab`). The RecipeCompiler
falls back to keyword+semantic discovery when a node names no skills. Discovery matches on the
task description — which is deliberately library-agnostic — so it injected whatever was nearest:
both the statsmodels and the sklearn causal recipes were handed `causal_inference_dowhy` +
`causal_ml_econml` (verified on the task rows of `cd57bf6f` and `56917ff0`). The model duly wrote
`gads_causal_estimate_ate` in an OLS run. Two consequences:

- **The arm was not testing H1.** The API surface in the *prompt* was DoWhy in all three arms,
  and asymmetrically so: DoWhy got *matched* knowledge, the other two got *conflicting* knowledge
  (skill says use the native, intent says statsmodels). The deliberation spirals that kept node 1
  from emitting code are what an instruction conflict looks like from the inside.
- **The dial ledger was overstating autonomy.** D3 is defined as "no curated skill" — D4 is the
  rung where skills live. Discovery lifts the effective rung at runtime while `dial_rung.py`
  computes it from what the recipe *declares*, so the ledger recorded D3 for a prompt carrying
  curated library knowledge. This is a measurement-integrity gap in the dial framework itself,
  not specific to this arm; it deserves its own issue.

Fix: `RecipeTask.attached_skills` becomes three-state (`None` = absent → discovery, `[]` =
declared empty → no skill, non-empty → verbatim). The old `= []` default made absent and
declared-empty indistinguishable after `.dict()`, so "this node is deliberately skill-free" was
inexpressible. All three `.directed` causal recipes pinned `attached_skills: []` on every node
(DoWhy included — otherwise that arm alone runs at an effective D4) at v1.1.0.

**A negative result worth keeping: do not raise the token budget.** With no skills in the prompt,
gemma-4-12b emits fenced code for node 1 at the current 8192 budget in 2/2 trials (reasoning
3354 / 1489 tokens, `finish=stop`). At 16384, one of two trials ran away and burned the entire
budget (16381 reasoning tokens, zero content). The over-deliberation is **content-driven, not
budget-bound** — more headroom buys more rambling, not more code.

**Method note for the arm itself.** Two protocol changes, both to protect the comparison:
(a) all 12 cells (4 benchmarks × 3 surfaces) run on the current commit, DoWhy re-run included —
the 2026-07-16 0/5 D3 baseline predates the error ledger, adaptive retry and resume, so
new-code-OLS vs old-code-DoWhy would confound API surface with harness improvements;
(b) grading uses the new `--metrics-only` flag, because the QRData benchmarks' methodology block
requires `gads_causal_estimate_ate(` and forbids `CausalModel(` — DoWhy-specific checks that
would systematically fail the arms using the library the experiment manipulates. `online_classroom`
stays N/A, so H1's "≥4/5" reads against 4 benchmarks. Config parity with the baseline verified:
`routing_mode: local`, `local_fallback: none`, gemma-4-12b (the fallback legs landed 2026-07-30,
two weeks *after* the baseline, so it was fallback-free by construction).

## 2026-08-15 — `[build]` The EDA recipe: a descriptive intent, and a reusable transformation contract

Built `tabular_eda.descriptive.standard` (approach_docs/021, Phase 1). GADS had 26 recipes and
none did exploratory data analysis, though `taxonomy.yaml` names `descriptive` as its **first**
intent — so "explore this dataset" matched nothing and fell to the drafted lane, the rung where
the local engine is weakest. Six nodes: profile → quality → univariate → bivariate → recommend
transformations → summary.

**The deliverable is the manifest, not the charts.** `eda_transformations.meta.json` records, per
column, an imputation / scaling / encoding strategy drawn from closed vocabularies, plus quality
flags and — when the data is ML-destined — a split block. A later run consumes it via
`gads_apply_transformations`, so judgment made once during exploration is applied consistently
afterwards. The `.meta.json` suffix is load-bearing: any other `.json` in a workspace was being
auto-registered as an interactive Plotly artifact.

**Division of labour, per the 019 rule.** *Recommending* a transformation is judgment that
depends on the data, so it stays model-generated and measured. *Applying* one is invariant
mechanics, so it is deterministic native code — and the ordering inside it is the reason it must
be: the applier splits FIRST, fits on the training partition only, then applies those fitted
parameters to every partition, persisting them to `transformation_provenance.meta.json` so a
second file reuses rather than refits. Fitting before splitting leaks held-out statistics into
training. Proven rather than asserted: on data where `age` trends with time, the fitted median is
41.021 (train) and not 49.970 (full) — identical numbers would mean the guard had failed.

**Three data-corrupting bugs the unit tests caught before they shipped.** Continuous float
columns were flagged `id_like` and DROPPED (uniqueness alone condemns nearly every continuous
feature). Datetime columns were flagged `id_like`, and separately frequency-encoded, turning
timestamps into codes that carry no signal and do not transfer. And a time-ordered split whose
time column had been dropped silently degraded to row order — it looked correct only because the
test data happened to be in date order; it now raises, and split columns are protected from the
drop pass.

**Cloud e2e — the first run passed its contracts while producing nothing.** `3fc0ec86`: all six
nodes "completed"; neither deliverable existed. `recommend_transformations` printed a manifest to
stdout, never wrote the file, and invented its own key names (`impute` not `recommended_impute`,
`strategy`/`train_fraction` not `method`/`ratios`) — unusable by the applier regardless.
`univariate_distributions` made **zero savefig calls**, emitting a dict *describing* figures that
never existed. Both passed because the contracts were written on **variables, not artifacts**.

Remedy (v1.1.0): a native `gads_write_transformation_manifest` — the model decides per column,
the native serializes the schema, so the format cannot drift (a *cloud* model got it wrong; a 12B
never had a chance). Plus a `required_metrics` scalar on every artifact-producing node
(`n_manifest_columns`, `n_univariate_figures`, `n_bivariate_figures`), which is hard-fail where
recipe `postconditions` are advisory. `5b30faa6` then produced a correct manifest, 6 real
figures, and all 7 metrics; applying it to the full 48,842-row Adult extract yields a stratified
34,188 / 7,326 / 7,328 split, 15 → 106 columns, zero residual nulls, class balance preserved.

**Local (gemma-4-12b) — not viable for this recipe, and that is the finding.** Unaided
(`5f2740ba`, `319e8d4e`): `exec_nodes: 0`. The generated code was *substantively correct* pandas
profiling carrying a uniform 4-space indent plus a real unmatched paren; because the stray-indent
repair only commits when the result parses, the model was told `unexpected indent at line 2` — a
harness-fixable artifact — while the true defect went unmentioned. Fixed (`bc10b39`): the gate
now reports the **residual** error after a best-effort dedent. With honest diagnoses the model
still failed, at lines 17/16/9/24/29 — five *different* paren bugs that
`normalize_error_reason` collapses to one reason, so the same-reason guard stopped it at two
attempts (→ issue #33). With `local_fallback: native` (`8dd73299`): `exec_nodes: 2`,
`fallback_pass: 2`, **`pass_at_model: 0.0`** — the two natives carried nodes 1–2 and the model
did nothing unaided. Nodes 3–4 (plotting) then blocked the run, and they deliberately have no
`fallback_native` so capability stays measured. A latent defect surfaced here too: node 1's
`fallback_call` referenced `df`, but in this failure mode nothing ever executed, so the safety
net would have NameError'd and silently no-opped in exactly the state it exists for (fixed
v1.1.1, proven from an empty namespace).

**Reuse works, and on the local model.** Follow-up `b1f055ba` on `5b30faa6`: keyword routing
injected the EDA preamble, `gads_apply_transformations` ran, and the transformed train/val/test
parquets plus provenance were written — `status: completed, model_used: local_model,
mode: followup`. Calling the native is within the 12B's reach even though authoring the recipe's
nodes is not, which is a clean illustration of what nativizing buys. Method note: an earlier
apparent pass was **my own manual applier run** left in the workspace; the stale files were
deleted and the test re-run before this claim was made.

**Two bugs found in passing, both pre-existing.** Every workspace `.json` was registered as a
Plotly figure, so `metrics.json` — and the natives' `model_checks.json`,
`survival_metrics.json`, `cox_report.json`, `risk_profiles.json`, `km_summary.json`,
`recommendation_profile.json` — rendered as broken tiles on **every** run; now detected by
content rather than an allow-list that would rot (`2851e24`, validated against 3 real figures and
all 89 historical `metrics.json` files). And the follow-up lane wrote the model *object* into the
JSON column, so every follow-up failure persisted an **empty** `result_json`, destroying the
diagnostics exactly when they matter (`b989a52`).

**Open:** the follow-up lane never calls `resolve_stage_model` — `ExecutionManager()` takes the
Coder's hardcoded `local_model` default — so it ignores `routing_mode` entirely; in `cloud` mode
a user silently gets the local engine. Not yet filed.

## 2026-08-21 — `[build]` One controlled vocabulary, and what measuring it actually showed

The Router's labels, recipe `applies_when` and spec `taxonomy:` blocks were three vocabularies
naming the same things, and nothing reconciled them (approach_docs/024 §1). The Router's schema
admitted 9 task terms; `taxonomy.yaml` already carried a 45-term crosswalk and a 24-family task
tree. **11 of 29 recipes declared an `applies_when.task_type` that no Router label could equal** —
survival (both), recommendation, LTR, causal discovery, anomaly detection, ordinal, transform and
all three analytics suites — so the coverage oracle could never confirm them, and `unmapped_task_types`
was non-empty for a third of the library.

Everything is now derived from `taxonomy.yaml`: the schema field descriptions, the prompt's
vocabulary blocks (`render_task_vocabulary`), the labels on the way out (`canonical_task` /
`canonical_modality`), the oracle's comparison (`tasks_overlap` — a bare family covers its
subtypes), and the SampleBudget training set (`training_task_families`). The vocabulary gained
`analytics.exploratory`, a `data_preparation` family, a `data_quality` regime and the six terms
recipes declared that nothing mapped. Six spec blocks were invalid; all 57 now validate.
`scripts/test_vocabulary.py` asserts the whole chain with no LLM calls.

**Second-order damage the drift was doing, beyond routing.** `intent.task_type` also feeds run
taxonomy tagging and the SampleBudget threshold. A survival, ranking or recommendation run whose
label fell outside the enum was not in `_TRAINING_TASK_TYPES`, so it got the **200K analysis row
cap instead of the 50K training one** — the guard silently inverted for exactly the recipes most
likely to time out.

**Seven recipes said "never match" in prose that nothing evaluated.** `_anti_signals_fire` handles
`objective_contains` and exact tag equality; a `routing:` key holding an English sentence fell
through both, so the dial arms and AAH rungs were offered to the Router *and* returned as oracle
candidates. Now `applies_when.pin_only: true` (legacy prose still honored), withheld from the
agent catalogue and from the oracle.

### The measurement, and three claims it killed

`gemini-3.7-flash` — the baseline's Router — went down mid-session (a 14-token prompt hung past
45s while haiku answered in 0.94s through the same proxy). Both arms were re-run on
`claude-haiku-4.5` instead: the pre-change arm from a detached worktree at `f629e86`, the
post-change arm from the working tree, then **both rescored by one scorer** with ground truth,
canonicalization and the pin-only exclusion taken from the current tree. Two runs per arm.

| | BEFORE (`f629e86`) | AFTER |
|---|---|---|
| task label correct | 90.7% / 88.9% | 94.4% / 94.4% |
| modality correct | 92.6% / 92.6% | 98.1% / 96.3% |
| selection (routable pins) | 19/19 | 19/19 |
| pin-only instrument selected | **3** | **0** |

**What the same-model comparison refuted.** (a) The selection improvement I expected does not
exist — both arms select 100% correctly, and the old harness's 47.6% was entirely the
arm-variant miscount. The Router was never getting those wrong. (b) The "8 specs classify as
`unknown`" evidence is Gemini-specific: on haiku the old enum produced **zero** unknowns, because
the model simply emitted terms outside its own declared enum (`recommendation`, `survival_ml`,
`data_transformation`) that happened to be crosswalk keys. The enum was not blocking a capable
model; it was being quietly violated by one. (c) The change was not a pure win — the richer
vocabulary introduced a real misroute, `qrdata_hospital_treatment_d0` going from
`causal_inference` + DoWhy to `regression.survival` + Cox, because adding survival to the menu
created an attractive wrong answer for a treatment-effect objective. Remedied in the prompt
(survival covers ML risk prediction *and* yields to causal when the question is about a
treatment's effect); both targeted regressions cleared on re-run.

**The noise floor, which is the methodological finding.** Two runs of the *identical* pre-change
code disagree on 1 task label and **3 recipe choices** (5.6 pp) — `aah_rung2_star` is unstable in
both arms. So a single-run selection number is not trustworthy better than ±5.6 pp, and the
label gain (+2 to +3 specs, with both post-change runs landing identically at 51/54) is real but
modest. Any future routing claim needs repeats; the harness has been reporting single runs.

**Two harness defects found while measuring, both of which had corrupted numbers already
reported.** `asyncio.TimeoutError` stringifies to `""`, and the scorer's `if r.get("error")`
treated that as success — six transport failures were being counted as Router misclassifications.
And `task`/`modality` are many-valued facets scored on their first value only, so `amlb_segment`
(`[image, tabular]`, image-region features flattened into a table) had its correct `tabular`
marked wrong. Both fixed; a circuit breaker now aborts after 5 consecutive failures rather than
logging an outage as a result — it fired correctly on an Anthropic `InternalServerError` burst at
spec 40 and wrote nothing.

**Open:** the 23 arm-variant pins are excluded from selection scoring rather than scored against
the arm the objective implies, which is right but leaves those specs unmeasured. Re-running both
arms on `gemini-3.7-flash` when it recovers would confirm the label gain holds across models —
the numbers above are haiku-only.
---

## 2026-08-21 — `[build]` A golden-task library: sourced, not yet validated

Second piece of work this session (approach_docs/026): a library of "golden reference"
specs — real datasets, an open hypothesis question, **no methodology named anywhere**,
`disable_recipes: true` so the Router/Planner draft freely. Distinct from every other spec
in `specs/`: 54 of the 60 pre-existing files name a method in the title or objective
(`"...RFM + K-Means"`, `"Using Bayesian inference..."`) — correct for D3+ recipe-realization
benchmarks, useless for measuring what the Planner *chooses*. The one prior exception,
`qrdata_hospital_treatment_d0.md`, was the template: it already proves the shape works and
is gradeable.

**Sourced two ways, deliberately kept separate in what "ground truth" means.**

QRData (10 new specs: LaLonde job training, billboard→deposits DiD, Prop 99 tobacco tax,
NJ/PA minimum wage, growth-mindset RCT, drinking-age mortality RDD, trainee-earnings
matching, India female-reservation policy, UK MPs' wealth RDD, Angrist-Krueger schooling
IV) — **CC BY-NC 4.0**, confirmed by fetching the repo's actual LICENSE file rather than
assuming (the sibling survey had flagged this unconfirmed). Each has QRData's own gold
scalar as an external anchor. 406 of 411 questions remain untapped; ~200 more are
causal-*discovery* shaped ("which direction is causal") rather than effect-estimation and
were skipped as a different task type.

causaldata / NickCH-K (6 new specs: Castle Doctrine self-defense laws, active-choice organ
donation nudge, Chinese farmers' insurance take-up, Malawi HIV-incentive RCT, GI Bill
mortgage subsidies, John Snow's 1854 cholera map) — **MIT**, new to the project (not in the
2026-07-09 sibling survey at all). Three source files were Stata `.dta`; converted losslessly
via `pandas.read_stata`, verified same row/column counts before and after.

**The honest finding this pass produced: causaldata has no scalar ground truth, and pretending
otherwise would be the same mistake as the routing-eval tolerance-fabrication caught earlier
this session.** QRData is a graded benchmark with a paper-derived gold number; causaldata is
*textbook example data* — there is no canonical scalar anywhere upstream. Every causaldata
`expected.json.metrics` is empty by design; `notes.md` instead records the published paper's
*directional* finding (Cheng & Hoekstra: Castle Doctrine laws → ~8% increase in homicide;
Thornton: any cash incentive roughly doubled HIV-result pickup from a 34% baseline). Writing
in a fabricated tolerance band around a number nobody had produced would have looked more
complete and been less true.

**One spec (`ak91_schooling_wages`) documents its own trap.** The QRData gold (8.53%) is
specifically a 2SLS estimate using quarter-of-birth as an instrument; naive OLS is confounded
by ability bias and gives a different, larger number. The spec deliberately doesn't name the
instrument — whether a run even recognizes the identification problem, not just whether it
hits the number, is part of what this one measures.

**`castle_doctrine_homicide` trimmed 139 upstream columns to 17.** Cheng & Hoekstra's original
regression carries region-by-quarter fixed effects and per-state linear time trends —
regression plumbing that would hand the identification strategy away through column names
alone, the same failure mode as naming the method in the objective text.

**All 16 pass `scripts/test_vocabulary.py`** (taxonomy blocks resolve — the guard built
earlier this session caught nothing wrong, which is the point of having it) and every
dataset was reloaded post-write to confirm no truncation from the copy/convert step.

**Explicitly not done: no reference runs.** Every `expected.json` here is sourced-but-
unvalidated — QRData's tolerance bands are provisional placeholders around an external
number, not derived from observed GADS run variance, and every notes.md says so. Launching
reference runs and tightening against them is the immediate next step, same discipline
`research/benchmarks/README.md` already requires everywhere else in this repo.

---

## 2026-08-21 — `[method]` The golden-task batch, validated — and what broke while validating it

Third piece of work this session: reference runs for all 16 golden hypothesis-investigation
specs sourced earlier (approach_docs/026 §5), cloud mode, `Task.assigned_to` verified on
every launch. Two sections, deliberately separate — one is evidence about GADS, the other
is evidence about the local stack this session ran on top of.

### The findings

**Exact/near-exact matches, on a design that names no method.** `billboard_deposits`
(6.52 = gold 6.52), `women_reservation_water` (9.2524 ≈ gold 9.25, plus an unprompted
2SLS structural estimate nobody asked for), `learning_mindset_achievement` (AIPW
0.3925 / LinearDML 0.3955 ≈ gold 0.39, full refutation checks), `thornton_hiv_incentive`
(ATE 0.4395, matching "34% baseline roughly doubled," plus a dose-response and a distance
heterogeneity finding nobody asked for), `mortgages_gi_bill` (Fuzzy RDD/2SLS, correct
sign, unprompted instrument recognition). Five specs where a freely-drafted plan reached
the right number by a route nobody dictated — the entire premise of building this library
D0 rather than D3.

**The framing/scale divergences are the actual research payoff of this pass, not noise
to average away.** `minwage_employment`: GADS modeled employment as a level, the gold is
a proportion — different outcome quantities. `prop99_cigarette_sales`: the *same run's*
Synthetic Control estimate (-41.71, matching the gold's direction) and its own TWFE
regression (+7.68) disagreed on SIGN, and its own narrative never noticed. `drinking_age_mortality`:
right direction, ~77x the gold's magnitude — the spec never restricted the RDD bandwidth
the way the gold's methodology did, and D0 specs, correctly, don't tell it to.
`mps_wealth`: identical method to the gold (sharp RDD) but reported on a log-points scale
instead of a wealth level. `trainee_program_earnings`: a genuine sign flip — regression
adjustment (-1021.94) vs. the gold's matching estimator (+2457.89) — on the exact class of
tiny confounded dataset the Dehejia-Wahba/LaLonde literature is *about*, reproducing a
known instability rather than exhibiting a new one. `social_insurance_takeup`: correct
but shallow — found the right near-zero direct effect, never touched the source paper's
actual social-network-spillover story, because nothing in the column list hinted at it.

**One reproducible failure**: `jobs_lalonde_training` hit the identical
`'numpy.ndarray' object has no attribute 'iloc'` error twice, the second time surviving a
full 5-model escalation ladder before giving up. Real, repeatable weak spot on a
17-column undifferentiated covariate block — not a fluke worth a third retry.

**One recognized-but-unexecuted trap, and probably the single most important result in
the batch**: `ak91_schooling_wages` correctly identified, unprompted, that it needed an
instrument (`linearmodels.IV2SLS`, no instrument named anywhere in the spec) — then
failed to execute the 2SLS across the whole model ladder, gave up, and shipped the
confounded naive OLS (6.73%) with a single caveat sentence buried in the prose
("2SLS remains unexecuted"). Knowing the right method and failing to deliver it, while
still producing a plausible-looking headline number, is a more dangerous failure mode
than either outright failure or not recognizing the problem — a reader who skims gets
the wrong answer with no obvious red flag.

**One likely self-inflicted underpowering**: `castle_doctrine_homicide` got the sign
right but lost significance — plausibly because trimming the source's 139 columns to 17
(done to avoid leaking the identification strategy through fixed-effect column names)
also removed the region-quarter FEs and state trends the original paper's power depends
on. A real tension in *how to build these benchmarks*, not a finding about the model.

`snow_cholera_water` (4 rows, the easiest case) passed as a sanity check. All 16
`expected.json` files now carry a `reference_runs.cloud` entry (or, for the one failure,
an honest record of two failed attempts); every number is n=1 and every tolerance is
provisional pending a second run.

### What broke while measuring it (infra, kept separate on purpose)

Three distinct incidents, none of them evidence about GADS:

1. **DB connection-pool exhaustion**, first launching all 13 remaining specs at once
   (`QueuePool limit of size 5 overflow 10 reached` — untuned SQLAlchemy defaults), then
   recurring at just 3-way concurrency — the recurrence at low concurrency is the
   important part; it points at a slow leak in a background loop rather than a pure
   oversubscription ceiling, and neither `pool_size`/`max_overflow` nor the leak were
   fixed this session.
2. **A full livelock** followed: 0% CPU, zero log growth, zero task progress across two
   10-minute windows, nothing ever marked `failed`. GADS has no path to resume a stuck
   project — the fix was a backend restart, which permanently abandoned every workflow
   that was mid-flight. Switched to one-project-at-a-time launches afterward specifically
   to make this diagnosable (a single stuck heartbeat is legible; thirteen interleaved
   ones are not).
3. **`routing_mode` reverts to the `.env` default on every restart** (already documented
   in `CLAUDE.md`, re-learned expensively): the post-livelock restart silently reset
   cloud mode back to local, and a batch of 4 specs launched and executed entirely on
   `local_model` before anyone checked `Task.assigned_to` rather than trusting `/config`'s
   self-report. That batch was discarded, not written up, and relaunched clean.

All three are now in `project_local_stack_gotchas.md` (items 3 and its 2026-08-21
update) so a future session inherits the lesson instead of re-discovering it at the same
cost. Recommended, not done: give the DB engine explicit pool sizing and audit the
executor's background loops for a session that isn't being closed.

## 2026-08-22 — `[method]` Local vs. cloud on the golden batch: not uniformly worse, differently dangerous

Re-ran all 16 golden specs from the 2026-08-21 cloud pass under `routing_mode=local`
(`local_model` only, no escalation ladder) to get a genuine local-vs-cloud comparison on
identical specs. Full write-up: `approach_docs/026_golden_hypothesis_task_sourcing.md`
§5c; per-spec detail in each `research/benchmarks/<spec>_v1/{expected.json,notes.md}`.

### What it showed

Local is not uniformly worse than cloud. `jobs_lalonde_training` succeeded outright
(ATE 0.0757 vs. gold 0.074) where BOTH cloud attempts failed identically on
`'numpy.ndarray' object has no attribute 'iloc'` — though the run's own 95% CI is
degenerate (`[0,0]`, a broken bootstrap the Synthesizer misread as reassuring precision,
not a genuine second win). `trainee_program_earnings` matched cloud to 4 decimal places
(-1021.9390 vs. -1021.94) — both engines called the identical deterministic native
`gads_causal_estimate_ate`, proof that when the calling code correctly identifies the
treatment/outcome/confounder roles, the computation itself is perfectly reproducible
regardless of which LLM wrote the surrounding code.

But three results — `prop99_cigarette_sales`, `castle_doctrine_homicide`, and
`organ_donation_nudge` — are the most important finding in this pass, and arguably in
the whole golden-batch effort so far: all three produced misidentified or inverted-sign
causal estimates via that same native ATE call, reported with full confidence, and ALL
THREE passed their own placebo/subset refutation checks with no visible warning.
`castle_doctrine_homicide`'s -0.1973 and `organ_donation_nudge`'s +0.1327 are the exact
opposite sign from both the published literature and GADS's own cloud run on the
identical spec. `prop99_cigarette_sales` silently substituted the wrong treatment
variable (`after_treatment` instead of `california`), measuring a nationwide trend
instead of a state-specific policy effect. The mechanism is the same every time: the
native function's arithmetic is exactly correct (as `trainee_program_earnings` proves),
but locally-generated code sometimes wires the wrong variables into it, and refutation
checks validate specification *robustness*, not specification *correctness* — a model
can misidentify the causal contrast and still sail through every check it runs on
itself. Worth naming as its own failure class: **confidently wrong, refutation-passing**.
Related: `ak91_schooling_wages` used its own instrument as the treatment, producing an
implausible 39% return with no caveat at all — more dangerous than cloud's honest "2SLS
remains unexecuted" precisely because it looks methodologically complete.

The remaining failures are a varied catalogue, not a single pattern: a Planner replan
whose JSON exceeded `max_tokens` and halted the whole project rather than just failing
the task (`billboard_deposits`); hallucinated nonexistent APIs
(`women_reservation_water`, `organ_donation_nudge`'s second failure mode); a
capability/complexity mismatch on trivial 4-row data (`snow_cholera_water`); the
same-reason-stop retry guard failing to fire on an identically-repeating error twice
(`minwage_employment`, `social_insurance_takeup`) while firing correctly elsewhere
(`thornton_hiv_incentive`); a data-plumbing bug where an upstream step silently dropped
a column a downstream step needed (`mps_wealth`); and a resource-budgeting failure
distinct from all of the above — `mortgages_gi_bill` never sampled its 214,144-row
dataset down before an expensive native call, so every retry cost ~10-11 minutes,
exhausting all 3 workflow replans over ~2h12m with zero artifacts produced (notably,
GADS's own cloud run on the identical spec hit the same unsampled-dataset timeout once
and recovered only because it had a faster model and replan budget to spare).

**Overall verdict**: the efficiency-boundary question this platform exists to measure
isn't just "does local finish the task" — it's "when local is wrong, does it *know* it's
wrong." Cloud's failures in the 2026-08-21 pass were mostly visible: crashes, or explicit
caveats like ak91's "2SLS remains unexecuted." Local produced multiple silent,
confidently-wrong, refutation-passing results (three inverted-sign) that would mislead a
researcher with no cloud run or original paper to check against. On this evidence, the
answer is usually no.

### What broke while measuring it (infra, kept separate on purpose)

Two operational footnotes, neither evidence about GADS's capability: an external
systemd `--user` session process was found periodically and harmlessly attempting to
restart the backend, failing on the port-8001 collision, throughout the pass —
investigated, confirmed unrelated, no action taken. And `backend.log` turned out to be
stdout-buffered rather than line-buffered once redirected to a file — it can sit frozen
for 10+ minutes while the process is genuinely alive and working, making DB heartbeat
freshness and the sandbox container's own access log (`docker logs sandbox`) the
reliable liveness signals, not log-file growth or mtime. Worth remembering for future
long-running-workflow monitoring.

---

*Add entries above this line. Keep the evidence discipline: UUID or it didn't happen.*
