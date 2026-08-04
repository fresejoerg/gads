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

---

*Add entries above this line. Keep the evidence discipline: UUID or it didn't happen.*
