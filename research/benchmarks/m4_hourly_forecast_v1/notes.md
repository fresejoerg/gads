# m4_hourly_forecast_v1 — provenance & tolerance rationale

**Established:** 2026-09-04. **Status:** first exercise of the forecasting stack.
**Recipe:** `timeseries_forecast.autogluon.standard` v1.1.0 (4 nodes, D4 — every node
carries the `autogluon_timeseries` skill).

**Why this benchmark exists.** The forecasting recipe had been built and was agent-routable,
but had never been run: no spec referenced it, no benchmark covered it, and it appears zero
times in 129 dial-ledger runs. Coverage was *declared*, not demonstrated. This is also the
first non-survival test of the D4 rung, which matters because the paper's headline result
(approach_docs/007 §6.3) is confounded by survival analysis dominating the D4 arm.

## Dataset

M4 Competition, Hourly split (Makridakis et al. 2018), first 50 series by numeric id —
35,000 rows, 700 observations per series, staged deterministically by
`scripts/stage_m4_hourly.py`. Hourly was chosen over Daily/Monthly because its strong daily
seasonality (m=24) gives the seasonal-naive baseline real teeth: a model that ignores
seasonality cannot beat it by accident.

The official held-out horizon (48 steps/series) is staged to `m4_hourly_test.csv` but is
**not given to GADS** and is not part of the pass criterion — see "What is not scored".

## Tolerance rationale

`n_series` and `prediction_length` are **exact**. Both are pure functions of the staged
data: 50 distinct `item_id`s, and the recipe's explicit rule (~10% of median series length,
min 1) applied to 700 observations gives 70.

`best_model_mase` is a **threshold (< 1.0), not a target value**, and this is forced by the
recipe rather than chosen for convenience. Its TIME BUDGET invariant mandates
`time_limit=120` with `presets='fast_training'`, and the recipe's own comment concedes the
consequence: *"a wall-clock budget makes model selection machine-load-dependent — this
recipe is for exploratory forecasting; reproducibility-critical runs must pin an explicit
fixed model set instead."* Under a wall-clock budget the set of models AutoGluon manages to
fit depends on machine load, so which model wins — and its MASE — is not reproducible. An
exact expected value would be a fiction that fails on a busy laptop.

MASE < 1 is nonetheless a real criterion: MASE is normalised by the in-sample seasonal-naive
error, so < 1 means the model beat the baseline it is measured against.

**If this benchmark is ever needed for reproducibility rather than exploration**, the fix is
the one the recipe already names: a variant that pins an explicit model set instead of a
wall-clock budget. That is a recipe change, not a tolerance change, and it should not be
smuggled in by loosening a number here.

## A recipe ambiguity this staging exposed

The profiling node asks for `naive_mae` — *"the mean absolute deviation of the target from
its own mean"* — without saying whether that is computed globally or per series. On this
panel the difference is not cosmetic:

| interpretation | value |
|---|---|
| global MAD over all pooled rows | 19562.06 |
| seasonal-naive MAE, mean over series (m=24) | 918.31 |

M4 Hourly series differ by orders of magnitude in scale, so the pooled figure is dominated
by *between-series* scale rather than within-series variability, and is close to meaningless
as a forecasting baseline. `naive_mae` is therefore recorded as **informational only** in
`expected.json`, so a future run's number can be interpreted rather than judged.

The underlying defect is in the recipe: for a multi-series panel it should specify a
per-series scaled baseline (which is what MASE already does properly). Worth fixing there
rather than papering over here.

## What is not scored

- **The official M4 horizon.** The recipe derives its own `prediction_length` (70) from the
  data; M4's competition horizon is 48. Forcing agreement would mean changing the recipe to
  suit the benchmark, which inverts the point of the exercise. `m4_hourly_test.csv` is
  staged for an optional external check, not for the pass criterion.
- **Competition leaderboard comparison.** No published M4 sMAPE/OWA figures are asserted
  here. Comparing to them would require reproducing M4's exact horizon, scoring rule, and
  full 414-series set; none of that is done, so quoting them would be misleading.

## Reference runs (2026-09-04)

| Mode | Engine | Project | Outcome |
|---|---|---|---|
| cloud | gemini-3.7-flash | `ed35ec90` | **PASS** — 4/4 nodes first attempt, 0 escalations |
| local | google/gemma-4-12b | `304dfafb` | **FAIL** — node 1 never produced valid Python (3/3 attempts) |
| local | prism-ml/bonsai-27b | `4c4dc4f8` | **FAIL** — nodes 1-3 passed 9/9; node 4 failed 3/3 on index handling |

The two local engines fail for completely different reasons, which is the most useful thing
this benchmark has produced so far. gemma-4-12b never emitted valid Python and never reached
the sandbox. bonsai-27b emitted valid Python every time, completed the three analysis nodes
on every one of three attempts, and reproduced the cloud run's metrics **bitwise**
(`best_model_mase` 0.9757443932230794, `prediction_length` 70, `naive_mae` identical to 16
significant figures). Its only failure is node 4.

An earlier bonsai run (`d893b76a`) completed all four analysis nodes but died in the
Synthesizer at 12,045 tokens — the model had been loaded with an 8,192-token context window.
Reloaded at 32,768 it passed the same nodes first-attempt. **Context length is a config trap
that surfaces late, in reporting, and looks like a model-quality failure.**

Cloud result: `n_series` 50, `prediction_length` 70, `best_model_mase` **0.9757**, all
required artifacts present, all methodology patterns satisfied.

**The recipe is sound; the local engine could not execute it.** That disambiguation is the
main value of running both arms, and it could not have been inferred from the local run
alone.

### The local failure is a codegen defect, not a forecasting one

Node 1 (`profile_time_series`) failed identically on every attempt — *"generated text is not
valid Python (closing parenthesis ']' does not match opening parenthesis '(')"* at line 10
and line 13. It never reached the sandbox, so nothing about forecasting was tested. This is
the same failure mode measured independently the day before on a held-out set
(`scripts/eval_coder.py`): gemma-4-12b parsed on only 74.3% of 35 held-out Coder prompts,
with unclosed and mismatched brackets dominating.

The node is a plausible trigger: its intent packs six numbered sub-requirements (identify
three columns, infer frequency, count series and lengths, compute a baseline) into a single
code block, which is a long uninterrupted generation for a 12B model. Splitting it, or
giving it a `fallback_native`, is the obvious remedy — the recipe declares no
`fallback_native` on any node, and the two registered forecasting natives
(`gads_timeseries_fit`, `gads_timeseries_predict`) are wired to nothing.

### Node 4 is the sticking point, and the errors say why

`generate_forecasts_and_visualize` failed three times under bonsai with three *different*
errors — `KeyError: 'item_id'`, `KeyError: 'id'`, then `"String or int arguments are only
possible..."`. All three are the same underlying mistake: AutoGluon returns predictions with
`item_id`/`timestamp` as a **MultiIndex**, not as columns, and the node's intent does not say
so. The distinct reasons are why the adaptive retry policy kept trying rather than declaring
a loop — correct behaviour, but it never found the answer.

This is a recipe gap, not only a model gap: the cloud engine happened to know the idiom, the
local one did not, and the recipe never states it. The fix belongs in the
`autogluon_timeseries` skill (show `forecasts.reset_index()` explicitly) and/or a
`fallback_native` on the node — of which this recipe has none.

### A pass@model defect this run exposed

The ledger row for `4c4dc4f8` reads `pass_at_model: 1.0` on a workflow that **failed**.
That is not a typo. `server.py` builds the denominator from tasks with
`status == "completed"`, so a node the model attempted and never got right is dropped from
the measurement entirely — 3 of 4 nodes succeeded, and the metric reports 3/3.

pass@model is therefore biased upward precisely where capability is weakest, and it cannot
fall below 1.0 on account of an outright failure. This matters for approach_docs/007 §5,
which presents the metric as what keeps the efficiency-boundary measurement honest, and for
§6.5, which reported mean pass@model 1.000 on both engines and attributed the lack of
discrimination to fallbacks being rarely enabled. That explanation is incomplete: the
denominator is structurally incapable of counting the model's hardest failures.

Fixing it changes the meaning of a field that 129 historical ledger rows already use, so it
is flagged here rather than silently changed.

### What this says about the D4 result in approach_docs/007

The paper's headline (§6.3) is that curated skills lift local success from 12.2% to 55.6%.
§7.1 flags that the D4 arm is dominated by survival analysis. These are the **first D4 local
attempts outside survival** and both failed at rung D4 with skills attached — two data
points, pointing the same way as the confound warning rather than against it. They should be
added to the corpus before that finding is defended.

The bonsai run also complicates the rung story usefully: it failed at D4 *despite* the model
being fully capable of the analysis, because one node's intent omitted an API detail. That
is a failure of the curated artifact, not of the delegation level — which is exactly the
kind of thing an ordinal "amount of scaffolding" scale cannot express.

### On the marginal forecast quality

MASE 0.9757 beats seasonal-naive by 2.4%. That clears the threshold, and the threshold is
the right criterion for a pipeline test — but it is not a good forecast, and the benchmark
should not be cited as evidence that GADS forecasts well. AutoGluon had a 120-second budget
across 50 series.

### `naive_mae` ambiguity — resolved empirically for this run

The cloud run computed `naive_mae` **globally**: 19562.060448963268, matching the staged
reference 19562.0604489633 to 10 significant figures. So the model chose the pooled
interpretation. That confirms the value is reproducible, and equally confirms it is the
less useful of the two readings — it remains informational, not a criterion.
