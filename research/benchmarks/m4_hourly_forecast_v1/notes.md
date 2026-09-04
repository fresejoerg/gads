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
| local | google/gemma-4-12b | `304dfafb` | **FAIL** — node 1 never produced valid Python across all 3 workflow attempts |

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

### What this says about the D4 result in approach_docs/007

The paper's headline (§6.3) is that curated skills lift local success from 12.2% to 55.6%.
§7.1 flags that the D4 arm is dominated by survival analysis. This run is the **first D4
local attempt outside survival** and it failed at rung D4 with skills attached — one data
point, but it points the same way as the confound warning rather than against it. It should
be added to the corpus before that finding is defended.

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
