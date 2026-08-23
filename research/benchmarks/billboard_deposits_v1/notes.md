# billboard_deposits_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData billboard_impact.csv (Causal Inference for the Brave and True, ch.13), gold DiD estimate. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
research/internal use; do not redistribute commercially.

## Ground truth

The external anchor is QRData's own gold answer, established by the benchmark's authors from
the cited textbook/paper — not computed by us. **No GADS reference run has been made yet.**
The `tol` in `expected.json` is a provisional placeholder band, not derived from observed
run-to-run variance (see approach_docs/024's lesson on the same mistake for the Router: an
unvalidated tolerance is a claim about precision we don't have).

## Why `methodology` is empty and `recipe_id` is null

This is deliberate, not an oversight: a D0/drafted-lane spec has no fixed DAG or fixed code
pattern to check against — that's the entire point of the D0 rung (approach_docs/013). What
"methodologically appropriate" means here is closer to BLADE's decision-space grading (a
distribution of defensible choices) than to a `required_patterns` regex check, and that
scorer does not exist yet (flagged as open work in approach_docs/026 §5).

## Next step

Launch a reference run (any mode) against this spec, record the result here, and tighten (or
justify) the tolerance — the same discipline `research/benchmarks/README.md` already requires
for every other benchmark in this repo. Until that happens, treat this benchmark as
**sourced but not yet validated**.

## Reference run (2026-08-21, cloud, project `e82d9d02`)

Drafted lane (D0, `disable_recipes: true`) chose an OLS DiD model
(`deposits ~ poa + jul + poa:jul`) unprompted — nothing in the spec named DiD, or any
method. Result: **ATE = +6.52**, matching the QRData external gold (6.52) essentially
exactly. The run's own analysis flagged the effect as **not statistically significant**
(p = 0.124) — a nuance the original QRData question didn't ask about but that a
competent free analysis surfaced anyway; worth keeping in mind when this benchmark is
used to judge "did the run get it right" — point-estimate accuracy and honest
uncertainty reporting are both part of methodological appropriateness here.

Only one reference run so far; tolerance (0.05) is tightened from the provisional band
but not yet validated against a second run's variance.

## Reference run — LOCAL mode (2026-08-22, project `7d3c6732`) — fatal halt, not a wrong answer

**This failed at the infrastructure level, not the analysis level.** Router, Planner,
PlanCritique all completed normally, and two execution steps ran (load data, create the
`poa_x_jul` interaction term). Then a Planner replan's structured-output call hit a
`max_tokens` ceiling mid-JSON (Instructor's retry exhaustion message: "output is
incomplete due to a max_tokens length limit") and the workflow terminated with
`[HALTED] Fatal system error` — not a graceful "replan failed, proceed to synthesis with
partial results," a hard stop with no report and no captured stdout from the completed
steps.

This is a genuinely different failure class from anything seen in the cloud pass: the
local model's plan-JSON verbosity outgrew whatever token budget the Planner call is
configured with, and the error handling around that specific failure mode doesn't
degrade — it kills the run. Worth flagging as a potential GADS robustness gap
(structured-output truncation during REPLANNING specifically, as opposed to during
execution, where the adaptive retry loop is much more forgiving) rather than treating
this as evidence about the local model's causal-reasoning capability, which this run
never got far enough to exercise.

No GADS-canonical local value exists for this benchmark.
