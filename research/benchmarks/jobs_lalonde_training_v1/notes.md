# jobs_lalonde_training_v1 — provenance & tolerance rationale

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks). Unlike the D3-directed `qrdata_*` benchmarks already in this repository, this spec is
**D0 (undirected)**: `disable_recipes: true`, no method named anywhere in the objective, no
`recipe_id`. It measures the free-drafted lane, not a pinned recipe's realization.

**Source:** QRData jobs_0.csv (Jobs dataset / LaLonde NSW), gold ATT. License: QRData is **CC BY-NC 4.0** (non-commercial) — retained for
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

## Reference run attempt 1 (2026-08-21, cloud, project `c25be7ee`) — FAILED

The drafted plan's causal-graph step crashed (`'numpy.ndarray' object has no attribute
'iloc'`), triggered a replan, and the second attempt's downstream tasks never ran
(`Workflow exhausted max planning attempts`). No estimate was produced — the only number
in the report is the naive unadjusted mean difference (-0.0890), explicitly flagged in
the report itself as confounded and requiring adjustment, so it is NOT recorded as a
canonical value here.

This is an honest, useful data point, not noise to discard: of the 4 wave-1 specs, this
is the one with by far the messiest covariate structure (17 columns, x0-x16, mixed
continuous/discrete confounders) — real evidence about where the drafted lane's code
generation is fragile even on cloud, worth keeping rather than quietly retrying until it
passes. A retry is queued; both outcomes will be recorded.

## Reference run attempt 2 (2026-08-21, cloud, project `7abc6881`) — FAILED, same root cause

Verified cloud model throughout (no config contamination this time — the intervening
attempt on `8872a4f5` was abandoned mid-run by the second livelock incident, not
written up). Escalated through the full ladder — gemini-3.5-flash-lite, gemini-3.7-flash,
claude-sonnet-5, gpt-5.6-terra, kimi-k2.7-code, 11 calls at the failing step, $0.40 total
— and still hit the **identical** error as attempt 1: `'numpy.ndarray' object has no
attribute 'iloc'` at the propensity-score estimation step.

**This is now a reproducible finding, not noise.** Two independent attempts, different
model mixes within the escalation ladder, same exact failure mode on the same step
(propensity-score / doubly-robust estimation over the `x0`-`x16` covariate block). The
drafted lane appears to have a systematic blind spot on this dataset's shape — 17
undifferentiated numeric covariate columns, mixing continuous and near-binary values,
is apparently enough to trip up generated propensity/DML code across a wide span of
model capability. Per the one-retry policy, no third attempt was launched. This
benchmark's status is: **sourced, genuinely hard for the drafted lane, no GADS canonical
value established.** Worth a targeted look at whether this is a Coder prompt gap (e.g.
not defensively handling a `.values`/`ndarray` return from a sklearn call before
`.iloc`-style indexing) rather than a dataset-specific issue — if so it would likely
recur on any spec with a wide raw covariate block.

## Reference run — LOCAL mode (2026-08-22, project `df75510b`) — cloud/local reversal

**The local engine succeeded where cloud failed twice.** Point estimate 0.0757 vs.
QRData's gold 0.074 — a close match, via DoWhy ATE estimation. Cost: $0.00, 26 LLM
calls, 102.3k tokens. The run hit a *different* error early on ('>' not supported
between instances of 'list' and 'int' during missing-value detection, at Step 2) but
recovered via the normal replan mechanism and reached the propensity-score/ATE step
cleanly — apparently generating different code than whatever pattern reliably triggers
cloud's `'numpy.ndarray' object has no attribute 'iloc'` bug on this same dataset.

**But the uncertainty quantification is broken, and the report doesn't notice.** The
reported 95% CI is `[0.0000, 0.0000]` — a zero-width interval is not a real confidence
interval on n=2,570 individual-level economic outcomes; it is almost certainly a failed
or defaulted bootstrap. The Synthesizer's own narrative reads this as *good* news
("the narrow confidence interval indicates some uncertainty") — inverting a red flag
into reassurance. This is a distinct and arguably more dangerous failure mode than
either extreme (missing entirely vs. clearly wrong): a plausible point estimate
wrapped in a broken-but-confident-sounding uncertainty claim.

**Framing for this benchmark going forward**: track point-estimate accuracy and
uncertainty-quantification correctness as separate axes. On the first axis, local beat
cloud here. On the second, local produced a silently-broken artifact that reads as
correct. Neither engine gets an unqualified pass on this spec.
