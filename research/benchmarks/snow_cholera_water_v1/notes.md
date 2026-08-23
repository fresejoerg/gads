# snow_cholera_water_v1 — provenance & grounding

**Established:** 2026-08-21. Sourced per approach_docs/026 (golden hypothesis-investigation
tasks), from the `causaldata` package (Huntington-Klein / Cunningham / Hernan-Robins textbook
data), **MIT licensed**: https://github.com/NickCH-K/causaldata.

**Format note:** the upstream file is Stata `.dta`; converted to CSV losslessly via
`pandas.read_stata` (2026-08-21, this session) for sandbox compatibility. No values altered.

## Ground truth — no scalar gold; this is the harder, more honest case

Unlike QRData, `causaldata` is textbook example data, not a graded benchmark — there is no
single external scalar answer to grade against. The external anchor recorded here is the
**published paper's finding**, which is directional/qualitative (see below), not a number
with an implied tolerance.

Snow (1855); this is the aggregated 4-row summary used to illustrate the before/after comparison across water-supplier groups (Huntington-Klein, The Effect). The historical finding is unambiguous — the contaminated source shows a much higher death rate — but the point of including this tiny, extreme-signal case is as a sanity check: does a run at least get the direction right on the easiest possible case?

## `metrics` is deliberately empty

No GADS reference run has been made for this spec. Establishing a GADS-canonical value (a
verified reference run's estimate) is explicit follow-up work, per approach_docs/026 §5 —
premature to fabricate a tolerance band around a number nobody has produced. Grading this
benchmark today means checking direction and defensibility of the identification strategy
against the published finding above, not a scalar match.

## Reference run (2026-08-21, cloud, project `050b55a4`)

Drafted lane chose a DiD framing (1849 vs. 1854, across supplier groups) unprompted on
this 4-row dataset and produced: **-56.9 deaths per 10,000** (a 40.13% relative
reduction) attributable to clean water. Direction matches the historical finding
(contaminated water increases cholera mortality) — this was always the easiest possible
sanity check in the batch (see original framing in approach_docs/026 §5: "does a run at
least get the direction right on the easiest possible case?"), and it did. This is now
the GADS-canonical reference value; there is still no external scalar to check it
against, so treat the `tol` as bounding *our own* reproducibility, not agreement with a
published number.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `31d4e902`) — failed on the easiest case in the batch

**This is the most striking local-mode result of the pass, precisely because the task
is trivial.** 4 rows, 4 columns, a plain before/after comparison across two supplier
groups — cloud solved it with a simple DiD in one clean pass. Local instead reached for
a heavier apparatus: encode `supplier`/`treatment` as categorical dummies, then invoke
the native `gads_causal_estimate_ate` DoWhy node (a keyword-triggered native preamble
meant for real observational datasets with many rows). The categorical encoder's
transform step failed with "unknown categories in column 0" — the signature of a
fit/transform split where the transform set contains a category the fit set never saw,
which is close to guaranteed on a 4-row table split any way at all.

**A complexity-mismatch failure, not a reasoning failure.** Nothing about identifying
"contaminated water increases mortality" from 4 rows requires machine-learning
infrastructure; local's plan applied ML-shaped tooling to a problem that needed
arithmetic. Two earlier tasks in this same load-and-sample step also failed for a
related reason ("missing required metrics: sample_size") — consistent with a pattern of
over-elaborating a trivial input rather than under-elaborating a hard one.

No GADS-canonical local value exists for this benchmark. Combined with `jobs_lalonde_training`
(local succeeded, cloud failed twice) and `women_reservation_water`/`billboard_deposits`
(local failed on execution mechanics), local-mode results across this batch are not
monotonically worse than cloud — they're differently distributed, failing on different
axes (planning-JSON overflow, API hallucination, complexity mismatch) than cloud's
failures (a specific numpy/pandas type bug, an unexecuted 2SLS).
