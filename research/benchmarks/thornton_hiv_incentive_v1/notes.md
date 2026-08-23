# thornton_hiv_incentive_v1 — provenance & grounding

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

Thornton (2008, American Economic Review 98(5)): 34% of the zero-incentive group learned their results; any nonzero incentive roughly doubled that share. This is randomized assignment, so a simple group-mean comparison should recover the effect directly — one of the cleanest identification cases in this batch.

## `metrics` is deliberately empty

No GADS reference run has been made for this spec. Establishing a GADS-canonical value (a
verified reference run's estimate) is explicit follow-up work, per approach_docs/026 §5 —
premature to fabricate a tolerance band around a number nobody has produced. Grading this
benchmark today means checking direction and defensibility of the identification strategy
against the published finding above, not a scalar match.

## Reference run (2026-08-21, cloud, project `46083e66`) — verified cloud model throughout

Strong, well-corroborated result and the cleanest match in the causaldata batch. GADS's
DoWhy + IPW ATE = **0.4395** (95% CI [0.3973, 0.4817]) sits comfortably with the external
anchor (34% baseline pickup, any incentive roughly doubled it). Went well beyond the
minimum: a dose-response analysis found +17.71 percentage points per additional
incentive unit, and a heterogeneity analysis found the effect is STRONGER for people
living further from the testing center (distance x incentive interaction = +0.0141) —
a substantively sensible result (distance is presumably a real barrier that money helps
overcome) that nothing in the spec asked for.

This is now the GADS-canonical value for this benchmark (no external scalar exists to
compare against exactly; the qualitative published finding is closely matched).

## Reference run — LOCAL mode (2026-08-22, project `4e55b6eb`)

Outright timeout failure, no ATE produced. The causal-estimation step (propensity score
matching / DiD) exceeded the 600s sandbox execution limit twice in a row; the adaptive
retry loop's same-reason-stop guard fired correctly this time ("Same failure reason twice
(Timeout) — stopping retries after 2 attempt(s)") and aborted the task cleanly rather than
burning through all 10 attempts. Everything upstream of that step succeeded: dataset load
+ mandatory N=100 sample constraint, median/mode imputation, and five confounder/outcome
visualizations. The run's own Synthesizer wrote an honest, self-aware executive summary
("critical execution failures... prevented the completion of core causal inference tasks")
rather than papering over the gap or fabricating a number.

Two things worth flagging:
1. This is the first local-mode result in this pass where the same-reason-stop guard
   visibly did its job on a Timeout — contrasting with `minwage_employment` and
   `social_insurance_takeup` earlier in the pass, where an identical error recurred 3-4
   times without the guard stopping it. Inconsistent, not systemically broken.
2. Unlike `castle_doctrine_homicide`, `prop99_cigarette_sales`, and `organ_donation_nudge`
   (all confidently-wrong sign flips that passed their own refutation checks), this failure
   is visible and honestly reported — a much safer failure mode than a silent wrong answer,
   even though neither produces a usable result.

This is the 15th of 16 golden specs to complete its local-mode attempt in this pass;
only `mortgages_gi_bill` remained in progress at the time this was written up.
