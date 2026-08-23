# castle_doctrine_homicide_v1 — provenance & grounding

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

Cheng & Hoekstra (2013), Journal of Human Resources 48(3): the laws are associated with a statistically significant ~8% increase in murder and non-negligent manslaughter, and did not deter burglary, robbery, or aggravated assault. This is a headline finding, not a scalar GADS gold — the original paper's regression uses region-by-quarter fixed effects and state linear trends not included in this trimmed extract.

## `metrics` is deliberately empty

No GADS reference run has been made for this spec. Establishing a GADS-canonical value (a
verified reference run's estimate) is explicit follow-up work, per approach_docs/026 §5 —
premature to fabricate a tolerance band around a number nobody has produced. Grading this
benchmark today means checking direction and defensibility of the identification strategy
against the published finding above, not a scalar match.

## Reference run (2026-08-21, cloud, project `450b37f1`) — verified cloud model throughout

Drafted lane fit a TWFE DiD with state+year fixed effects, clustered SEs, and
poverty/unemployment/youth-demographic controls: **+0.1057 homicides per 100,000**,
p=0.6037 — right direction (positive, matching the published finding), but NOT
statistically significant, whereas Cheng & Hoekstra's original ~8% increase was.

**Likely self-inflicted, and worth recording as a lesson about the trimming choice
itself, not just the model's answer.** `castle_doctrine_homicide.csv` was deliberately
cut from the source's 139 columns to 17 (approach_docs/026 §5 explains why: the
region-by-quarter fixed effects and 51 state-specific linear time trend columns looked
like they'd hand the identification strategy away through column names alone). But
those dropped columns are exactly what the original paper's precision depends on to
soak up state-specific secular trends — without them, a simpler TWFE spec has
materially less power, which is a very plausible full explanation for losing
significance even with the same sign and a broadly similar-order-of-magnitude point
estimate. **This is a real tension in how these benchmarks should be built**: trimming
too aggressively to avoid leaking the method can silently degrade the identification
problem itself, making a "the drafted lane failed to find significance" reading
unfair. Worth revisiting whether this dataset should include a *derived* trend/FE
scaffold (without literally naming "region_quarter_fe" columns) rather than the raw
139-column dump or this thin 17-column cut.

Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `214c60c4`) — sign contradiction

Clean run, 12/12, zero failures. ATE = **-0.1973**, reported as "a statistically
significant reduction in homicide rates" when Castle Doctrine laws are adopted — the
**opposite sign** from the published finding (Cheng & Hoekstra 2013: laws are
associated with a significant ~8% INCREASE in homicide) and from GADS's own cloud run
on this identical spec (+0.1057, correct direction though not significant). Placebo
(0.0163, correctly near zero) and subset (-0.1939, stable) refutation checks both
"pass" — which is exactly the trap: those checks validate that a given specification is
internally consistent under resampling, not that the specification has the treatment
polarity right. A likely mechanism (unverified without re-deriving the regression):
the `post`/treatment indicator's coding got inverted somewhere in the generated code, a
single-bit error that a robustness check by construction cannot catch since it
resamples the same (mis-)specification.

This is the second local-mode result in this pass (after `prop99_cigarette_sales`) that
is confidently wrong and passes its own validation — reinforcing that this class of
error (right process, wrong polarity/specification, undetected by refutation checks) is
a real recurring local-mode risk, not a one-off. Both cases used the native
`gads_causal_estimate_ate` node; the node's *arithmetic* is presumably correct given its
inputs, so the risk sits entirely in how the calling code sets up the treatment,
outcome, and confounder roles.

No GADS-canonical local value should be treated as directionally trustworthy here.
