# mortgages_gi_bill_v1 — provenance & grounding

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

Fetter (2013, AEJ: Economic Policy 5(2)): finds the mortgage subsidies raised mid-century home-ownership rates, using the qob_minus_kw cutoff as the identifying variation. No single scalar headline number captured here — the paper reports effects across several subgroups.

## `metrics` is deliberately empty

No GADS reference run has been made for this spec. Establishing a GADS-canonical value (a
verified reference run's estimate) is explicit follow-up work, per approach_docs/026 §5 —
premature to fabricate a tolerance band around a number nobody has produced. Grading this
benchmark today means checking direction and defensibility of the identification strategy
against the published finding above, not a scalar match.

## Reference run (2026-08-21, cloud, project `99c1429a`) — verified cloud model throughout

Strong result, correctly identifying a fuzzy RDD/2SLS design unprompted: used GI-bill
mortgage-subsidy eligibility (the `qob_minus_kw` cutoff) as an instrument for actual
veteran status `vet_wwko`, since eligibility doesn't perfectly determine veteran status
(a fuzzy, not sharp, discontinuity). First-stage effect -0.1923 (t=-43.91), reduced-form
-0.0547 (t=-13.02), baseline 2SLS LATE = **0.2846** (p<0.0001), refined to **0.3938**
(SE=0.0126) after adding race and state-of-birth fixed effects. Both large, positive,
and highly significant — matches Fetter (2013)'s published direction (mortgage
subsidies raised home ownership).

One replan cycle along the way: an early robustness/placebo step timed out on the full
214,144-row dataset and the workflow correctly self-recovered via its normal
replan-on-failure mechanism (attempt 2/3), reaching a clean final report. This is
routine adaptive-retry behavior, not one of the session's operational incidents — noted
here only because it happened to be directly observed while monitoring this run.

This is now the GADS-canonical value for this benchmark (no external scalar exists).
Closes out the full reference-run pass — all 16 golden specs sourced in approach_docs/026
now have at least one cloud reference run recorded.

## Reference run — LOCAL mode (2026-08-22, project `4a765cf1`)

Total failure. Never produced an estimate, never even got causal estimation running.
Step 1 (load + mandatory sample constraint) itself exceeded the 600s sandbox limit on
attempt 1 — remarkable, since loading a CSV and applying a row cap should be trivial;
the mandatory sample constraint clearly wasn't applied before something expensive ran
downstream in that same execution block. Step 2 (`gads_causal_estimate_ate`) also
timed out. The workflow correctly triggered its replan-on-failure mechanism twice more
(MAX_WORKFLOW_ATTEMPTS=3): attempt 2 got as far as successfully loading the full,
UNSAMPLED 214,144x6 dataframe into the kernel but never reached a causal estimate before
attempt 3 exhausted the replan budget, at which point the remaining steps are explicitly
marked "Workflow exhausted max planning attempts; task never ran."

Total wall-clock time: ~2h12m (launched 03:22, resolved 05:34) — by far the longest run
in this local-mode pass. Root cause, confirmed by direct observation during monitoring:
the sandbox never sampled the dataset down before calling the native ATE function, so
every retry attempt cost ~10-11 minutes (near the 600s timeout) just running DoWhy on
the full 214K rows, leaving no room to actually iterate on fixing the underlying issue
within the retry budget. This is a resource-budgeting failure, not a reasoning failure —
the model's code was plausible, it was simply too slow to iterate.

Notably, this row-count risk is not exclusive to local: GADS's own CLOUD reference run
for this identical spec (see above) also hit one timeout on the same unsampled
214,144-row dataset during a robustness/placebo step, and recovered only because it had
a full replan cycle's worth of budget and a faster model to spend it with. Local hit the
same structural risk with less room to recover — three replans, all consumed by the same
timeout pattern, ending in the only complete, fully-visible outright failure across the
whole local-mode pass with zero artifacts produced (thornton_hiv_incentive at least
produced exploratory plots before its causal-estimation step failed).

This closes out ALL 16 golden specs' local-mode reference-run attempts.
