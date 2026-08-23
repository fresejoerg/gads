# organ_donation_nudge_v1 — provenance & grounding

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

Kessler & Roth (2014, NBER WP 20378): the active-choice framing DECREASED registration relative to the opt-in default — the nudge backfired. Direction is the citable finding; the paper does not reduce to one simple scalar in the way a numeric benchmark answer would (it reports several related estimates across specifications).

## `metrics` is deliberately empty

No GADS reference run has been made for this spec. Establishing a GADS-canonical value (a
verified reference run's estimate) is explicit follow-up work, per approach_docs/026 §5 —
premature to fabricate a tolerance band around a number nobody has produced. Grading this
benchmark today means checking direction and defensibility of the identification strategy
against the published finding above, not a scalar match.

## Reference run (2026-08-21, cloud, project `bf24de12`) — verified cloud model throughout

Strong, well-corroborated match to the published direction. TWFE DiD with
state-clustered SEs: **-0.0225** (a 2.25 percentage-point DECLINE in registration),
p=0.0008 — the nudge backfired, exactly as Kessler & Roth (2014) found (the run's own
report cites them by name, correctly attributing its own external validation source
without being told to). Backed by real, unprompted robustness work: an event-study
confirming no significant pre-trend divergence (parallel-trends assumption holds), and
an in-space placebo test across 26 control states where only 4/26 placebo estimates
matched or exceeded the real effect's magnitude — a randomization-inference-style
check giving informal p≈0.15, reasonable evidence the effect isn't spurious on a small
panel.

One minor internal inconsistency, not substantive: a single trailing caption line in
the raw task log states the policy "significantly increased" registration — the
opposite of every other sentence in the same report, including the explicit numeric
finding and the Kessler & Roth citation. Looks like a stray/garbled auto-caption
artifact from report assembly rather than an analysis error; the substantive finding
(decline, -0.0225) is unambiguous and consistent everywhere else.

This is now the GADS-canonical value for this benchmark (no external scalar exists to
compare against; the published finding validated here is directional). Only one
reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `2302a681`) — third sign contradiction

**A pattern is emerging, not three isolated incidents.** An earlier completed step
produced `causal_effect_result.json` via `gads_causal_estimate_ate(treatment_col='treatment',
outcome_col='Rate', confounders=['period','Year'])`: **ATE = +0.1327** — the opposite
sign from the published finding (registration DECREASED after the active-choice switch)
and from cloud's own -0.0225 on this identical spec. A later, redundant verification
step in the same run then got stuck retrying on an unrelated bug (`ImportError:
cannot import name 'LinearRegressionEstimator' from 'dowhy'` — that class doesn't exist
in the installed dowhy version) and never produced a final synthesized report; this
number is taken directly from the earlier successful step rather than a polished
narrative.

**This is the third local-mode result in this pass with an inverted causal sign**
(after `castle_doctrine_homicide`, -0.1973 vs. the correct positive direction), on top
of `prop99_cigarette_sales`'s separate omitted-comparison-group error (also confidently
wrong, different mechanism). Three sign/specification errors via the same native
`gads_causal_estimate_ate` call pattern, out of roughly a dozen local attempts that got
far enough to call it, is a rate worth treating as systemic rather than coincidental —
plausibly something about how locally-generated code constructs the binary treatment
column upstream of the native call (e.g., the sense of a boolean comparison, or which
group gets coded 1 vs. 0) is a recurring blind spot. Worth a dedicated look at whether
the Coder's prompt or a skill needs a stronger invariant here (e.g., an explicit
"verify treatment==1 corresponds to the group named as treated in the objective" check)
given the native function's own arithmetic is not in question — only what it's fed.

No GADS-canonical local value should be treated as directionally trustworthy here.
