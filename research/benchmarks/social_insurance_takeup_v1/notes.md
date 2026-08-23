# social_insurance_takeup_v1 — provenance & grounding

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

Cai, De Janvry & Sadoulet (2015, AEJ: Applied Economics 7(2)): the paper's focus is social-network spillovers in take-up decisions, using this data with an instrumental-variables design centered on the `intensive` and `default` assignment and village-level peer take-up — a genuinely identification-strategy-dependent question, no single external scalar gold recorded here.

## `metrics` is deliberately empty

No GADS reference run has been made for this spec. Establishing a GADS-canonical value (a
verified reference run's estimate) is explicit follow-up work, per approach_docs/026 §5 —
premature to fabricate a tolerance band around a number nobody has produced. Grading this
benchmark today means checking direction and defensibility of the identification strategy
against the published finding above, not a scalar match.

## Reference run (2026-08-21, cloud, project `173550fe`) — verified cloud model throughout

Drafted lane confirmed covariate balance (all |SMD| < 0.1, correctly recognizing this
is randomized data), then found the DIRECT effect of `intensive` on take-up to be
statistically indistinguishable from zero across regression-with-fixed-effects,
placebo refutation (-0.0008), and CATE heterogeneity analysis (~-0.0044 across risk
aversion and default subgroups). Clean, well-executed analysis of the question it chose
to ask.

**Not wrong, but answers a narrower question than the paper.** Cai, De Janvry &
Sadoulet's actual contribution is about **social-network spillovers** — using
`intensive`/`default` assignment as instruments for peer take-up rates, since the
paper's real finding is that take-up spreads through village social networks, not that
the information session itself has a large direct individual effect. A near-zero direct
effect is consistent with that framing (nobody expects a big direct effect if the true
mechanism is peer influence) but this run never attempted the network/IV analysis the
paper is actually about — the spec's column list (`address`, `village`, `default`,
`intensive`, etc.) doesn't name "spillover" or "instrument," so nothing steered it
either way. Worth watching whether other D0 specs on this general shape — data with a
richer identification story than a simple treatment/control comparison — produce the
same kind of "correct but shallow" outcome.

This is now the GADS-canonical value for the DIRECT-effect question (no external scalar
exists to compare against). Only one reference run so far.

## Reference run — LOCAL mode (2026-08-22, project `d454a8af`) — stuck at EDA

Never got past exploratory data analysis. The identical error — "Validation Error:
sequence item 13: expected str instance, int found" (a type-coercion bug, likely a
string-join or f-string mixing a string list with an integer somewhere) — recurred
across four separate retry attempts at Step 2, the same same-reason-repeat pattern
seen on `minwage_employment`. No treatment variable was constructed, no ATE was
attempted; nothing comparable to cloud's near-zero direct-effect finding exists here.

No GADS-canonical local value exists for this benchmark.
