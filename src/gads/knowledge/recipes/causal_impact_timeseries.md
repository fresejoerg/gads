---
id: causal_effect.timeseries.causalimpact
version: 1.1.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference, time_series]
  data_modality: [tabular]
  signals:
    - temporal_ordering_required: true
    - intervention_event: true
  anti_signals:
    - task: causal_discovery

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [pycausalimpact, pandas, matplotlib]

# ——— DAG TEMPLATE ———
dag:
  - id: prepare_time_series
    intent: "Parse and validate the time index. Identify the outcome metric column and any available control series (unaffected by the intervention). Identify the intervention date that splits pre-period from post-period. Print a summary of the timeline and any gaps/nulls."
    worker_tier: T2
    attached_skills: [causal_impact_timeseries]
    produces: [ts_data, pre_period, post_period]
    postconditions:
      - "ts_data is not None"
      - "isinstance(pre_period, list)"
      - "isinstance(post_period, list)"

  - id: run_causal_impact
    intent: "Fit the CausalImpact model using pycausalimpact. Pass the pre_period and post_period. If control series are available, include them as additional columns. Store results in `ci`. Print ci.summary()."
    depends_on: [prepare_time_series]
    worker_tier: T2
    attached_skills: [causal_impact_timeseries]
    produces: [ci]
    postconditions:
      - "ci is not None"

  - id: extract_impact_metrics
    intent: "Extract the average absolute effect, relative effect (%), cumulative effect, and p-value from ci.summary_data. Store the absolute effect in a variable named exactly `abs_effect` and relative effect in `rel_effect`. Print the full summary table."
    depends_on: [run_causal_impact]
    worker_tier: T2
    attached_skills: [causal_impact_timeseries]
    produces: [abs_effect, rel_effect]
    postconditions:
      - "isinstance(abs_effect, float)"
    required_metrics: [abs_effect, rel_effect]

  - id: visualize_impact
    intent: "Generate the CausalImpact three-panel chart (original + counterfactual, pointwise effect, cumulative effect). Save as Figure 1. Also plot the pre-period fit quality to validate the model. Save as Figure 2."
    depends_on: [extract_impact_metrics]
    worker_tier: T2
    attached_skills: [causal_impact_timeseries, visualization_best_practices]
    postconditions:
      - "abs_effect is not None"

  - id: interpret_results
    intent: "Write a plain-language interpretation: state the intervention date, the estimated effect size and direction, whether the effect is statistically credible (p < 0.05), and any caveats about the counterfactual quality (e.g. lack of control series). Save a JSON summary artifact."
    depends_on: [visualize_impact]
    worker_tier: T2
    postconditions:
      - "abs_effect is not None"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "The pre-period must contain enough observations to fit the state-space model (minimum ~20 time points)."
  - "Control series must not themselves be affected by the intervention."
  - "Store abs_effect and rel_effect as plain Python floats, not numpy scalars."
---

# Time-Series Causal Impact Analysis (pycausalimpact)

## Rationale
When an intervention occurs at a single point in time (a product launch, policy change, or marketing campaign), **CausalImpact** uses a Bayesian structural time-series model to synthesise a counterfactual — what the metric *would have been* without the intervention — by learning its relationship to unaffected control series during the pre-period. The difference between the observed and counterfactual post-period yields the causal effect estimate with full uncertainty bounds.

## When to use
Use when: the data is a time series (daily, weekly, monthly), there is a clear before/after intervention boundary, and you want to estimate the total or average impact of the event. Examples: launch of a feature and its effect on engagement, a price change and revenue impact, a public health intervention and disease incidence.

## Key Constraints
- Requires a defined pre-period and post-period split date.
- Control series (if available) dramatically improve counterfactual quality.
- Minimum ~20 pre-period observations needed to fit the state-space model.
- Does not handle multiple simultaneous interventions.
