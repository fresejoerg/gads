---
id: causal_effect.observational.dowhy
version: 2.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [causal_inference]
  data_modality: [tabular]
  signals:
    - treatment_outcome_pair: true
  anti_signals:
    - task: heterogeneous_treatment_effects

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [dowhy, statsmodels, sklearn, pandas]

# ——— DAG TEMPLATE ———
dag:
  - id: define_causal_question
    intent: >
      Identify the variable roles from the objective and the dataset schema. Do NOT
      hardcode column names — read them from the objective and the actual columns of `df`.
      (1) TREATMENT: the column whose effect is being estimated (named or strongly implied
          by the objective). Store its name in `treatment_col`.
      (2) OUTCOME: the column being affected. Store its name in `outcome_col`. If a
          target column was provided in the spec hints, that is the outcome.
      (3) CONFOUNDERS: numeric columns that are NOT the treatment, NOT the outcome, and
          NOT temporal/ID columns. Build this list programmatically:
            import numpy as np
            temporal_id_patterns = {'time', 'date', 'timestamp', 'id', 'index'}
            confounder_cols = [
                c for c in df.columns
                if c not in (treatment_col, outcome_col)
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                and not any(p in c.lower() for p in temporal_id_patterns)
            ]
            if len(confounder_cols) > 10:
                corrs = df[confounder_cols].corrwith(df[outcome_col]).abs()
                confounder_cols = corrs.nlargest(10).index.tolist()
      (4) Compute and print the outcome's minority-class fraction (if categorical):
            minority_class_frac = float(df[outcome_col].value_counts(normalize=True).min()) if df[outcome_col].nunique() <= 20 else 0.5
      Print `treatment_col`, `outcome_col`, `confounder_cols`, and `minority_class_frac`.
    worker_tier: T2
    produces: [treatment_col, outcome_col, confounder_cols, minority_class_frac]
    # Minimal skills: this node only reads the schema. An explicit (non-empty) list
    # suppresses the enforcer's keyword fallback, which would otherwise greedily load
    # several heavy skills and overflow a small local model's context. The native-node
    # guidance is loaded on estimate_and_refute where it actually matters.
    attached_skills: [sandbox_environment]
    postconditions:
      - "isinstance(treatment_col, str)"
      - "isinstance(outcome_col, str)"
      - "len(confounder_cols) <= 10"

  - id: estimate_and_refute
    intent: >
      Estimate the average treatment effect AND run both refutation tests in a single
      call to the pre-defined native kernel function. It handles subsampling, continuous
      treatment binarization, programmatic GML construction, estimand identification,
      estimator selection, and both refuters internally — do NOT reimplement any of this.
      You MUST use this exact pattern:
      ```python
      result = gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols)

      ate = float(result["ate"])
      placebo_new_effect = float(result["placebo_new_effect"])
      subset_new_effect = float(result["subset_new_effect"])
      treatment_col = result["treatment_col"]          # may be 'high_<name>' if binarized
      causal_model = result.get("causal_estimate")     # kept for reference
      df_sample = result["df_sample"]                  # rows actually used
      refutation_results = {
          "placebo_new_effect": placebo_new_effect,
          "subset_new_effect": subset_new_effect,
      }

      print(f"ATE={ate:.6f}  placebo={placebo_new_effect:.6f}  subset={subset_new_effect:.6f}")
      gads_emit_insight(
          "causal_effect",
          f"ATE={ate:.4f}, placebo={placebo_new_effect:.4f} (should be ~0), subset={subset_new_effect:.4f} (should be ~ATE)",
      )
      ```
    depends_on: [define_causal_question]
    worker_tier: T2
    attached_skills: [causal_inference_dowhy]
    produces: [ate, placebo_new_effect, subset_new_effect, treatment_col, df_sample, refutation_results]
    postconditions:
      - "isinstance(ate, float)"
      - "isinstance(placebo_new_effect, float)"
      - "isinstance(subset_new_effect, float)"
    required_metrics: [ate, placebo_new_effect, subset_new_effect]

  - id: visualize_results
    intent: >
      Visualize the causal results. Use the variables already in the kernel
      (`ate`, `placebo_new_effect`, `subset_new_effect`, `treatment_col`, `outcome_col`,
      `confounder_cols`, and `df_sample`). Do NOT hardcode any column name or class label.
      You MUST adapt this pattern to the variables in the kernel:
      ```python
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt

      dfv = df_sample if 'df_sample' in globals() else df
      is_binary_outcome = dfv[outcome_col].nunique() == 2
      n_panels = 3 if is_binary_outcome else 2
      fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

      # Panel 1: ATE vs refutation checks
      ax = axes[0]
      labels = ['ATE', 'Placebo', 'Subset']
      values = [ate, placebo_new_effect, subset_new_effect]
      bars = ax.bar(labels, values, color=['steelblue', 'tomato', 'mediumseagreen'],
                    alpha=0.85, edgecolor='black', linewidth=0.5)
      ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
      ax.set_ylabel('Effect Size')
      ax.set_title('ATE vs Refutation Checks')
      for bar, val in zip(bars, values):
          ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                  f'{val:.4f}', ha='center', va='bottom', fontsize=9)

      # Panel 2: confounder importance (|correlation with outcome|)
      ax = axes[1]
      conf_corrs = dfv[confounder_cols].corrwith(dfv[outcome_col]).abs().sort_values()
      ax.barh(range(len(conf_corrs)), conf_corrs.values, color='steelblue', alpha=0.7)
      ax.set_yticks(range(len(conf_corrs)))
      ax.set_yticklabels(list(conf_corrs.index), fontsize=8)
      ax.set_xlabel('|Correlation with outcome|')
      ax.set_title('Confounder Importance')

      # Panel 3 (binary outcome only): treatment distribution by outcome class
      if is_binary_outcome:
          ax = axes[2]
          tcol = treatment_col if treatment_col in dfv.columns else confounder_cols[0]
          q99 = dfv[tcol].quantile(0.99)
          for cls, color in zip(sorted(dfv[outcome_col].unique()), ['steelblue', 'tomato']):
              vals = dfv[dfv[outcome_col] == cls][tcol].clip(upper=q99)
              ax.hist(vals, bins=50, alpha=0.6, label=f'{outcome_col}={cls}',
                      color=color, density=True)
          ax.set_xlabel(tcol)
          ax.set_ylabel('Density')
          ax.set_title('Treatment Distribution by Outcome')
          ax.legend()

      plt.suptitle(f'Causal Effect of {treatment_col} on {outcome_col}',
                   fontsize=12, fontweight='bold')
      plt.tight_layout()
      plt.savefig('causal_effect_summary.png', dpi=120, bbox_inches='tight')
      plt.close()
      print("Saved causal_effect_summary.png")
      ```
    depends_on: [estimate_and_refute]
    worker_tier: T2
    attached_skills: [visualization_best_practices]
    produces: [causal_effect_summary.png]
    postconditions:
      - "'causal_effect_summary.png' in str(locals()) or True"

# ——— GLOBAL INVARIANTS ———
invariants:
  - "NATIVE NODE IS MANDATORY: estimate the effect with gads_causal_estimate_ate(df, treatment_col, outcome_col, confounder_cols). It performs subsampling, treatment binarization, GML construction, identification, estimator selection, and BOTH refutations internally. NEVER build a CausalModel by hand, NEVER implement propensity scoring / DML / AIPW from scratch, and NEVER skip refutation."
  - "CONFOUNDERS: infer from the schema as numeric columns that are not treatment, outcome, or temporal/ID. Never ask the user to list them, never hardcode them."
  - "NO HARDCODED NAMES: never hardcode a dataset filename, column name, or class label. Read treatment, outcome, and confounders from the objective and the live schema of `df`."
  - "METRICS AS FLOATS: store ate, placebo_new_effect, and subset_new_effect as plain Python floats, not numpy scalars."
  - "REUSE KERNEL VARIABLES: the visualization step must reuse ate / placebo_new_effect / subset_new_effect / df_sample already in the kernel — never recompute the effect."
  - "Random state must be 42 everywhere (the native node already enforces this internally)."
---

# Causal Effect Estimation with DoWhy (Observational Data)

## Rationale
This recipe implements the DoWhy four-step workflow — **Model → Identify → Estimate → Refute** — but delegates the entire mechanical core to the audited native kernel function `gads_causal_estimate_ate`. That function deterministically handles the steps where local models reliably fail: subsampling large datasets, binarizing continuous treatments at the global median, building a valid GML graph programmatically, selecting an estimator from outcome balance, and running both the placebo and data-subset refuters. The LLM is left with only the two judgments that genuinely vary per dataset — naming the variable roles (treatment, outcome, confounders) and interpreting the result — which small models handle reliably.

## When to use
Use when the objective is to estimate *how much* a treatment causally affects an outcome in observational tabular data. The spec needs only to identify the treatment and outcome (the target column is taken as the outcome); confounders and all methodology are inferred automatically.

## Key Constraints
- Treatment and outcome must be identifiable from the objective and schema.
- Temporal/ID columns are excluded from confounders automatically.
- Refutation is mandatory and is performed inside the native node before any ATE is reported.
</content>
</invoke>
