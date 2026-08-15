---
id: tabular_eda.descriptive.standard
version: 1.1.1
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [eda, exploratory_analysis, data_profiling, descriptive_analysis]
  data_modality: [tabular]
  signals:
    - no_labeled_target: true
    - objective_contains: [explore, exploratory, eda, profile, profiling, understand the data,
                           data quality, what is in this dataset, describe the data,
                           summarize the dataset, data audit, inspect the data]
  anti_signals:
    - objective_contains: [train a model, build a classifier, forecast, predict the]
    - task: causal_inference

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [pandas, matplotlib, seaborn]

# ——— DAG TEMPLATE ———
dag:
  - id: profile_dataset
    intent: >
      Load the dataset into `df` and profile EVERY column on the FULL data (do not sample —
      a sampled profile is what the DataAnalyzer already provides). For each column record
      dtype, non-null count, null rate and number of unique values; for numeric columns also
      min, max, mean, std, the quartiles and skew; for categorical columns the most frequent
      values. Store the per-column result in `profile` and print a readable table.
      Bind the dataset-level scalars as plain Python numbers: `n_rows`, `n_cols`, and
      `missing_cell_rate` (fraction of all cells that are null, between 0 and 1).
    worker_tier: T2
    produces: [df, profile, n_rows, n_cols, missing_cell_rate]
    required_metrics: [n_rows, n_cols, missing_cell_rate]
    attached_skills: [tabular_profiling]
    fallback_native: gads_profile_dataframe
    # Self-sufficient by necessity: this node LOADS df, so when its retries are exhausted
    # by pre-sandbox codegen failures nothing has executed and `df` does not exist. A
    # fallback_call referencing df would NameError and the safety net would silently no-op
    # in exactly the state it exists for. Downstream nodes may reference their upstream's
    # `produces` variables, which are guaranteed by contract; this one may not.
    fallback_call: "import glob as _g, pandas as _pd; _d = globals().get('df'); _fs = sorted([f for f in _g.glob('*.csv') + _g.glob('*.parquet') if not f.startswith('transformed')]); df = _d if _d is not None else (_pd.read_parquet(_fs[0]) if _fs[0].endswith('.parquet') else _pd.read_csv(_fs[0])); profile = gads_profile_dataframe(df); n_rows = profile['n_rows']; n_cols = profile['n_cols']; missing_cell_rate = profile['missing_cell_rate']"
    postconditions:
      - "n_rows > 0"
      - "n_cols > 0"

  - id: assess_quality
    intent: >
      Assess data quality and store the result in `quality`. Report: the number of duplicate
      rows; columns that are constant or near-constant; columns that look like row
      identifiers (nearly one distinct value per row — but NEVER classify a continuous
      numeric measurement or a datetime as an identifier); columns with a high missing rate;
      and the outlier rate of each numeric column using the 1.5*IQR rule. Also list columns
      that repeat across rows and could serve as a grouping key (an entity such as a customer
      or patient id), in `quality['group_candidates']`. Bind `n_flagged_columns` as a plain
      integer: how many columns carry at least one quality flag.
    depends_on: [profile_dataset]
    worker_tier: T2
    produces: [quality, n_flagged_columns]
    required_metrics: [n_flagged_columns]
    attached_skills: [tabular_profiling]
    fallback_native: gads_assess_quality
    fallback_call: "quality = gads_assess_quality(df, profile); n_flagged_columns = quality['n_flagged_columns']"
    postconditions:
      - "quality is not None"

  - id: univariate_distributions
    intent: >
      Visualise the distribution of the columns: histograms for numeric columns and bar
      charts of the most frequent categories for categorical ones. Group several columns
      into each figure rather than emitting one file per column. Skip columns flagged
      constant or identifier-like in `quality`. If the dataset is large you may sample for
      PLOTTING ONLY — and if you do, say so in the figure title.
      You MUST actually create the image files: call `plt.savefig("<name>.png")` for every
      figure and close it. Describing a figure, or returning its filename in a dict without
      calling savefig, produces nothing and is a failure of this step. Collect the filenames
      you actually saved in `univariate_figures` and bind
      `n_univariate_figures = len(univariate_figures)`.
    depends_on: [assess_quality]
    worker_tier: T2
    produces: [univariate_figures, n_univariate_figures]
    required_metrics: [n_univariate_figures]
    attached_skills: [tabular_visualization, visualization_best_practices]
    postconditions:
      - "n_univariate_figures > 0"

  - id: bivariate_relationships
    intent: >
      Examine relationships between columns. Compute and plot a correlation matrix over the
      numeric columns (Spearman, and Pearson when the relationship looks linear), and flag
      pairs whose absolute correlation exceeds 0.8 as collinear in `correlations`.
      If the SPEC HINTS name a target column, additionally quantify each feature's
      association with that target and plot the strongest ones. If no target is named, skip
      the target-specific part — it is optional, not a failure.
      You MUST actually create the image files with `plt.savefig("<name>.png")`; collect the
      saved filenames in `bivariate_figures` and bind
      `n_bivariate_figures = len(bivariate_figures)`.
    depends_on: [assess_quality]
    worker_tier: T2
    produces: [correlations, bivariate_figures, n_bivariate_figures]
    required_metrics: [n_bivariate_figures]
    attached_skills: [tabular_visualization, visualization_best_practices]
    postconditions:
      - "n_bivariate_figures > 0"

  - id: recommend_transformations
    intent: >
      Decide how each column should be prepared, then hand those decisions to
      `gads_write_transformation_manifest` — do NOT write the JSON yourself.
      Build a dict `decisions` mapping each column name to
      `{"impute": ..., "scale": ..., "encode": ..., "rationale": "..."}`, choosing ONLY from
      these vocabularies and using None when nothing applies:
        impute: median | mean | mode | constant | forward_fill | drop_rows | drop_column | None
        scale:  standard | minmax | robust | log1p | quantile_normal | None
        encode: onehot | ordinal | target | frequency | None
      Justify each choice from the statistics already measured. Never scale or encode a
      datetime column. Leave a declared target column untransformed.
      If the objective or SPEC HINTS indicate the data is destined for a machine-learning
      model AND only one input file was supplied, also build a `split` dict with
      `method` (time_ordered when a time column orders the rows, grouped when a repeated
      entity would otherwise span partitions, stratified for a discrete/imbalanced target,
      random only if none apply), `ratios`, the relevant column, and a `rationale`.
      Otherwise leave `split` as None.
      Then call:
        transformations = gads_write_transformation_manifest(decisions, df=df,
            target_column=<target or None>, split=split, source_file=<the data filename>)
        n_manifest_columns = len(transformations["files"][<the data filename>]["columns"])
      The function writes `eda_transformations.meta.json` in the exact schema the
      downstream applier requires and validates your vocabulary values.
    depends_on: [assess_quality]
    worker_tier: T2
    produces: [transformations, n_manifest_columns]
    required_metrics: [n_manifest_columns]
    attached_skills: [tabular_profiling]
    fallback_native: gads_recommend_transformations
    fallback_call: "transformations = gads_recommend_transformations(df, profile, quality, target_col=None, ml_intent=True); n_manifest_columns = len(list(transformations['files'].values())[0]['columns'])"
    postconditions:
      - "transformations is not None"
      - "n_manifest_columns > 0"

  - id: eda_summary
    intent: >
      Write a short narrative summary of the analysis to `eda_summary.md` and store the text
      in `eda_summary_text`: what the dataset contains, what is wrong with it (missingness,
      duplicates, outliers, unusable columns), the notable relationships found, and what
      should be done to the data before it is modelled. Reference the actual numbers
      measured in the earlier steps — do not restate the task or invent findings.
    depends_on: [recommend_transformations, univariate_distributions, bivariate_relationships]
    worker_tier: T2
    produces: [eda_summary_text]
    attached_skills: []
    fallback_native: gads_eda_summary
    fallback_call: "eda_summary_text = gads_eda_summary(profile, quality, transformations)"
    postconditions:
      - "eda_summary_text is not None"

# ——— INVARIANTS ———
invariants:
  - "Profiling and quality assessment run on the FULL dataset — never sample for them. Sampling is permitted for plotting only, and must be stated in the figure."
  - "A step that produces a file must WRITE that file. Returning a filename, or a dict describing a figure or a manifest, without calling plt.savefig / the manifest writer, produces nothing — the deliverables of this recipe are artifacts on disk, not variables in the kernel."
  - "The manifest is written by gads_write_transformation_manifest, never by hand-rolled json.dump — the applier depends on exact key names (`recommended_impute`, not `impute`) and hand-written schemas drift."
  - "The transformation manifest is written as `eda_transformations.meta.json`; any other `.json` name is registered as an interactive Plotly artifact and will render as a broken figure."
  - "Recommendation values come only from the declared vocabularies — an unknown value makes the manifest unusable by gads_apply_transformations."
  - "Datetime columns are never scaled or encoded; derive calendar features explicitly instead."
  - "A column that is nearly unique is only an identifier if it is not a continuous measurement and not a datetime."
  - "The manifest recommends; it never transforms. Applying it is gads_apply_transformations' job, because the split-then-fit ordering is a leakage guard."
---

## Rationale

GADS had 26 recipes and none of them did exploratory data analysis, even though
`taxonomy.yaml` names `descriptive` as its first intent. "Explore this dataset" therefore
matched no recipe and fell to the drafted lane — the rung where a local model is weakest.

This recipe exists to make EDA a first-class, repeatable deliverable, and to make its output
*reusable*. The deliverable is not only the charts a human reads: it is
`eda_transformations.meta.json`, a machine-readable statement of how each field should be
imputed, scaled and encoded, and — when the data is headed for a model — how it should be
split. A later run consumes that manifest through `gads_apply_transformations` to produce the
transformed dataset, so the judgment made once during exploration is applied consistently
afterwards.

The division of labour follows the project's native-node design rule (approach_docs/019).
*Recommending* a transformation is judgment that depends on the data in front of you, so it
stays model-generated and the model's capability stays measured; the natives are opt-in
fallbacks for the nodes that are documented local-model failure modes (the manifest and the
summary). *Applying* a transformation is invariant mechanics with exactly one right answer,
so it is deterministic, audited native code.

The ordering inside the applier is the point that most deserves the native. Fitting an
imputer or a scaler before splitting leaks held-out statistics into training — the median,
the category map and the scaler centre would all have seen the test rows. The applier
therefore splits first, fits on the training partition alone, and applies those fitted
parameters to every partition, recording them in `transformation_provenance.meta.json` so a
second file can be transformed with identical parameters instead of refitted. That guarantee
belongs in code that always runs the same way, not in a prompt a model may or may not honour.

Postconditions here are structural rather than numeric, which is the shape difference from
every other recipe in the registry. EDA has no accuracy number to hit; what can be checked is
coverage — that every column was profiled, that the manifest parses, that the scalars the
downstream steps depend on are bound.
