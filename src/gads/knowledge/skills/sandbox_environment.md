---
id: sandbox_environment
description: "Sandbox constraints: fixed package set (no pip install), pickle BLOCKED (use joblib), imblearn NOT installed; kernel code-style discipline (global scope, persistent variables, plain prints)"
triggers: ["available packages", "install", "import error", "what packages", "ModuleNotFoundError"]
---
# Sandbox Environment

The IPython kernel sandbox has a **fixed set of pre-installed packages**. You CANNOT install new packages at runtime (no `pip install`, no `subprocess` installs, no `%pip`).

## Available Packages

| Category | Packages |
|---|---|
| Data | pandas, numpy, polars, pyarrow, duckdb |
| ML/Modeling | scikit-learn, joblib, torch, lightgbm, xgboost, shap, skore |
| AutoML | autogluon.tabular (`TabularPredictor`), autogluon.timeseries (`TimeSeriesPredictor`) |
| Survival | lifelines (`CoxPHFitter`, `KaplanMeierFitter`), scikit-survival (`sksurv`: `RandomSurvivalForest`, `Surv`) |
| Causal | dowhy, econml, causalml, causallearn, linearmodels, statsmodels, pymc, arviz, bambi, pycausalimpact, pgmpy |
| NLP | sentence-transformers (`all-MiniLM-L6-v2` cached), nltk, textblob |
| Visualization | matplotlib, seaborn, plotly, kaleido, networkx |

## NOT Available (do NOT import)

- `pickle` — **BLOCKED by security policy**. Serialize with `joblib` instead: `joblib.dump(obj, 'file.joblib')` / `joblib.load('file.joblib')`.
- `imbalanced-learn` / `imblearn` — use `class_weight='balanced'` in sklearn models instead. Never `from imblearn.over_sampling import SMOTE`.
- `spacy`, `transformers` (huggingface), `gensim`, `langchain`, `catboost`
- `vaderSentiment` standalone — use `nltk.sentiment.vader` instead.

## Rules

1. If a package is not listed above, it is NOT available — pick an equivalent from the list.
2. nltk corpus downloads (e.g. `nltk.download('vader_lexicon', quiet=True)`) are allowed — they fetch data files, not code.
3. For 50K+ rows, never compute row-by-row in a pure-Python loop — use vectorized pandas/numpy operations (`.str` methods, `.apply` on a Series at most).

## Kernel & Code-Style Discipline

Tasks run sequentially in ONE persistent IPython kernel — later tasks read earlier tasks'
variables directly:

1. **Global scope only.** Never wrap code in `def main()` or any function — variables
   inside a function are invisible to the next task.
2. **Variables persist.** If a previous task loaded `df` or fitted `predictor`, use it —
   do NOT call `pd.read_csv()` again or re-fit.
3. **Plain `print()` for scalar evidence** (`print("metric:", value)`) — avoid f-strings
   with complex expressions inside `{}`; compute into a simple variable first.
4. **Never print lines starting with `GADS_`** — prefixes like `GADS_INSIGHTS_JSON:` are
   reserved harness sentinels.
5. **Long calls on one logical line** inside the parentheses — no backslash continuations.
6. **matplotlib is headless**: call `matplotlib.use('Agg')` before `pyplot`; save figures
   to files (never `plt.show()`).
7. `gads_emit_insight(...)`, `gads_calibrate_threshold(...)`, `gads_causal_estimate_ate(...)`
   are pre-defined in the kernel — call them directly, never import them.
