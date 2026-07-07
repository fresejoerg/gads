---
id: sandbox_environment
description: "Sandbox constraints: fixed package set (no pip install), pickle BLOCKED (use joblib), imblearn NOT installed"
triggers: ["available packages", "install", "import error", "what packages", "ModuleNotFoundError"]
---
# Sandbox Environment

The IPython kernel sandbox has a **fixed set of pre-installed packages**. You CANNOT install new packages at runtime (no `pip install`, no `subprocess` installs, no `%pip`).

## Available Packages

| Category | Packages |
|---|---|
| Data | pandas, numpy, polars, pyarrow, duckdb |
| ML/Modeling | scikit-learn, joblib, torch, lightgbm, xgboost, shap |
| AutoML | autogluon.tabular (`TabularPredictor`), autogluon.timeseries (`TimeSeriesPredictor`) |
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
