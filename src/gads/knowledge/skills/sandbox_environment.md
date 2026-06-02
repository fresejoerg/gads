---
id: sandbox_environment
description: "Sandbox constraints: pickle BLOCKED (use joblib), imblearn NOT installed. AutoML: autogluon.tabular + autogluon.timeseries available. Causal stack: dowhy, econml, causalml, pymc, bambi."
triggers: ["available packages", "install", "import error", "sentiment", "textblob", "spacy", "nltk", "what packages", "ModuleNotFoundError", "feature engineering", "nlp", "text features", "text analysis", "classification"]
---
# Sandbox Environment

The IPython kernel sandbox has a **fixed set of pre-installed packages**. You CANNOT install new packages at runtime (no `pip install`, no `subprocess` installs). Only use packages from this list.

## Available Packages

| Category | Packages |
|---|---|
| Data | pandas, numpy, polars, pyarrow, duckdb |
| ML/Modeling | scikit-learn, joblib, torch, lightgbm, xgboost, shap |
| **AutoML** | **autogluon.tabular** (`TabularPredictor`), **autogluon.timeseries** (`TimeSeriesPredictor`) |
| Causal Inference | dowhy, econml, causalml, causallearn (`causal-learn`), linearmodels, statsmodels |
| Bayesian Causal | pymc, arviz, bambi, pycausalimpact (`from causalimpact import CausalImpact`), pgmpy, CausalInference |
| NLP/Embeddings | sentence-transformers (`all-MiniLM-L6-v2` cached), nltk, textblob |
| Visualization | matplotlib, seaborn, plotly, kaleido, networkx |
| HTTP/Async | httpx, fastapi, uvicorn |
| Notebook | jupyter_client, ipykernel, nbformat |

For AutoML (classification/regression/forecasting): see skill `autogluon_tabular`.
For causal inference patterns see skills: `causal_inference_dowhy`, `causal_ml_econml`, `causal_discovery`, `bayesian_causal_pymc`, `causal_impact_timeseries`.

## NOT Available (do NOT import)

- `spacy`, `transformers` (huggingface), `gensim`, `langchain`
- `catboost` — blocked
- `imbalanced-learn` / `imblearn` — **NOT installed**. For class imbalance use `class_weight='balanced'` in sklearn models, or stratified subsampling. Never `from imblearn.over_sampling import SMOTE`.
- `vaderSentiment` standalone — use `nltk.sentiment.vader` instead
- `pickle` — **BLOCKED by sandbox security policy**. Use `joblib` for model serialization instead.

## Sentiment Analysis Pattern

Use `textblob` (preferred, simple) or `nltk` VADER:

```python
# Option A: textblob (simple polarity score -1 to +1)
from textblob import TextBlob
polarity = TextBlob(text).sentiment.polarity
subjectivity = TextBlob(text).sentiment.subjectivity

# Option B: NLTK VADER (better for social-media/informal text)
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores(text)  # {'neg': .., 'neu': .., 'pos': .., 'compound': ..}
```

## Gradient Boosting — All Options Available

`lightgbm` and `xgboost` are **working** (libgomp1 is installed). All three options are available:

```python
# Option A: sklearn (always safe, no extra deps)
from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Option B: LightGBM
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Option C: XGBoost
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)
```

For feature importance via a Random Forest:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
```

## Model Serialization

**NEVER use `pickle`** — it is blocked by sandbox security. Use `joblib`:

```python
import joblib
joblib.dump(model, 'model.joblib')   # save
model = joblib.load('model.joblib')  # load
```

## Vectorization Rules (avoid timeouts on large datasets)

For datasets with 50K+ rows, **never compute row-by-row in a Python loop**. Use vectorized pandas/numpy:

```python
# Jaccard similarity — vectorized via sets on pre-split tokens
tokens_a = df['response_a'].str.lower().str.split()
tokens_b = df['response_b'].str.lower().str.split()
df['jaccard_sim'] = [
    len(set(a) & set(b)) / len(set(a) | set(b)) if (set(a) | set(b)) else 0.0
    for a, b in zip(tokens_a, tokens_b)
]

# TTR (Type-Token Ratio) — apply is acceptable; avoid nested loops
df['ttr_a'] = tokens_a.apply(lambda t: len(set(t)) / len(t) if t else 0.0)
```

## CRITICAL RULES

1. **NO INSTALLS**: Never write `pip install`, `subprocess.run(["pip", ...])`, or `%pip install`. If a package is not in the list above, it is not available.
2. **USE ALTERNATIVES**: For any package NOT in the list, find an equivalent from the available list (e.g., use `sklearn.ensemble.GradientBoostingClassifier` instead of `xgboost`).
3. **NLTK DOWNLOADS**: nltk corpus downloads (e.g., `nltk.download('vader_lexicon', quiet=True)`) are allowed since they fetch small data files, not code packages.
4. **NO PICKLE**: Use `joblib` for any model or object serialization.
