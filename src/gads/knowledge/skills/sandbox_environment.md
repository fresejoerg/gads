---
id: sandbox_environment
triggers: ["available packages", "install", "import error", "sentiment", "textblob", "spacy", "nltk", "what packages", "ModuleNotFoundError", "feature engineering", "nlp", "text features", "text analysis", "classification"]
---
# Sandbox Environment

The IPython kernel sandbox has a **fixed set of pre-installed packages**. You CANNOT install new packages at runtime (no `pip install`, no `subprocess` installs). Only use packages from this list.

## Available Packages

| Category | Packages |
|---|---|
| Data | pandas, numpy, polars, pyarrow, duckdb |
| ML/Modeling | scikit-learn, joblib, torch, lightgbm, xgboost, shap |
| NLP/Embeddings | sentence-transformers (`all-MiniLM-L6-v2` cached), nltk, textblob |
| Visualization | matplotlib, seaborn, plotly, kaleido |
| HTTP/Async | httpx, fastapi, uvicorn |
| Notebook | jupyter_client, ipykernel, nbformat |

## NOT Available (do NOT import)

- `spacy`, `transformers` (huggingface), `gensim`, `langchain`
- `catboost` — use `lightgbm` or `xgboost` instead
- `vaderSentiment` standalone — use `nltk.sentiment.vader` instead

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

## CRITICAL RULES

1. **NO INSTALLS**: Never write `pip install`, `subprocess.run(["pip", ...])`, or `%pip install`. If a package is not in the list above, it is not available.
2. **USE ALTERNATIVES**: For any package NOT in the list, find an equivalent from the available list (e.g., use `sklearn.ensemble.GradientBoostingClassifier` instead of `xgboost`).
3. **NLTK DOWNLOADS**: nltk corpus downloads (e.g., `nltk.download('vader_lexicon', quiet=True)`) are allowed since they fetch small data files, not code packages.
