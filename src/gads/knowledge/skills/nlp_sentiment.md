---
id: nlp_sentiment
description: "Sentiment analysis patterns: textblob polarity or NLTK VADER — no transformers, no external APIs"
triggers: ["sentiment", "textblob", "vader", "polarity", "subjectivity", "emotion", "opinion mining"]
---
# Sentiment Analysis

Use `textblob` (simple) or NLTK VADER (better for social-media/informal text). `transformers` and `spacy` are NOT installed.

```python
# Option A: textblob (polarity score -1 to +1)
from textblob import TextBlob
df['polarity'] = df['text'].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)

# Option B: NLTK VADER
import nltk
nltk.download('vader_lexicon', quiet=True)  # data download, allowed
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['compound'] = df['text'].apply(lambda t: sia.polarity_scores(str(t))['compound'])
```

`.apply` on a text Series is acceptable for scoring; avoid nested Python loops over rows.
