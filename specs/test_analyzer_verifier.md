---
name: "Amazon Reviews EDA + Rating Predictor"
datasets:
  - amazon_sample_100.csv
target_column: rating
---
Analyze the Amazon product reviews in amazon_sample_100.csv. The dataset has columns:
rating, title, text, asin, user_id, timestamp, helpful_vote, verified_purchases, target.

Perform the following analyses:

1. Exploratory analysis: show the distribution of `rating` values and the `verified_purchases`
   breakdown. Report what fraction of reviews are verified.

2. Text feature engineering: compute word count and character count for the `text` column.
   Also compute a Type-Token Ratio (unique words / total words) for each review.

3. Sentiment analysis: use NLTK VADER to compute a compound sentiment score for each review
   text. Store results in a column called `sentiment_score`.

4. Rating prediction: train a Gradient Boosting classifier to predict `rating` (treat as
   a 3-class problem: 1-2 = negative, 3 = neutral, 4-5 = positive) using the engineered
   features (word_count, char_count, ttr, sentiment_score, helpful_vote).
   Report macro-F1 accuracy. Store the macro-F1 value in a variable named exactly `macro_f1`.

5. Feature importance: identify which of the five features most strongly predicts rating.
   Save a bar chart of feature importances.
