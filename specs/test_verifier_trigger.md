---
name: "Amazon Reviews Quick EDA"
datasets:
  - amazon_sample_100.csv
target_column: rating
---
Perform a focused analysis of amazon_sample_100.csv (100 Amazon product reviews).

1. Load the dataset and compute the distribution of the `rating` column. Report how many
   reviews fall into each star rating (1 through 5). Save an interactive bar chart.

2. Compute the mean `helpful_vote` per rating value. Report which rating group receives
   the most helpful votes on average. Save the results as a second bar chart.

Report a Pearson correlation coefficient between `rating` and `helpful_vote` and store
it in a variable named exactly `pearson_r`.
