---
name: "Bayesian Causal Effect of Transaction Amount on Fraud"
datasets:
  - creditcard.csv
target_column: Class
domain: financial fraud / bayesian causal inference
recipe_id: causal_effect.bayesian.pymc
---
Estimate the **causal effect of transaction amount on fraud probability** using
a fully Bayesian approach on the credit card dataset (284,807 rows).

The dataset columns:
- `V1`–`V28`: PCA-anonymised behavioural features (confounders)
- `Amount`: transaction value (treatment driver)
- `Class`: 0 = legitimate, 1 = fraud (binary outcome)

## Instructions

1. **Subsample**: Draw a stratified sample of **5,000 rows** preserving the
   fraud rate (~0.17%). Use
   `pd.concat([fraud_df, legit_df.sample(n, random_state=42)])` where fraud_df
   contains all fraud rows and legit_df fills the rest to 5,000 total.

2. **Engineer treatment**: create binary column `high_amount` = 1 if Amount >
   median(Amount in the full dataset before subsampling), else 0.

3. **Fit Bayesian logistic regression** using Bambi:
   ```
   "Class ~ high_amount + V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10"
   family="bernoulli", link="logit"
   ```
   Use MCMC: `draws=500, tune=300, chains=1, cores=1, progressbar=False, random_seed=42`.

4. **Extract posterior**: from `idata.posterior["high_amount"]`, compute:
   - Posterior mean log-odds effect (store as `ate`)
   - 94% Highest Density Interval
   - `P(effect > 0)` — probability that high-amount transactions increase fraud
   Store the mean as a variable named exactly `ate`.

5. **Visualise posterior**: plot the posterior distribution of the
   `high_amount` coefficient as a histogram with mean, zero-effect reference
   line, and 94% HDI shaded. Save as Figure 1.

6. **Interpret**: compare with the naive (unadjusted) log-odds ratio between
   `high_amount` and `Class`. State whether the Bayesian posterior supports a
   causal effect of transaction amount on fraud, and quantify the uncertainty.
