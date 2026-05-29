---
name: "Causal Effect of Transaction Amount on Fraud"
datasets:
  - creditcard.csv
target_column: Class
domain: financial fraud
recipe_id: causal_effect.observational.dowhy
---
Does the size of a credit card transaction causally increase the probability
of fraud, after controlling for the cardholder's underlying behavioural patterns?

Dataset: 284,807 credit card transactions.
- `Amount`: transaction value in currency units (continuous — the treatment of interest)
- `Class`: 0 = legitimate, 1 = fraud (binary outcome, severely imbalanced ~0.17%)
- `V1`–`V28`: PCA-anonymised behavioural features that may confound the relationship
- `Time`: seconds elapsed since the first transaction (temporal — not a confounder)

Estimate the Average Treatment Effect (ATE) of high-value transactions on
fraud probability, with refutation results to validate the causal claim.
