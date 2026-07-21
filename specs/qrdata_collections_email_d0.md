---
name: "QRData collections email (causal effect with post-treatment traps) [dial D0]"
datasets:
  - qrdata/collections_email.csv
disable_recipes: true
---
Did sending the collections email cause customers to pay more of their debt?

Dataset: 5,000 customers of a collections operation.
- `email`: 1 = received the collections email, 0 = not
- `payments`: amount paid
- `credit_limit`, `risk_score`: customer characteristics on file before the email was sent
- `opened`: whether the customer opened the email
- `agreement`: whether the customer subsequently made a payment agreement
