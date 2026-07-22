---
id: aah_star_schema
description: "aah grounding rung 2 — the clean star schema over the messy habit-app CSVs. Provides a ready DuckDB connection (aah_star.connect) with dim_/fct_ views, their columns, and SQL patterns. Cleans names/enums/types only; carries NO business rules (those are rungs 4-5)."
triggers: ["aah_star_schema", "aah_star", "habit app star schema"]
---
# aah Star Schema — clean dimensional views over the raw habit-app tables

This is the **rung-2 grounding** for the aah analytics benchmark: the modelling an
analytics engineer does over the raw application database. It fixes cryptic names,
inconsistent enums (`ios`/`iOS`/`IOS`), integer status codes, and the opens-vs-completes
grain trap. It does **not** encode business rules (who counts as "active", the APAC
launch cutoff, the partnerships test channel) — those are deliberately absent here.

## Use the prebuilt star, do not query the raw tables

The messy raw CSVs (`u`, `hab`, `evt`, `subs`, `spend`, `ref`) and `aah_star.py` are in
the working directory. Build the clean views and query those:

```python
from aah_star import connect
con = connect(".")          # in-memory DuckDB: raw CSVs loaded + star views created
df = con.execute("SELECT count(*) FROM fct_value_moments").df()
```

Fixed clock: **today is 2026-07-16; data is complete through 2026-07-12** (the last full
ISO week, Mon–Sun). "Last week" = 2026-07-06 .. 2026-07-12.

## The star (query these views, never `u`/`evt`/`subs`/…)

- **dim_users**(`user_id`, `signup_ts`, `signup_date`, `channel`, `country`, `region`,
  `platform`, `is_internal`) — enums normalized; `region` ∈ {Americas, EMEA, APAC, Other};
  `platform` ∈ {ios, android, web, unknown}; `channel` ∈ {paid_search, referral,
  content_seo, partnerships, organic}. `is_internal` flags test/employee accounts.
- **dim_habits**(`habit_id`, `user_id`, `name`, `category`, `created_date`, `archived_date`, `is_archived`).
- **fct_value_moments**(`value_moment_id`, `user_id`, `habit_id`, `completed_ts`,
  `completed_date`, `week`, `source`) — a value moment = a *completed* habit. Opens and
  reminder clicks are already excluded (the grain-trap fix).
- **fct_reminders**(`reminder_id`, `user_id`, `reminded_ts`, `reminded_date`, `week`).
- **fct_subscriptions**(`subscription_id`, `user_id`, `plan` ∈ {monthly, annual},
  `billed_amount`, `started_date`, `ended_date`, `status`, `is_active`) — one row per
  term; multiple rows per user are normal (churn + resubscribe). `billed_amount` is the
  amount billed (annual plans are billed as a yearly lump — normalizing to monthly is a
  metric decision, not done here).
- **fct_marketing_spend**(`spend_date`, `channel`, `spend`).
- **fct_referrals**(`referral_id`, `referrer_user_id`, `referred_user_id`, `referred_ts`, `status`).

## SQL patterns

```python
# lookup: value moments in June 2026
con.execute("""SELECT count(*) FROM fct_value_moments
               WHERE completed_date BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'""").fetchone()[0]

# segment: value moments from iOS users in June
con.execute("""SELECT count(*) FROM fct_value_moments f JOIN dim_users u USING(user_id)
               WHERE u.platform='ios' AND f.completed_date BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'""").fetchone()[0]

# active annual subscriptions (use is_active, not status text)
con.execute("SELECT count(*) FROM fct_subscriptions WHERE is_active AND plan='annual'").fetchone()[0]
```

## What this rung does NOT give you

No governed metric definitions (MRR normalization, the "active user" population, power-user
threshold, activation window) and no business context (APAC launch date, partnerships being
a test channel). Compute what the question literally asks over the clean star. If a question
needs a rule you don't have, answer with the straightforward reading of the data.
