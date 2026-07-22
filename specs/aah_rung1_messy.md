---
name: "aah grounding · rung 1 (messy raw data)"
datasets:
  - aah/u.csv
  - aah/hab.csv
  - aah/evt.csv
  - aah/subs.csv
  - aah/spend.csv
  - aah/ref.csv
  - aah/aah_questions.json
domain: consumer habit-tracking app analytics (business KPIs over a warehouse)
recipe_id: analytics.answer_suite.raw
---

You are a data analyst for a habit-tracking app. Today is 2026-07-16; data is complete
through 2026-07-12 (last full ISO week, Mon–Sun), so "last week" = 2026-07-06 to 2026-07-12.
Answer each of the 25 business questions by computing the number from the data — never guess.

The questions and their ids are in `aah_questions.json` ({id: text}). Answer every question by
its id and key the outputs by those exact ids (never renumber to q1/q2).

**Output (required):** write `aah_answers.json` = {id: "answer text with the number + one-line why"}
for all 25 ids, and `metrics.json` = {id: number} for the numeric ones. Answer each question in
its own try/except; always write both files at the end.

## The 25 questions (identical across all grounding rungs; ids authoritative)

**Tier 1 · lookup**
- `t1_value_moments_june`: How many habit completions (value moments) were logged in June 2026?
- `t1_signups_june`: How many new users signed up in June 2026?
- `t1_total_habits`: How many habits have been created in total?
- `t1_spend_june`: What was our total marketing spend in June 2026?
- `t1_ios_value_moments_june`: How many value moments came from the iOS platform in June 2026?

**Tier 2 · filtered / segmented**
- `t2_web_value_moments_june`: How many value moments came from the web platform in June 2026?
- `t2_referral_signups_q2`: How many users signed up through the referral channel in the second quarter of 2026 (April–June)?
- `t2_paid_search_spend_q2`: How much did we spend on paid search in the second quarter of 2026 (April–June)?
- `t2_americas_value_moments_june`: How many value moments came from users in the Americas region in June 2026?
- `t2_active_annual_subs`: How many currently-active subscriptions are on the annual plan?

**Tier 3 · governed metric**
- `t3_mrr`: What is our current MRR (monthly recurring revenue)?
- `t3_arpu`: What is our ARPU (average monthly recurring revenue per paying user)?
- `t3_active_users_last_week`: How many active users did we have last week?
- `t3_power_users`: How many power users did we have in June 2026?
- `t3_activation_rate_june`: What was the activation rate for users who signed up in June 2026?

**Tier 4 · business knowledge**
- `t4_apac_value_moments_q2`: How many value moments came from the APAC region in the second quarter of 2026 (April–June)?
- `t4_real_signups_june`: How many signups came from real acquisition channels in June 2026?
- `t4_real_acquisition_spend_june`: How much did we spend on real acquisition channels in June 2026?
- `t4_retention_trend`: How has our retention been trending over the last couple of months?
- `t4_business_health`: Is the app healthy right now — how would you tell?

**Tier 5 · diagnostic / root-cause**
- `t5_why_drop`: Weekly value moments dropped last week. What caused it?
- `t5_engagement_drop`: Engagement fell last week. Which part of the business is responsible?
- `t5_not_breadth`: Did our active user base shrink last week, or is something else going on with value moments?
- `t5_masked_by_growth`: Value moments came in soft last week even though we added users. What happened?
- `t5_which_lever`: Which part of the North Star weakened last week — breadth, frequency, or depth?
