# GADS User Guide: Optimizing Your Research Instructions

GADS is designed to handle the "heavy lifting" of Data Science, but like any expert team, it performs best when given clear, unambiguous instructions. You don't need to be a data scientist to get professional results; you just need to be specific about **what** you have and **what** you want to see.

Follow these best practices to ensure GADS delivers high-quality narratives and functional visualizations every time.

---

## 1. Explicitly Map Your Data (The "Schema Rule")
GADS automatically scans your dataset, but local models can sometimes "hallucinate" or guess column names if your objective is too vague.

*   **Bad**: "Analyze my sales data."
*   **Good**: "Analyze the sales data in `transactions.csv`. Specifically, use the `amount` column for values and `created_at` for the date. The `customer_type` column identifies if the user is a subscriber."

**Why it matters**: This prevents the system from guessing names like `price` or `date`, which leads to "Column Not Found" errors.

---

## 2. Define Your Output Requirements (The "Format Rule")
Tell GADS exactly how you want the final findings structured. If you need specific metrics or a specific tone, say so.

*   **Best Practice**: Request specific aggregations or segments.
*   **Example**: "When calculating the average satisfaction, please group the results by `region`. I need the final narrative to highlight the top 3 and bottom 3 regions specifically."

---

## 3. Describe Your Vision (The "Visualization Rule")
GADS defaults to professional, interactive Plotly charts. You can guide the system to create exactly what you need for your presentation or report.

*   **Bad**: "Show some charts."
*   **Good**: "Create an **interactive line chart** showing the trend of `rating` over `timestamp`. Also, include a **bar chart** comparing the frequency of the top 5 `product_categories`."

**Professional Tip**: You can also request specific layout details, such as "Set the height of all charts to 500 pixels" or "Use a log scale for the Y-axis."

---

## 4. Signal "Big Data" Intensity
If you know your file is massive (e.g., over 500MB or millions of rows), mentioning this in your instruction helps the Project Manager choose the correct "out-of-core" engines (DuckDB or Polars).

*   **Example**: "The dataset `reviews_large.csv` is over 2GB. You **must** use DuckDB or Polars streaming to process it. Do not attempt to load it entirely into Pandas."

---

## 5. Understanding "Bypassed" Tasks & Handover Bundles
For very heavy computations (e.g., training a large Random Forest on millions of rows), GADS uses a **Predictive Runtime Oracle**.
*   **Safety Threshold**: If a task is estimated to take more than **5 minutes**, the system will mark it as **"Bypassed"** in the timeline.
*   **Handover ZIP**: Instead of failing, GADS generates a **Reproducible Project Bundle**. You can download this ZIP from the dashboard. It contains your cleaned data artifacts, a standalone Python script, and instructions to run the heavy training code on your local machine.
*   **Why?**: This prevents your browser session or the server from hanging, while still giving you the exact code you need to finish the job.

---

## 6. Reproducible Spec-Based Launches

For recurring or well-defined workflows, use a **spec file** (`specs/*.md`) instead of typing an objective each time.  Spec files are Markdown with YAML frontmatter:

```yaml
---
name: "My Project"
datasets:
  - folder/file.csv
save_model: true
---
Your objective text goes here.
```

Key frontmatter options:

| Key | Purpose |
|-----|---------|
| `datasets` | Files to copy into the workspace (paths relative to `GADS_DATASETS_ROOT`) |
| `recipes` | Knowledge-base recipe files to load as planning priors |
| `target_column` | Tells the Planner which column to predict |
| `save_model` | `true` → always serialise the trained classifier to `model.joblib` after a successful run |

Launch via `POST /projects/from-spec` or the **"Launch from Spec"** button in the UI.

---

## 7. Follow-up and Iteration
GADS features **Durable Project Memory**. This means you can build on previous results without starting over.

*   **Best Practice**: Reference variables or findings from earlier in the session.
*   **Example**: "Based on the `df_cleaned` we created in the previous step, now perform a correlation analysis between `word_count` and `rating`."

---

## The Perfect Instruction Checklist
Before clicking **"Launch Workflow"**, ask yourself:
1. [ ] Did I mention the **filename**?
2. [ ] Did I name the **specific columns** I want to analyze?
3. [ ] Did I specify which **charts** I want (Line, Bar, Scatter)?
4. [ ] Did I mention any **filters** (e.g., "Only consider 'Verified Purchases'")?
5. [ ] Is there a **Big Data** warning if the file is huge?

---

### Example Comparison

**❌ Vague Instruction:**
> "Look at the Amazon reviews and tell me what people like."
> *Result: Might sample incorrectly, guess column names, or produce generic charts.*

**✅ Precision Instruction:**
> "Analyze `amazon_reviews.csv`. Perform a thematic analysis by extracting themes from the `review_text` column. Correlate these themes with the `star_rating` to see which ones drive satisfaction. Use `timestamp` for a temporal trend line chart. Ensure all visualizations are interactive Plotly charts."
> *Result: High-fidelity code, correct column usage, and a professional, data-grounded research report.*
