---
id: large_dataset_handling
description: "Strategies for processing datasets >50MB or 500K rows in the sandbox without timeout or OOM errors"
triggers: ["large", "100mb", "50mb", "huge", "millions", "800k", "vectorize"]
---
# Context-Aware Large Dataset Handling

When processing datasets exceeding 50MB or 500,000 rows within the sandbox, you MUST apply one of the following strategies depending on the task's mathematical requirements to prevent 60-second timeouts or out-of-memory errors:

## 1. Approximation-Tolerant Tasks (EDA)
For exploratory data analysis, plotting (histograms, scatterplots), or identifying high-frequency categories, exact precision is not required.
- You **MUST** use engine-level sampling to remain fast. 
- **DO NOT** load the full file into Pandas first and then sample (this defeats the purpose and will OOM).
- **Correct DuckDB Pattern**: `duckdb.query("SELECT * FROM read_csv_auto('file.csv') USING SAMPLE 100000 ROWS").to_df()`
- **Correct Polars Pattern**: `pl.scan_csv('file.csv').sample(100000).collect()`
- **Disclosure**: If sampling is used, you MUST explicitly state "Based on a 100K-row sample" in the plot title or printed output.

## 2. Exact-Aggregate Tasks & Transformations
For exact aggregations (global sum, mean, count distinct, group totals) or data transformations/vectorizations where every row must be processed:
- You **MUST NOT** downsample.
- You **MUST** use Out-of-Core engines. Process the full file by querying it directly on disk.
- **Correct DuckDB Pattern**: `duckdb.query("SELECT category, COUNT(*) FROM 'file.csv' GROUP BY category").to_df()`
- **Correct Polars Pattern**: Use LazyFrames with `.sink_parquet('output.parquet')` or `.collect(streaming=True)`.