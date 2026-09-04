"""Stage an M4-Hourly subset as a forecasting benchmark corpus.

M4 (Makridakis et al., 2018 — https://github.com/Mcompetitions/M4-methods) is the standard
public forecasting competition set. The Hourly split is used here because it carries strong
daily seasonality (m=24), which gives the seasonal-naive baseline real teeth: a forecaster
that ignores seasonality cannot beat it by luck.

Writes into $GADS_DATASETS_ROOT/m4:
    m4_hourly_train.csv    item_id, timestamp, target   (long format — what GADS sees)
    m4_hourly_test.csv     the official held-out horizon (NOT given to GADS; external check)
    PROVENANCE.md          source, subset rule, and the reference values below

Also prints the deterministic reference values for the benchmark's expected.json. These are
pure data functions — no model, no randomness — so they are exact-matchable:

    n_series, n_rows, naive_mae, seasonal_naive_mase

`seasonal_naive_mase` is 1.0 by construction (MASE is defined against the seasonal-naive
in-sample error) and is emitted as the yardstick the recipe's `best_model_mase` must beat.

Usage:  PYTHONPATH=src uv run python scripts/stage_m4_hourly.py [--n-series 50]
Network required (host only; the sandbox is offline by design).
"""
import argparse
import csv
import io
import os
import statistics
import urllib.request
from datetime import datetime, timedelta

BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
TRAIN_URL = f"{BASE}/Train/Hourly-train.csv"
TEST_URL = f"{BASE}/Test/Hourly-test.csv"
SEASON = 24          # hourly data, daily seasonality
# M4's official horizon for the Hourly split. Recorded for the external check only — the
# recipe derives its own prediction_length from the data and is not forced to match.
M4_HOURLY_HORIZON = 48


def fetch_wide(url):
    """M4 ships wide: row = one series, col V1 = id, V2..Vn = values (ragged, blank-padded)."""
    req = urllib.request.Request(url, headers={"User-Agent": "GADS-benchmark-staging/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        text = r.read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    out = {}
    for row in rows[1:]:
        if not row:
            continue
        sid, vals = row[0], [v for v in row[1:] if v not in ("", "NA")]
        out[sid] = [float(v) for v in vals]
    return out


def seasonal_naive_mae(values, season):
    """In-sample MAE of the seasonal-naive predictor — the denominator of MASE."""
    diffs = [abs(values[i] - values[i - season]) for i in range(season, len(values))]
    return sum(diffs) / len(diffs) if diffs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=50,
                    help="how many series to stage (M4 Hourly has 414)")
    args = ap.parse_args()

    root = os.getenv("GADS_DATASETS_ROOT", "/home/joergf/datasets")
    out = os.path.join(root, "m4")
    os.makedirs(out, exist_ok=True)

    print(f"fetching {TRAIN_URL} ...")
    train = fetch_wide(TRAIN_URL)
    print(f"fetching {TEST_URL} ...")
    test = fetch_wide(TEST_URL)
    print(f"  M4 Hourly: {len(train)} series")

    # Deterministic subset: the first n by M4's own numeric id order, so the corpus is
    # reproducible from this script alone and does not depend on dict ordering.
    ids = sorted(train, key=lambda s: int(s[1:]))[:args.n_series]

    # M4 gives no usable per-series wall-clock start for Hourly, so timestamps are
    # synthesized from a fixed epoch at hourly frequency. Only the SPACING matters to a
    # forecaster; anchoring every series to the same epoch keeps the file reproducible.
    epoch = datetime(2015, 1, 1)

    train_rows, test_rows = [], []
    for sid in ids:
        vals = train[sid]
        for i, v in enumerate(vals):
            train_rows.append({"item_id": sid,
                               "timestamp": (epoch + timedelta(hours=i)).isoformat(sep=" "),
                               "target": v})
        for j, v in enumerate(test.get(sid, [])):
            test_rows.append({"item_id": sid,
                              "timestamp": (epoch + timedelta(hours=len(vals) + j)).isoformat(sep=" "),
                              "target": v})

    def write(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "timestamp", "target"])
            w.writeheader()
            w.writerows(rows)

    write(f"{out}/m4_hourly_train.csv", train_rows)
    write(f"{out}/m4_hourly_test.csv", test_rows)

    # ---- deterministic reference values for expected.json ----
    targets = [r["target"] for r in train_rows]
    mean = sum(targets) / len(targets)
    # The recipe defines naive_mae as "mean absolute deviation of the target from its own
    # mean". Computed GLOBALLY here. The recipe's wording does not say global vs per-series,
    # which is a genuine ambiguity — see the benchmark notes.
    naive_mae = sum(abs(t - mean) for t in targets) / len(targets)
    lengths = [len(train[s]) for s in ids]
    med_len = statistics.median(lengths)
    snaive = [seasonal_naive_mae(train[s], SEASON) for s in ids]

    print(f"\nstaged -> {out}")
    print(f"  train {len(train_rows)} rows / {len(ids)} series | test {len(test_rows)} rows")
    print(f"  series length: min {min(lengths)} median {med_len:.0f} max {max(lengths)}")
    print("\n--- deterministic reference values (expected.json) ---")
    print(f"  n_series           = {len(ids)}")
    print(f"  n_rows             = {len(train_rows)}")
    print(f"  naive_mae (global) = {naive_mae:.10f}")
    print(f"  prediction_length  = {max(1, int(round(0.10 * med_len)))}   "
          f"(recipe rule: ~10% of median series length, min 1)")
    print(f"  seasonal-naive MAE = {statistics.mean(snaive):.6f} (mean over series, m={SEASON})")
    print(f"  => best_model_mase must be < 1.0 to beat seasonal-naive")
    print(f"  M4 official horizon (external check only) = {M4_HOURLY_HORIZON}")

    with open(f"{out}/PROVENANCE.md", "w") as f:
        f.write(
            "# M4-Hourly subset — provenance\n\n"
            f"Source: {BASE} (M4 Competition, Makridakis et al. 2018).\n"
            f"Subset: first {len(ids)} series by numeric id — deterministic, reproducible\n"
            "via `scripts/stage_m4_hourly.py`.\n\n"
            f"- train: {len(train_rows)} rows / {len(ids)} series (long format)\n"
            f"- test:  {len(test_rows)} rows — the official held-out horizon "
            f"({M4_HOURLY_HORIZON} steps/series). NOT provided to GADS.\n"
            "- timestamps are synthesized hourly from 2015-01-01; only spacing is meaningful.\n\n"
            "## Reference values (pure data functions)\n\n"
            f"| quantity | value |\n|---|---|\n"
            f"| n_series | {len(ids)} |\n| n_rows | {len(train_rows)} |\n"
            f"| naive_mae (global MAD) | {naive_mae:.10f} |\n"
            f"| seasonal-naive MAE (mean over series, m={SEASON}) | {statistics.mean(snaive):.6f} |\n")
    print(f"  wrote {out}/PROVENANCE.md")


if __name__ == "__main__":
    main()
