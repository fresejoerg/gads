"""Stage the AirPassengers series as a univariate forecasting benchmark corpus.

AirPassengers (Box & Jenkins, *Time Series Analysis: Forecasting and Control*, 1976, Series G)
is the textbook univariate series: monthly totals of international airline passengers in
thousands, 1949-01 to 1960-12 (144 observations), with a clear upward trend and
multiplicative yearly seasonality (m=12). It is the canonical test case for seasonal models;
Box & Jenkins's "airline model" (seasonal ARIMA(0,1,1)(0,1,1)12 on log values) is named after it.

The values are taken from the copy shipped inside `statsforecast` (statsforecast.utils.
AirPassengersDF), so staging needs no network. The 144 values are fixed and well known, and
the checksum below pins them.

Writes into $GADS_DATASETS_ROOT/airpassengers (default /home/joergf/datasets):
    air_passengers.csv   month (YYYY-MM-01), passengers   — what GADS sees
    PROVENANCE.md        source, transformation, checksum

Usage:  uv run --no-project --with statsforecast==2.0.1 python scripts/stage_air_passengers.py
"""
import hashlib
import os

from statsforecast.utils import AirPassengersDF

ROOT = os.path.join(os.environ.get("GADS_DATASETS_ROOT", "/home/joergf/datasets"), "airpassengers")
EXPECTED_SUM = 40363.0   # sum of the 144 Series G values; guards against a changed upstream copy


def main():
    df = AirPassengersDF.copy()
    assert len(df) == 144 and float(df["y"].sum()) == EXPECTED_SUM, "upstream AirPassengers changed"
    # statsforecast stamps month-END dates; month-start is the conventional label for monthly totals.
    df["month"] = df["ds"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    out = df[["month", "y"]].rename(columns={"y": "passengers"})
    out["passengers"] = out["passengers"].astype(int)

    os.makedirs(ROOT, exist_ok=True)
    csv_path = os.path.join(ROOT, "air_passengers.csv")
    out.to_csv(csv_path, index=False)
    sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

    with open(os.path.join(ROOT, "PROVENANCE.md"), "w") as f:
        f.write(f"""# AirPassengers — provenance

Source: Box, G.E.P. & Jenkins, G.M. (1976), *Time Series Analysis: Forecasting and Control*,
Series G — monthly international airline passengers (thousands), 1949-01 … 1960-12.
Copy used: `statsforecast.utils.AirPassengersDF` (statsforecast 2.0.1), staged by
`scripts/stage_air_passengers.py` in the GADS repo.

- {len(out)} rows, one series, no gaps, no missing values
- columns: `month` (first day of the month, ISO date), `passengers` (integer, thousands)
- transformation: month-end dates from statsforecast relabelled to month-start; values unchanged
- sum of values: {EXPECTED_SUM:.0f}
- sha256(air_passengers.csv): `{sha}`
""")
    print(f"wrote {csv_path} ({len(out)} rows) sha256={sha}")
    print(out.head(3).to_string(index=False))
    print(out.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
