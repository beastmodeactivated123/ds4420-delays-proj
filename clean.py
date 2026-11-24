# Combines the monthly files, removes cancelled/diverted flights, and adds features
from pathlib import Path

import pandas as pd

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

files = sorted(Path("data/raw").glob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
n_raw = len(df)

# Cancelled and diverted flights have no arrival delay, so drop them
n_cancelled = int(df["Cancelled"].sum())
n_diverted = int(df["Diverted"].sum())
df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
df = df.dropna(subset=["ArrDel15"])

# Share of delay minutes caused by each reason (for the intro, not used by the models)
causes = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
cause_share = df[causes].sum() / df[causes].sum().sum()

# Features known before departure
date = pd.to_datetime(df["FlightDate"])
df = pd.DataFrame({
    "year": date.dt.year,
    "month": date.dt.month,
    "day_of_week": date.dt.dayofweek,
    "dep_hour": (df["CRSDepTime"] // 100) % 24,
    "arr_hour": (df["CRSArrTime"] // 100) % 24,
    "carrier": df["Reporting_Airline"].astype("category"),
    "origin": df["Origin"].astype("category"),
    "dest": df["Dest"].astype("category"),
    "distance": df["Distance"],
    "late": df["ArrDel15"].astype(int),
})
df.to_parquet("data/processed/flights.parquet", index=False)

# Numbers for the report
pairs = df.groupby(["carrier", "origin"], observed=True).size()
summary = f"""Months: {len(files)}
Raw flights: {n_raw:,}
Cancelled removed: {n_cancelled:,}
Diverted removed: {n_diverted:,}
Flights after cleaning: {len(df):,}
Share late (15+ min): {df['late'].mean():.2%}
2025 flights: {(df['year'] == 2025).sum():,}
2026 flights: {(df['year'] == 2026).sum():,}
Carrier-airport pairs: {len(pairs):,}
Pairs with < 100 flights: {(pairs < 100).sum():,}

Share of delay minutes by cause:
{cause_share.map("{:.1%}".format).to_string()}
"""
print(summary)
Path("results/data_summary.txt").write_text(summary)