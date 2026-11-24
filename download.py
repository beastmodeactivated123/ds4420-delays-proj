# Downloads BTS on-time data for 2025 and 2026 into data/raw/
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["FlightDate", "Reporting_Airline", "Tail_Number", "Origin", "Dest",
        "CRSDepTime", "CRSArrTime", "ArrDel15", "Cancelled", "Diverted", "Distance",
        "ArrDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
        "LateAircraftDelay"]  # delay columns are for summary stats only, never model inputs

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")

for year in [2025, 2026]:
    for month in range(1, 13):
        out_file = OUT / f"{year}_{month:02d}.parquet"
        if out_file.exists():
            continue

        print(f"Downloading {year}-{month:02d}")
        req = urllib.request.Request(URL.format(y=year, m=month),
                                     headers={"User-Agent": "Mozilla/5.0"})
        try:
            data = urllib.request.urlopen(req).read()
            z = zipfile.ZipFile(io.BytesIO(data))
        except (urllib.error.HTTPError, zipfile.BadZipFile):
            print("  not released yet, skipping")
            continue

        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(csv_name), usecols=COLS, low_memory=False)
        df.to_parquet(out_file, index=False)
        print(f"  {len(df):,} flights")