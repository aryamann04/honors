import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/Users/aryaman/honors/calibration/data"
CSV_PATH = os.path.join(DATA_DIR, "CDS spreads.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "CDS_USD_5Y.parquet")

usecols = [
    "date",
    "ticker",
    "shortname",
    "region",
    "country",
    "avrating",
    "tier",
    "currency",
    "docclause",
    "runningcoupon",
    "tenor",
    "parspread",
    "cdsassumedrecovery",
    "dp",
    "jtd",
]

def build_filtered_parquet_from_csv():
    chunks = []
    for chunk in pd.read_csv(
        CSV_PATH,
        usecols=usecols,
        parse_dates=["date"],
        low_memory=False,
        chunksize=200_000,
    ):
        chunk["parspread"] = pd.to_numeric(chunk["parspread"], errors="coerce")
        chunk["cdsassumedrecovery"] = pd.to_numeric(chunk["cdsassumedrecovery"], errors="coerce")
        chunk["dp"] = pd.to_numeric(chunk["dp"], errors="coerce")
        chunk["jtd"] = pd.to_numeric(chunk["jtd"], errors="coerce")

        chunk = chunk.dropna(subset=["date", "country", "currency", "tenor", "parspread"])
        chunk = chunk[chunk["currency"] == "USD"]
        chunk = chunk[chunk["tenor"] == "5Y"]

        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No USD 5Y CDS rows found in CSV.")

    cds = pd.concat(chunks, ignore_index=True)
    cds = cds.sort_values(["country", "date"]).reset_index(drop=True)
    cds.to_parquet(PARQUET_PATH, index=False)
    return cds

def visualize_CDS_data():
    if os.path.exists(PARQUET_PATH):
        CDS = pd.read_parquet(PARQUET_PATH)
    else:
        CDS = build_filtered_parquet_from_csv()

    CDS["date"] = pd.to_datetime(CDS["date"], errors="coerce")
    CDS["parspread"] = pd.to_numeric(CDS["parspread"], errors="coerce")
    CDS["cdsassumedrecovery"] = pd.to_numeric(CDS["cdsassumedrecovery"], errors="coerce")
    CDS["dp"] = pd.to_numeric(CDS["dp"], errors="coerce")
    CDS["jtd"] = pd.to_numeric(CDS["jtd"], errors="coerce")

    CDS = CDS.dropna(subset=["date", "country", "parspread"])
    CDS = CDS.sort_values(["country", "date"]).reset_index(drop=True)

    print("Filtered dataset shape:", CDS.shape)
    print("\nColumns:")
    print(CDS.columns.tolist())
    print("\nHead:")
    print(CDS.head())

    country_counts = CDS.groupby("country")["date"].count().sort_values(ascending=False)
    print("\nTop countries by observations:")
    print(country_counts.head(20))

    top_countries = country_counts.head(12).index.tolist()
    CDS_top = CDS[CDS["country"].isin(top_countries)].copy()

    pivot = CDS_top.pivot_table(
        index="date",
        columns="country",
        values="parspread",
        aggfunc="mean"
    ).sort_index()

    pivot_monthly = pivot.resample("M").mean()

    summary = CDS.groupby("country")["parspread"].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    ).sort_values("mean", ascending=False)

    print("\nSummary statistics:")
    print(summary.head(20))

    avg_spread = pivot_monthly.mean(axis=1)
    median_spread = pivot_monthly.median(axis=1)

    plt.figure(figsize=(14, 7))
    for c in pivot_monthly.columns:
        plt.plot(pivot_monthly.index, pivot_monthly[c], alpha=0.8, lw=1.2, label=c)
    plt.title("USD 5Y Sovereign CDS Par Spreads")
    plt.xlabel("Date")
    plt.ylabel("Spread (bps)")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(avg_spread, label="Average")
    plt.plot(median_spread, label="Median")
    plt.title("Average Sovereign CDS Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread (bps)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(CDS["parspread"].dropna(), bins=100)
    plt.title("Distribution of CDS Spreads")
    plt.xlabel("Spread (bps)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    latest = pivot_monthly.ffill().iloc[-1].dropna().sort_values(ascending=False)

    plt.figure(figsize=(10, 7))
    plt.barh(latest.index[::-1], latest.values[::-1])
    plt.title("Latest USD 5Y Sovereign CDS Spreads")
    plt.xlabel("Spread (bps)")
    plt.tight_layout()
    plt.show()

    log_changes = np.log(pivot_monthly).diff()
    volatility = log_changes.std().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(volatility.index, volatility.values)
    plt.title("Volatility of Monthly CDS Spread Changes")
    plt.ylabel("Std Dev")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    corr = log_changes.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation of CDS Spread Changes")
    plt.tight_layout()
    plt.show()