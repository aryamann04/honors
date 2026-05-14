"""
build_panel.py
==============
Merge intensities + multi-tenor CDS spreads into a clean monthly panel.

Inputs
------
  calibration/processed_data/intensities.csv
  calibration/data/CDS spreads.csv

Output
------
  calibration/processed_data/panel.csv

Columns: country, date, lambda_total, lambda_global, lambda_country,
         cds_spread_1y, cds_spread_2y, cds_spread_3y, cds_spread_5y,
         cds_spread_7y, cds_spread_10y

CLI
---
    python -m calibration.build_panel
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

INTENSITIES_PATH  = os.path.join(BASE_DIR, "processed_data", "intensities.csv")
CAPE_PANEL_PATH   = os.path.join(BASE_DIR, "raw_data", "country_cape_panel.csv")
CDS_CSV_PATH      = os.path.join(BASE_DIR, "raw_data", "CDS spreads.csv")
OUT_PATH          = os.path.join(BASE_DIR, "processed_data", "panel.csv")

# Map CDS dataset country names → CAPE ISO-2 codes
COUNTRY_NAME_MAP: dict[str, str] = {
    "United States":          "US",
    "United Kingdom":         "UK",
    "Japan":                  "JP",
    "Germany":                "DE",
    "Argentina":              "AR",
    "Korea (Republic of)":    "KR",
    "Russian Federation":     "RU",
    "Mainland China":         "CN",
    "Italy":                  "IT",
    "Mexico":                 "MX",
    "Spain":                  "ES",
    "South Africa":           "ZA",
    "France":                 "FR",
    "Brazil":                 "BR",
    "Australia":              "AU",
    "Indonesia":              "ID",
    "Malaysia":               "MY",
    "Sweden":                 "SE",
    "Netherlands":            "NL",
    "Thailand":               "TH",
    "Turkey":                 "TR",
    "India":                  "IN",
    "Chile":                  "CL",
    "United Arab Emirates":   "AE",
}

TENOR_MAP: dict[str, float] = {
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0,
}

MIN_OBS_PER_COUNTRY = 24  # months


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_intensities(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "country", "lambda_total", "lambda_global", "lambda_country"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"intensities.csv missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


def _load_cds_from_csv(path: str) -> pd.DataFrame:
    """Load multi-tenor sovereign CDS from the raw CSV."""
    print(f"  Loading CDS CSV: {path}")

    header_df = pd.read_csv(path, nrows=0)
    all_cols  = header_df.columns.tolist()
    lower_map = {c.lower(): c for c in all_cols}

    def find(*candidates):
        for c in candidates:
            if c in lower_map:
                return lower_map[c]
        return None

    date_col     = find("date", "trade_date", "asofdate")
    country_col  = find("country", "country_name")
    tenor_col    = find("tenor", "maturity", "term")
    spread_col   = find("parspread", "spread", "mid_spread", "mid", "cds_spread")
    currency_col = find("currency", "ccy")
    tier_col     = find("tier", "seniority")
    recovery_col = find("cdsassumedrecovery", "recovery", "recovery_rate")

    missing_cols = [name for name, col in [
        ("date", date_col), ("country", country_col),
        ("tenor", tenor_col), ("spread", spread_col),
    ] if col is None]
    if missing_cols:
        raise ValueError(f"CDS CSV missing required columns: {missing_cols}. "
                         f"Available: {all_cols}")

    usecols = list({date_col, country_col, tenor_col, spread_col}
                   | ({currency_col} if currency_col else set())
                   | ({tier_col} if tier_col else set())
                   | ({recovery_col} if recovery_col else set()))

    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=200_000):
        # Filter to USD + SNRFOR early to save memory
        if currency_col and currency_col in chunk.columns:
            chunk = chunk[chunk[currency_col].astype(str).str.upper() == "USD"]
        if tier_col and tier_col in chunk.columns:
            chunk = chunk[chunk[tier_col].astype(str).str.upper() == "SNRFOR"]
        # Only keep tenors we care about
        if tenor_col in chunk.columns:
            chunk = chunk[chunk[tenor_col].isin(TENOR_MAP)]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    rename_map = {
        date_col:    "date",
        country_col: "country_raw",
        tenor_col:   "tenor",
        spread_col:  "spread",
    }
    if recovery_col and recovery_col in df.columns:
        rename_map[recovery_col] = "recovery"
    df = df.rename(columns=rename_map)
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    if "recovery" in df.columns:
        df["recovery"] = pd.to_numeric(df["recovery"], errors="coerce")
    df = df.dropna(subset=["date", "country_raw", "tenor", "spread"])
    return df


def load_cds_wide(csv_path:     str = CDS_CSV_PATH) -> pd.DataFrame:
    """Load CDS data, pivot to wide format by tenor, map country names."""
    if os.path.exists(csv_path):
        raw = _load_cds_from_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No CDS data found at {csv_path}."
        )

    # Map country names → ISO-2 codes; drop unmapped
    raw["country"] = raw["country_raw"].map(COUNTRY_NAME_MAP)
    n_before = len(raw)
    raw = raw[raw["country"].notna()].copy()
    n_dropped = n_before - len(raw)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with unmapped country names.")

    # Per-country median recovery (computed before pivoting, on all tenors combined)
    recovery_map: dict[str, float] = {}
    if "recovery" in raw.columns:
        recovery_map = (
            raw.dropna(subset=["recovery"])
            .groupby("country")["recovery"]
            .median()
            .to_dict()
        )
        if recovery_map:
            print(f"  Recovery rates (median): "
                  f"{ {k: f'{v:.2f}' for k, v in sorted(recovery_map.items())} }")

    # Month-end date index
    raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp("M")

    # Aggregate: mean spread per (country, date, tenor)
    agg = (
        raw.groupby(["country", "date", "tenor"], as_index=False)["spread"]
        .mean()
    )

    # Pivot to wide
    agg["tenor_col"] = "cds_spread_" + agg["tenor"].str.lower()
    wide = agg.pivot_table(
        index=["country", "date"],
        columns="tenor_col",
        values="spread",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    if recovery_map:
        wide["recovery_rate"] = wide["country"].map(recovery_map)

    print(f"  CDS wide panel: {len(wide):,} rows, "
          f"columns: {[c for c in wide.columns if c.startswith('cds')]}")
    return wide


# ---------------------------------------------------------------------------
# Merge & validate
# ---------------------------------------------------------------------------

def map_country_names(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """Map full country names to ISO-2 codes (in-place copy)."""
    out = df.copy()
    out[country_col] = out[country_col].map(COUNTRY_NAME_MAP).fillna(out[country_col])
    return out


def align_to_monthly_end(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Coerce date column to month-end timestamps."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.to_timestamp("M")
    return out


def load_cape_panel(path: str | None = None) -> pd.DataFrame | None:
    """Load CAPE panel for merging into the final panel.  Returns None if unavailable."""
    candidates = [p for p in [path, CAPE_PANEL_PATH] if p]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            df = df[["date", "country", "cape"]].copy()
            df["cape"] = pd.to_numeric(df["cape"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
            df = df.dropna(subset=["cape"])
            print(f"  Loaded CAPE from {p}: {len(df):,} rows")
            return df
    warnings.warn("CAPE panel not found; 'cape' column will be absent from merged panel.")
    return None


def merge_panel(
    intensities_df: pd.DataFrame,
    cds_wide_df:    pd.DataFrame,
    cape_df:        pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Inner join intensities + CDS on (country, date); left-join CAPE."""
    left   = align_to_monthly_end(intensities_df)
    right  = align_to_monthly_end(cds_wide_df)
    merged = left.merge(right, on=["country", "date"], how="inner")

    if cape_df is not None:
        cape_aligned = align_to_monthly_end(cape_df)
        merged = merged.merge(cape_aligned, on=["country", "date"], how="left")

    merged = merged.sort_values(["country", "date"]).reset_index(drop=True)
    return merged


def validate_panel(panel: pd.DataFrame) -> None:
    """Raise ValueError on structural problems; warn on soft issues."""
    cds_cols = [c for c in panel.columns if c.startswith("cds_spread_")]
    if not cds_cols:
        raise ValueError("Panel has no CDS spread columns.")

    for country, grp in panel.groupby("country"):
        n_obs = len(grp)
        if n_obs < MIN_OBS_PER_COUNTRY:
            raise ValueError(
                f"{country}: only {n_obs} observations (minimum {MIN_OBS_PER_COUNTRY})."
            )
        n_cds = grp[cds_cols].notna().any(axis=1).sum()
        if n_cds == 0:
            raise ValueError(f"{country}: all CDS spread values are NaN.")

    neg_g = (panel["lambda_global"] < 0).sum()
    neg_i = (panel["lambda_country"] < 0).sum()
    if neg_g:
        raise ValueError(f"{neg_g} rows have lambda_global < 0.")
    if neg_i:
        raise ValueError(f"{neg_i} rows have lambda_country < 0.")

    print(f"Validation passed: {panel['country'].nunique()} countries, "
          f"{len(panel):,} rows, tenors: {cds_cols}")


def save_panel(panel: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    panel.to_csv(path, index=False)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(INTENSITIES_PATH):
        raise FileNotFoundError(
            f"Intensities file not found: {INTENSITIES_PATH}\n"
            "Run  python -m calibration.extract_intensities  first."
        )

    print("Loading intensities …")
    intensities = load_intensities(INTENSITIES_PATH)

    print("Loading CAPE panel …")
    cape_df = load_cape_panel()

    print("Loading CDS spreads …")
    cds_wide = load_cds_wide()

    print("Merging …")
    panel = merge_panel(intensities, cds_wide, cape_df)

    print("Validating …")
    validate_panel(panel)

    save_panel(panel, OUT_PATH)

    print("\nPanel summary:")
    cds_cols = [c for c in panel.columns if c.startswith("cds_spread_")]
    for country, grp in panel.groupby("country"):
        n_total   = len(grp)
        n_cds     = grp[cds_cols].notna().any(axis=1).sum()
        n_cape    = int(grp["cape"].notna().sum()) if "cape" in grp.columns else 0
        date_range = f"{grp['date'].min().date()} → {grp['date'].max().date()}"
        miss_frac = {c: f"{grp[c].isna().mean()*100:.0f}%" for c in cds_cols}
        print(f"  {country:4s}: {n_total:4d} months ({date_range}), "
              f"{n_cds} with CDS, {n_cape} with CAPE")
        print(f"         missing CDS by tenor: { {k.replace('cds_spread_',''):v for k,v in miss_frac.items()} }")


if __name__ == "__main__":
    main()
