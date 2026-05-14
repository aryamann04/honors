from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, List, Sequence, Tuple

import sys
import os
import numpy as np
import pandas as pd


# =========================================================
# 1) CONFIG
# =========================================================

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

START_DATE = "1980-01-01"
END_DATE = None
ROLLING_MONTHS = 120
MIN_OBS_FOR_CAPE = 120

OUT_RAW = os.path.join(BASE_DIR, "raw_data", "country_cape_raw.csv")
OUT_PANEL = os.path.join(BASE_DIR, "raw_data", "country_cape_panel.csv")
OUT_SUMMARY = os.path.join(BASE_DIR, "raw_data", "country_cape_summary.csv")

OUT_US_SHILLER_COMPARISON = os.path.join(
    BASE_DIR, "raw_data", "us_lseg_vs_shiller_cape.png"
)
SHILLER_CAPE_CSV = os.path.join(BASE_DIR, "raw_data", "shiller CAPE.csv")

COUNTRY_SERIES: Dict[str, Dict] = {
    "US": {
        "price_universes_fields": [(".SPX","TRDPRC_1"),(".SPX","CLOSE"),(".SPX","TR.PriceClose")],
        "pe_universes": [".SPX"],
        "cpi_universes": ["aUSCPI"],
    },
    "UK": {
        "price_universes_fields": [(".FTSE","TRDPRC_1"),(".FTSE","CLOSE"),(".FTSE","TR.PriceClose")],
        "pe_universes": [".FTSE"],
        "cpi_universes": ["aGBCPI"],
    },
    "JP": {
        "price_universes_fields": [(".N225","TRDPRC_1"),(".N225","CLOSE"),(".N225","TR.PriceClose")],
        "pe_universes": [".N225"],
        "cpi_universes": ["aJPCPI"],
    },
    "DE": {
        "price_universes_fields": [(".GDAXI","TRDPRC_1"),(".GDAXI","CLOSE"),(".GDAXI","TR.PriceClose")],
        "pe_universes": [".GDAXI"],
        "cpi_universes": ["aDECPI"],
    },
    "MX": {
        "price_universes_fields": [(".MXX","TRDPRC_1"),(".MXX","CLOSE"),(".MXX","TR.PriceClose")],
        "pe_universes": [".MXX"],
        "cpi_universes": ["aMXCPI"],
    },

    "ES": {
        "price_universes_fields": [(".IBEX","TRDPRC_1"),(".IBEX","CLOSE"),(".IBEX","TR.PriceClose")],
        "pe_universes": [".IBEX"],
        "cpi_universes": ["aESCPI"],
    },
    "FR": {
        "price_universes_fields": [(".FCHI","TRDPRC_1"),(".FCHI","CLOSE"),(".FCHI","TR.PriceClose")],
        "pe_universes": [".FCHI"],
        "cpi_universes": ["aFRCPI"],
    },
    "BR": {
        "price_universes_fields": [(".BVSP","TRDPRC_1"),(".BVSP","CLOSE"),(".BVSP","TR.PriceClose")],
        "pe_universes": [".BVSP"],
        "cpi_universes": ["aBRCPI"],
    },
    "ID": {
        "price_universes_fields": [(".JKSE","TRDPRC_1"),(".JKSE","CLOSE"),(".JKSE","TR.PriceClose")],
        "pe_universes": [".JKSE"],
        "cpi_universes": ["aIDCPI"],
    },
    "MY": {
        "price_universes_fields": [(".KLSE","TRDPRC_1"),(".KLSE","CLOSE"),(".KLSE","TR.PriceClose")],
        "pe_universes": [".KLSE"],
        "cpi_universes": ["aMYCPI"],
    },
    "SE": {
        "price_universes_fields": [(".OMXS30","TRDPRC_1"),(".OMXS30","CLOSE"),(".OMXS30","TR.PriceClose")],
        "pe_universes": [".OMXS30"],
        "cpi_universes": ["aSECPI"],
    },
    "NL": {
        "price_universes_fields": [(".AEX","TRDPRC_1"),(".AEX","CLOSE"),(".AEX","TR.PriceClose")],
        "pe_universes": [".AEX"],
        "cpi_universes": ["aNLCPI"],
    },
    "TH": {
        "price_universes_fields": [(".SETI","TRDPRC_1"),(".SETI","CLOSE"),(".SETI","TR.PriceClose")],
        "pe_universes": [".SETI"],
        "cpi_universes": ["aTHCPI"],
    },
    "TR": {
        "price_universes_fields": [(".XU100","TRDPRC_1"),(".XU100","CLOSE"),(".XU100","TR.PriceClose")],
        "pe_universes": [".XU100"],
        "cpi_universes": ["aTRCPI"],
    },
    "CL": {
        "price_universes_fields": [(".SPIPSA","TRDPRC_1"),(".SPIPSA","CLOSE"),(".SPIPSA","TR.PriceClose")],
        "pe_universes": [".SPIPSA"],
        "cpi_universes": ["aCLCPI"],
    },
}

CPI_HISTORY_FIELDS: List[Optional[str]] = [None, "TR.Value"]
CHUNK_YEARS = 5


# =========================================================
# 2) DATA MODELS
# =========================================================

@dataclass
class SeriesResult:
    data: pd.DataFrame
    universe: str
    field: Optional[str]


@dataclass
class CountryConfig:
    country: str
    price_universes_fields: Sequence[Tuple[str, Optional[str]]]
    pe_universes: Sequence[str]
    cpi_universes: Sequence[str]


# =========================================================
# 3) SESSION
# =========================================================

def lseg_open_session():
    import lseg.data as ld
    ld.open_session()
    return ld


def lseg_close_session(ld) -> None:
    try:
        ld.close_session()
    except Exception:
        pass


# =========================================================
# 4) HELPERS
# =========================================================

def _date_chunks(
    start: str,
    end: Optional[str],
    chunk_years: int = CHUNK_YEARS,
) -> List[Tuple[str, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end else pd.Timestamp.now()
    chunks = []
    cur = s
    while cur < e:
        nxt = min(cur + pd.DateOffset(years=chunk_years), e)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.DateOffset(days=1)
    return chunks


def _coerce_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index = out.index.to_period("M").to_timestamp("M")
    out = out[~out.index.duplicated(keep="last")]
    return out


def _standardize_single_series(
    obj: pd.DataFrame | pd.Series,
    value_name: str,
) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return _coerce_month_end_index(obj.to_frame(name=value_name))

    out = obj.copy()

    if "Date" in out.columns:
        out = out.set_index("Date")
    elif "date" in out.columns:
        out = out.set_index("date")
    else:
        date_cols = [c for c in out.columns if "date" in str(c).lower()]
        if date_cols:
            out = out.set_index(date_cols[0])

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(
            f"No numeric columns for '{value_name}'. Got: {out.columns.tolist()}"
        )

    preferred = [
        c for c in numeric_cols
        if str(c).lower() in {"value", "close", "trdprc_1", "price close", "pe"}
        or value_name.lower() in str(c).lower()
    ]
    chosen = preferred[0] if preferred else numeric_cols[0]
    out = out[[chosen]].rename(columns={chosen: value_name})
    return _coerce_month_end_index(out)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _validate_country_panel(df: pd.DataFrame, country: str) -> None:
    needed = ["price", "earnings", "cpi"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{country}: missing columns {missing}")
    for col in needed:
        if df[col].dropna().empty:
            raise ValueError(f"{country}: '{col}' is empty after cleaning.")
    if (df["cpi"].dropna() <= 0).any():
        raise ValueError(f"{country}: cpi contains non-positive values.")


# =========================================================
# 5) CHUNKED FETCH
# =========================================================

def fetch_chunked(
    ld,
    universe: str,
    field: Optional[str],
    start: str,
    end: Optional[str],
    interval: str,
    value_name: str,
) -> pd.DataFrame:
    chunks = _date_chunks(start, end)
    pieces: List[pd.DataFrame] = []

    for chunk_start, chunk_end in chunks:
        try:
            kwargs: dict = dict(
                universe=universe,
                start=chunk_start,
                end=chunk_end,
                interval=interval,
            )
            if field is not None:
                kwargs["fields"] = [field]

            raw = ld.get_history(**kwargs)
            if raw is None or raw.empty:
                continue

            piece = _standardize_single_series(raw, value_name)
            if not piece[value_name].dropna().empty:
                pieces.append(piece)
        except Exception:
            continue

    if not pieces:
        raise ValueError(
            f"fetch_chunked returned no data for {universe}/{field}"
        )

    combined = pd.concat(pieces)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def try_price_candidates(
    ld,
    universes_fields: Sequence[Tuple[str, Optional[str]]],
    start: str,
    end: Optional[str],
) -> SeriesResult:
    errors: List[str] = []
    for universe, field in universes_fields:
        try:
            df = fetch_chunked(
                ld=ld, universe=universe, field=field,
                start=start, end=end,
                interval="monthly", value_name="price",
            )
            n = df["price"].dropna().shape[0]
            if n > 0:
                print(f"  Price: {universe}/{field} → {n} obs "
                      f"({df['price'].dropna().index.min().date()} → "
                      f"{df['price'].dropna().index.max().date()})")
                return SeriesResult(df, universe, field)
            errors.append(f"  {universe}/{field}: 0 obs")
        except Exception as e:
            errors.append(f"  {universe}/{field}: {e}")
    raise RuntimeError("All price candidates failed:\n" + "\n".join(errors))


def try_cpi_candidates(
    ld,
    universes: Sequence[str],
    fields: Sequence[Optional[str]],
    start: str,
    end: Optional[str],
) -> SeriesResult:
    errors: List[str] = []
    for universe in universes:
        for field in fields:
            try:
                df = fetch_chunked(
                    ld=ld, universe=universe, field=field,
                    start=start, end=end,
                    interval="monthly", value_name="cpi",
                )
                n = df["cpi"].dropna().shape[0]
                if n > 0:
                    print(f"  CPI:   {universe}/{field} → {n} obs "
                          f"({df['cpi'].dropna().index.min().date()} → "
                          f"{df['cpi'].dropna().index.max().date()})")
                    return SeriesResult(df, universe, field)
                errors.append(f"  {universe}/{field}: 0 obs")
            except Exception as e:
                errors.append(f"  {universe}/{field}: {e}")
    raise RuntimeError("All CPI candidates failed:\n" + "\n".join(errors))


def fetch_pe_series(
    ld,
    pe_universes: Sequence[str],
    start: str,
    end: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    errors: List[str] = []
    for universe in pe_universes:
        try:
            df = fetch_chunked(
                ld=ld, universe=universe,
                field="TR.Index_PE_RTRS",
                start=start, end=end,
                interval="monthly", value_name="pe",
            )
            n = df["pe"].dropna().shape[0]
            if n > 0:
                print(f"  PE:    {universe}/TR.Index_PE_RTRS → {n} obs "
                      f"({df['pe'].dropna().index.min().date()} → "
                      f"{df['pe'].dropna().index.max().date()})")
                return df, universe
            errors.append(f"  {universe}: 0 obs")
        except Exception as e:
            errors.append(f"  {universe}: {e}")
    raise RuntimeError("All PE candidates failed:\n" + "\n".join(errors))


# =========================================================
# 6) COUNTRY COMPONENT FETCH
# =========================================================

def fetch_country_components(
    ld,
    cfg: CountryConfig,
    start: str,
    end: Optional[str],
) -> pd.DataFrame:
    price_res = try_price_candidates(
        ld=ld,
        universes_fields=cfg.price_universes_fields,
        start=start, end=end,
    )
    cpi_res = try_cpi_candidates(
        ld=ld,
        universes=cfg.cpi_universes,
        fields=CPI_HISTORY_FIELDS,
        start=start, end=end,
    )
    pe_df, pe_universe = fetch_pe_series(
        ld=ld,
        pe_universes=cfg.pe_universes,
        start=start, end=end,
    )

    # Derive nominal EPS = price / PE
    price_pe = price_res.data.join(pe_df, how="outer")
    price_pe["earnings"] = _safe_divide(price_pe["price"], price_pe["pe"])
    earnings_df = price_pe[["earnings"]]

    df = (
        price_res.data
        .join(earnings_df, how="outer")
        .join(cpi_res.data, how="outer")
    )
    df = df.sort_index()

    if df.empty:
        raise ValueError(f"{cfg.country}: no data after joins.")

    monthly_index = pd.date_range(df.index.min(), df.index.max(), freq="ME")
    df = df.reindex(monthly_index)

    df["price"]    = pd.to_numeric(df["price"],    errors="coerce").ffill(limit=2)
    df["earnings"] = pd.to_numeric(df["earnings"], errors="coerce").ffill(limit=12)
    df["cpi"]      = pd.to_numeric(df["cpi"],      errors="coerce").ffill(limit=3)
    df = df.infer_objects(copy=False)

    df.attrs["price_universe"]    = price_res.universe
    df.attrs["price_field"]       = price_res.field
    df.attrs["earnings_universe"] = pe_universe
    df.attrs["earnings_field"]    = "TR.Index_PE_RTRS → EPS=Price/PE"
    df.attrs["cpi_universe"]      = cpi_res.universe
    df.attrs["cpi_field"]         = cpi_res.field

    return df


# =========================================================
# 7) CAPE CALCULATION
# =========================================================

def compute_cape(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Correct Shiller CAPE using a time-varying CPI reference.

    At each date t, historical earnings are inflated to time-t dollars:
        real_earnings_{t,j} = earnings_{t-j} × (CPI_t / CPI_{t-j})

    Equivalently:
        avg_real_10y_t = CPI_t × rolling_mean_120( earnings_t / CPI_t )
        CAPE_t         = price_t / avg_real_10y_t

    This matches: CAPE_t = P_t / [1/120 × Σ_{j=0}^{119} (CPI_t/CPI_{t-j}) × E_{t-j}]
    """
    out = df.copy()
    _validate_country_panel(out, country=country)

    # Deflate earnings by the contemporaneous CPI — this puts earnings in
    # consistent real units regardless of the CPI index level.
    out["deflated_earnings"] = _safe_divide(out["earnings"], out["cpi"])

    # 10-year rolling mean of deflated earnings
    out["avg_deflated_earnings_10y"] = (
        out["deflated_earnings"]
        .rolling(window=ROLLING_MONTHS, min_periods=MIN_OBS_FOR_CAPE)
        .mean()
    )

    # Scale back to current-period dollars: multiply by CPI_t
    out["avg_real_earnings_10y"] = out["avg_deflated_earnings_10y"] * out["cpi"]

    # For reporting: real earnings in current-period dollars
    out["real_earnings"] = out["deflated_earnings"] * out["cpi"]   # = earnings_t (trivially)

    out["cape"] = _safe_divide(out["price"], out["avg_real_earnings_10y"])
    out["cyclically_adjusted_earnings_yield"] = _safe_divide(
        out["avg_real_earnings_10y"], out["price"]
    )
    return out


def normalize_country_cape(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cape = out["cape"]
    hist_mean   = cape.mean(skipna=True)
    hist_std    = cape.std(skipna=True)
    hist_median = cape.median(skipna=True)
    out["cape_zscore"] = (
        (cape - hist_mean) / hist_std
        if pd.notna(hist_std) and hist_std > 0 else np.nan
    )
    out["cape_to_hist_median"] = (
        cape / hist_median
        if pd.notna(hist_median) and hist_median != 0 else np.nan
    )
    out["cape_hist_percentile"] = cape.rank(method="average", pct=True)
    return out


# =========================================================
# 8) PANEL BUILD
# =========================================================

def build_country_cape_panel(
    start: str,
    end: Optional[str],
) -> pd.DataFrame:
    ld = lseg_open_session()
    all_frames: List[pd.DataFrame] = []
    failures:   List[str] = []

    try:
        for country, mapping in COUNTRY_SERIES.items():
            print(f"\n[{country}] fetching ({start} → {end or 'today'})...")
            cfg = CountryConfig(
                country=country,
                price_universes_fields=mapping["price_universes_fields"],
                pe_universes=mapping["pe_universes"],
                cpi_universes=mapping["cpi_universes"],
            )

            try:
                raw = fetch_country_components(
                    ld=ld, cfg=cfg, start=start, end=end
                )
                out = compute_cape(raw, country=country)
                n_cape = out["cape"].dropna().shape[0]
                latest_cape = out["cape"].dropna().iloc[-1]
                print(f"  CAPE: {n_cape} months | latest = {latest_cape:.2f}")

                out = normalize_country_cape(out)
                out["country"]         = country
                out["price_series"]    = raw.attrs.get("price_universe")
                out["price_field"]     = raw.attrs.get("price_field")
                out["earnings_series"] = raw.attrs.get("earnings_universe")
                out["earnings_field"]  = raw.attrs.get("earnings_field")
                out["cpi_series"]      = raw.attrs.get("cpi_universe")
                out["cpi_field"]       = raw.attrs.get("cpi_field")

                all_frames.append(
                    out.reset_index().rename(columns={"index": "date"})
                )

            except Exception as e:
                msg = f"{country}: {e}"
                failures.append(msg)
                warnings.warn(msg)

    finally:
        lseg_close_session(ld)

    if not all_frames:
        raise RuntimeError(
            "No country panels built.\nFailures:\n" +
            ("\n".join(failures) if failures else "none")
        )

    panel = pd.concat(all_frames, axis=0, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country", "date"]).reset_index(drop=True)
    return panel


# =========================================================
# 9) OUTPUT
# =========================================================

def make_summary(panel: pd.DataFrame) -> pd.DataFrame:
    latest = (
        panel.sort_values("date")
        .groupby("country", as_index=False)
        .tail(1)
        .copy()
    )
    cols = [
        "country", "date", "cape", "cape_zscore", "cape_to_hist_median",
        "cape_hist_percentile", "price", "earnings", "cpi",
        "price_series", "price_field",
        "earnings_series", "earnings_field",
        "cpi_series", "cpi_field",
    ]
    return (
        latest[[c for c in cols if c in latest.columns]]
        .sort_values("country")
        .reset_index(drop=True)
    )


def save_outputs(panel: pd.DataFrame) -> None:
    panel.to_csv(OUT_PANEL, index=False)
    raw_cols = [
        "country", "date", "price", "earnings", "cpi",
        "real_earnings", "avg_real_earnings_10y", "cape",
        "cyclically_adjusted_earnings_yield",
        "price_series", "price_field",
        "earnings_series", "earnings_field",
        "cpi_series", "cpi_field",
    ]
    panel[[c for c in raw_cols if c in panel.columns]].to_csv(OUT_RAW, index=False)
    make_summary(panel).to_csv(OUT_SUMMARY, index=False)

def plot_us_lseg_vs_shiller_cape(
    panel: pd.DataFrame,
    shiller_csv: str = SHILLER_CAPE_CSV,
    out_path: str = OUT_US_SHILLER_COMPARISON,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    us = panel.loc[panel["country"].eq("US"), ["date", "cape"]].copy()
    us["date"] = pd.to_datetime(us["date"])
    us = us.rename(columns={"cape": "lseg_cape"})

    shiller = pd.read_csv(shiller_csv)
    shiller.columns = [str(c).strip() for c in shiller.columns]

    date_col = next(
        (c for c in shiller.columns if c.lower() in {"date", "month"}),
        shiller.columns[0],
    )

    cape_col = next(
        (
            c for c in shiller.columns
            if "cape" in c.lower()
            or "cyclically adjusted" in c.lower()
            or "p/e10" in c.lower()
            or "pe10" in c.lower()
        ),
        None,
    )
    if cape_col is None:
        raise ValueError(
            f"Could not identify Shiller CAPE column. Columns: {shiller.columns.tolist()}"
        )

    shiller = shiller[[date_col, cape_col]].rename(
        columns={date_col: "date", cape_col: "shiller_cape"}
    )

    if pd.api.types.is_numeric_dtype(shiller["date"]):
        shiller["date"] = shiller["date"].astype(float)
        years = np.floor(shiller["date"]).astype(int)
        months = np.round((shiller["date"] - years) * 100).astype(int)
        months = months.where(months.between(1, 12), 1)
        shiller["date"] = pd.to_datetime(
            {"year": years, "month": months, "day": 1}
        )
    else:
        shiller["date"] = pd.to_datetime(shiller["date"], errors="coerce")

    shiller["date"] = shiller["date"].dt.to_period("M").dt.to_timestamp("M")
    shiller["shiller_cape"] = pd.to_numeric(shiller["shiller_cape"], errors="coerce")

    us["date"] = us["date"].dt.to_period("M").dt.to_timestamp("M")

    comparison = (
        us.merge(shiller, on="date", how="inner")
        .dropna(subset=["lseg_cape", "shiller_cape"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if comparison.empty:
        raise ValueError("No overlapping non-missing observations between LSEG and Shiller CAPE.")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(comparison["date"], comparison["lseg_cape"], label="LSEG-derived U.S. CAPE")
    ax.plot(comparison["date"], comparison["shiller_cape"], label="Shiller U.S. CAPE")

    ax.set_title("U.S. CAPE: LSEG-Derived vs. Shiller")
    ax.set_xlabel("Date")
    ax.set_ylabel("CAPE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return comparison

# =========================================================
# 10) MAIN
# =========================================================

def main() -> None:
    panel = build_country_cape_panel(start=START_DATE, end=END_DATE)
    save_outputs(panel)
    print(f"\n✓ {len(panel):,} rows | {panel['country'].nunique()} countries | "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")
    print("\nLatest summary:")
    print(make_summary(panel).to_string(index=False))


if __name__ == "__main__":
    main()
