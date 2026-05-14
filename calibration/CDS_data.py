from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# =========================================================
# CONFIG — change only this
# =========================================================

CSV_PATH = "/Users/aryaman/honors/calibration/data/CDS spreads.csv"

# Column name hints — the profiler will try to auto-detect these.
# Override here if your CSV uses different names.
HINTS = {
    "date":      None,   # auto-detect
    "spread":    None,   # auto-detect (parspread, spread, mid, etc.)
    "country":   None,   # auto-detect
    "tenor":     None,   # auto-detect
    "ticker":    None,   # auto-detect
    "currency":  None,   # auto-detect
    "tier":      None,   # auto-detect
    "rating":    None,   # auto-detect
}

OUT_DIR = os.path.dirname(CSV_PATH)
CHUNKSIZE = 200_000

# =========================================================
# 1) AUTO-DETECT COLUMNS
# =========================================================

def _detect_columns(cols: list[str]) -> dict[str, str | None]:
    lower = {c.lower(): c for c in cols}
    def find(*candidates):
        for c in candidates:
            if c in lower:
                return lower[c]
        return None

    return {
        "date":     find("date", "date_", "trade_date", "asofdate"),
        "spread":   find("parspread", "spread", "mid_spread", "mid", "cds_spread", "value"),
        "country":  find("country", "country_name", "nation"),
        "tenor":    find("tenor", "maturity", "term"),
        "ticker":   find("ticker", "ric", "id", "name", "shortname", "entity"),
        "currency": find("currency", "ccy", "curr"),
        "tier":     find("tier", "seniority", "rank"),
        "rating":   find("avrating", "rating", "avg_rating", "credit_rating"),
        "docclause":find("docclause", "doc_clause", "restructuring"),
        "recovery": find("cdsassumedrecovery", "recovery", "recovery_rate"),
        "dp":       find("dp", "default_prob", "default_probability"),
    }


# =========================================================
# 2) LOAD (chunked, memory-safe)
# =========================================================

def load_csv(path: str) -> pd.DataFrame:
    print(f"Reading header...")
    header_df = pd.read_csv(path, nrows=0)
    all_cols = header_df.columns.tolist()
    print(f"  {len(all_cols)} columns found: {all_cols[:10]}{'...' if len(all_cols)>10 else ''}")

    detected = _detect_columns(all_cols)
    present  = {k: v for k, v in detected.items() if v is not None}
    print(f"\nAuto-detected columns:")
    for k, v in present.items():
        print(f"  {k:12s} → '{v}'")
    missing = [k for k, v in detected.items() if v is None]
    if missing:
        print(f"  Not found: {missing}")

    usecols = list(set(present.values()))

    print(f"\nLoading CSV in chunks ({CHUNKSIZE:,} rows each)...")
    chunks = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(
        path, usecols=usecols, low_memory=False, chunksize=CHUNKSIZE
    )):
        total += len(chunk)
        if i % 5 == 0:
            print(f"  chunk {i+1}: {total:,} rows so far")
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows loaded: {len(df):,}")

    # Coerce types
    date_col = present.get("date")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for col in ["spread", "recovery", "dp"]:
        raw = present.get(col)
        if raw:
            df[raw] = pd.to_numeric(df[raw], errors="coerce")

    df.attrs["cols"] = present
    return df


# =========================================================
# 3) PROFILE
# =========================================================

def profile(df: pd.DataFrame) -> None:
    c = df.attrs["cols"]
    date_col    = c.get("date")
    spread_col  = c.get("spread")
    country_col = c.get("country")
    tenor_col   = c.get("tenor")
    currency_col= c.get("currency")
    tier_col    = c.get("tier")
    rating_col  = c.get("rating")
    ticker_col  = c.get("ticker")

    sep = "=" * 70

    # ── 1. SHAPE ─────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SECTION 1 — DATASET SHAPE")
    print(sep)
    print(f"  Rows:    {len(df):>10,}")
    print(f"  Columns: {df.shape[1]:>10,}")
    print(f"\n  Null counts per column:")
    for col in df.columns:
        n_null = df[col].isna().sum()
        pct    = 100 * n_null / len(df)
        print(f"    {col:30s}  {n_null:>8,}  ({pct:5.1f}% null)")

    # ── 2. DATE COVERAGE ──────────────────────────────────────────────────────
    if date_col:
        print(f"\n{sep}")
        print("SECTION 2 — DATE COVERAGE")
        print(sep)
        dates = df[date_col].dropna()
        print(f"  Earliest:  {dates.min().date()}")
        print(f"  Latest:    {dates.max().date()}")
        print(f"  Span:      {(dates.max() - dates.min()).days / 365.25:.1f} years")

        # Infer frequency
        unique_dates = sorted(dates.unique())
        if len(unique_dates) > 1:
            diffs = pd.Series(unique_dates).diff().dropna().dt.days
            med_diff = diffs.median()
            if med_diff <= 1:    freq_guess = "Daily"
            elif med_diff <= 8:  freq_guess = "Weekly"
            elif med_diff <= 35: freq_guess = "Monthly"
            elif med_diff <= 95: freq_guess = "Quarterly"
            else:                freq_guess = "Annual or irregular"
            print(f"  Unique dates: {len(unique_dates):,}")
            print(f"  Median gap:   {med_diff:.0f} days → inferred frequency: {freq_guess}")

        # Obs per year
        df["_year"] = df[date_col].dt.year
        obs_by_year = df.groupby("_year").size()
        print(f"\n  Observations per year (sample):")
        for yr, n in obs_by_year.items():
            bar = "█" * min(int(n / obs_by_year.max() * 40), 40)
            print(f"    {yr}  {bar:<40s}  {n:>8,}")
        df.drop(columns=["_year"], inplace=True)

    # ── 3. COUNTRIES ──────────────────────────────────────────────────────────
    if country_col:
        print(f"\n{sep}")
        print("SECTION 3 — COUNTRIES")
        print(sep)
        country_counts = df[country_col].value_counts()
        print(f"  Unique countries: {country_counts.shape[0]:,}")
        print(f"\n  Top 30 by observation count:")
        for ctry, n in country_counts.head(30).items():
            bar = "█" * min(int(n / country_counts.iloc[0] * 40), 40)
            print(f"    {str(ctry):30s}  {bar:<40s}  {n:>8,}")

        if date_col:
            coverage = (
                df.dropna(subset=[date_col, country_col])
                  .groupby(country_col)[date_col]
                  .agg(["min","max","count"])
                  .rename(columns={"min":"first","max":"last","count":"n_obs"})
            )
            coverage["years"] = (coverage["last"] - coverage["first"]).dt.days / 365.25
            coverage = coverage.sort_values("n_obs", ascending=False)
            print(f"\n  Date coverage per country (top 20):")
            print(f"    {'Country':30s}  {'First':12s}  {'Last':12s}  {'Years':>6s}  {'N obs':>8s}")
            print(f"    {'-'*30}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*8}")
            for ctry, row in coverage.head(20).iterrows():
                print(f"    {str(ctry):30s}  {str(row['first'].date()):12s}  "
                      f"{str(row['last'].date()):12s}  {row['years']:6.1f}  {int(row['n_obs']):>8,}")

    # ── 4. TENORS ─────────────────────────────────────────────────────────────
    if tenor_col:
        print(f"\n{sep}")
        print("SECTION 4 — TENORS")
        print(sep)
        tenor_counts = df[tenor_col].value_counts()
        print(f"  Unique tenors: {tenor_counts.shape[0]}")
        for t, n in tenor_counts.items():
            bar = "█" * min(int(n / tenor_counts.iloc[0] * 40), 40)
            print(f"    {str(t):10s}  {bar:<40s}  {n:>8,}")

    # ── 5. CURRENCIES ─────────────────────────────────────────────────────────
    if currency_col:
        print(f"\n{sep}")
        print("SECTION 5 — CURRENCIES")
        print(sep)
        ccy_counts = df[currency_col].value_counts()
        print(f"  Unique currencies: {ccy_counts.shape[0]}")
        for ccy, n in ccy_counts.items():
            bar = "█" * min(int(n / ccy_counts.iloc[0] * 40), 40)
            print(f"    {str(ccy):10s}  {bar:<40s}  {n:>8,}")

    # ── 6. TIER / SENIORITY ───────────────────────────────────────────────────
    if tier_col:
        print(f"\n{sep}")
        print("SECTION 6 — TIER / SENIORITY")
        print(sep)
        for t, n in df[tier_col].value_counts().items():
            print(f"    {str(t):30s}  {n:>8,}")

    # ── 7. RATINGS ────────────────────────────────────────────────────────────
    if rating_col:
        print(f"\n{sep}")
        print("SECTION 7 — RATINGS")
        print(sep)
        for r, n in df[rating_col].value_counts().head(20).items():
            print(f"    {str(r):10s}  {n:>8,}")

    # ── 8. SPREAD STATISTICS ──────────────────────────────────────────────────
    if spread_col:
        print(f"\n{sep}")
        print("SECTION 8 — SPREAD STATISTICS")
        print(sep)
        s = df[spread_col].dropna()
        print(f"  Non-null observations: {len(s):,}")
        print(f"  Min:    {s.min():>10.1f} bps")
        print(f"  p1:     {s.quantile(0.01):>10.1f} bps")
        print(f"  p5:     {s.quantile(0.05):>10.1f} bps")
        print(f"  p25:    {s.quantile(0.25):>10.1f} bps")
        print(f"  Median: {s.median():>10.1f} bps")
        print(f"  Mean:   {s.mean():>10.1f} bps")
        print(f"  p75:    {s.quantile(0.75):>10.1f} bps")
        print(f"  p95:    {s.quantile(0.95):>10.1f} bps")
        print(f"  p99:    {s.quantile(0.99):>10.1f} bps")
        print(f"  Max:    {s.max():>10.1f} bps")
        print(f"  Std:    {s.std():>10.1f} bps")
        print(f"\n  Extreme values (>2000 bps): {(s > 2000).sum():,} rows")
        print(f"  Negative values:             {(s < 0).sum():,} rows")
        print(f"  Zero values:                 {(s == 0).sum():,} rows")

    # ── 9. CROSS-TABS ─────────────────────────────────────────────────────────
    if tenor_col and currency_col:
        print(f"\n{sep}")
        print("SECTION 9 — TENOR × CURRENCY CROSS-TAB (row counts)")
        print(sep)
        xt = pd.crosstab(df[tenor_col], df[currency_col])
        print(xt.to_string())

    if tenor_col and country_col and spread_col:
        print(f"\n{sep}")
        print("SECTION 10 — MEAN SPREAD BY TENOR (across all countries)")
        print(sep)
        tenor_spread = (
            df.dropna(subset=[tenor_col, spread_col])
              .groupby(tenor_col)[spread_col]
              .agg(["mean","median","count"])
              .sort_values("mean")
        )
        print(tenor_spread.to_string())

    print(f"\n{sep}")
    print("PROFILING COMPLETE")
    print(sep)

    # ── 11. RECOVERY STATISTICS ───────────────────────────────────────────────
    recovery_col = c.get("recovery")
    if recovery_col:
        print(f"\n{sep}")
        print("SECTION 11 — ASSUMED RECOVERY (cdsassumedrecovery)")
        print(sep)

        rec_df = df[[country_col, recovery_col]].dropna()

        if rec_df.empty:
            print("  No non-null recovery values found.")
        else:
            grouped = rec_df.groupby(country_col)[recovery_col]

            summary = grouped.agg(
                count="count",
                mean="mean",
                median="median",
                min="min",
                max="max"
            ).sort_values("count", ascending=False)

            print(f"  {'Country':30s}  {'N':>8s}  {'Mean':>10s}  {'Median':>10s}  {'Min':>10s}  {'Max':>10s}")
            print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

            for ctry, row in summary.iterrows():
                print(
                    f"  {str(ctry):30s}  "
                    f"{int(row['count']):>8,}  "
                    f"{row['mean']:>10.4f}  "
                    f"{row['median']:>10.4f}  "
                    f"{row['min']:>10.4f}  "
                    f"{row['max']:>10.4f}"
                )


# =========================================================
# 4) QUICK PLOTS
# =========================================================

def make_plots(df: pd.DataFrame) -> None:
    c    = df.attrs["cols"]
    date_col    = c.get("date")
    spread_col  = c.get("spread")
    country_col = c.get("country")
    tenor_col   = c.get("tenor")
    currency_col= c.get("currency")

    if not (date_col and spread_col and country_col):
        print("Skipping plots — need date, spread, and country columns.")
        return

    print("\nGenerating plots...")

    # Filter to most common tenor + currency for cleaner plots
    filt = df.dropna(subset=[date_col, spread_col, country_col]).copy()
    if tenor_col and filt[tenor_col].nunique() > 1:
        top_tenor = filt[tenor_col].value_counts().index[0]
        filt = filt[filt[tenor_col] == top_tenor]
        print(f"  Plots filtered to tenor='{top_tenor}' (most common)")
    if currency_col and filt[currency_col].nunique() > 1:
        top_ccy = filt[currency_col].value_counts().index[0]
        filt = filt[filt[currency_col] == top_ccy]
        print(f"  Plots filtered to currency='{top_ccy}' (most common)")

    top_countries = (
        filt.groupby(country_col)[spread_col].count()
            .sort_values(ascending=False)
            .head(12).index.tolist()
    )

    pivot = (
        filt[filt[country_col].isin(top_countries)]
        .pivot_table(index=date_col, columns=country_col, values=spread_col, aggfunc="mean")
        .resample("ME").mean()
        .sort_index()
    )

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1 — time series
    ax1 = fig.add_subplot(gs[0, :])
    for col in pivot.columns:
        ax1.plot(pivot.index, pivot[col], lw=1.2, alpha=0.85, label=col)
    ax1.set_title("CDS Par Spread — Top Countries (monthly avg)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Spread (bps)")
    ax1.legend(ncol=4, fontsize=7, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot 2 — latest bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    latest = pivot.ffill().iloc[-1].dropna().sort_values(ascending=False)
    ax2.barh(latest.index[::-1], latest.values[::-1], color="steelblue")
    ax2.set_title(f"Latest Spreads ({pivot.index[-1].date()})", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Spread (bps)")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Plot 3 — distribution
    ax3 = fig.add_subplot(gs[1, 1])
    clipped = filt[spread_col].clip(upper=filt[spread_col].quantile(0.99))
    ax3.hist(clipped.dropna(), bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax3.set_title("Distribution of Spreads (clipped at p99)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Spread (bps)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Plot 4 — obs over time
    ax4 = fig.add_subplot(gs[2, 0])
    obs_ts = filt.set_index(date_col).resample("ME")[spread_col].count()
    ax4.bar(obs_ts.index, obs_ts.values, width=20, color="steelblue", alpha=0.8)
    ax4.set_title("Observations per Month", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Count")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Plot 5 — avg vs median spread over time
    ax5 = fig.add_subplot(gs[2, 1])
    ts_mean   = filt.groupby(date_col)[spread_col].mean().resample("ME").mean()
    ts_median = filt.groupby(date_col)[spread_col].median().resample("ME").median()
    ax5.plot(ts_mean.index,   ts_mean.values,   label="Mean",   lw=1.5)
    ax5.plot(ts_median.index, ts_median.values, label="Median", lw=1.5, linestyle="--")
    ax5.set_title("Cross-Country Average Spread Over Time", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Spread (bps)")
    ax5.legend()
    ax5.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(OUT_DIR, "cds_profile_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plots → {out_path}")
    plt.close()


# =========================================================
# 5) MAIN
# =========================================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: File not found: {CSV_PATH}")
        print("Edit CSV_PATH at the top of this script.")
        sys.exit(1)

    df = load_csv(CSV_PATH)
    profile(df)
    make_plots(df)


if __name__ == "__main__":
    main()