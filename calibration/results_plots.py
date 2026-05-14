"""
results_plots.py
================
Thesis-quality calibration and empirical figures.

Figure inventory
----------------
 1. valuation_vs_intensity.png          — CAPE vs λ_total time series (dual axis)
 2. global_intensity_timeseries.png     — λ^g over time
 3. country_intensity_panels.png        — λ^i per selected country
 4. estimated_parameters.png            — h0, η_f, η_g, global share bar charts
 5. model_vs_observed_scatter.png       — scatter with 45° line, coloured by tenor
 6. model_vs_observed_5y_timeseries.png — obs vs model 5Y CDS per country
 7. residuals_by_tenor.png              — boxplot of residuals by tenor
 8. residuals_over_time.png             — rolling MAE over time
 9. cds_data_coverage_heatmap.png       — country × tenor observation counts
10. rmse_by_country_tenor.png           — RMSE heatmap
11. latest_cds_term_structure_fit.png   — obs vs model term structure at latest date
12. hazard_proxy_scatter.png            — model hazard vs CDS-implied proxy scatter
13. hazard_proxy_timeseries.png         — time series comparison per country
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

FIGS_DIR = os.path.join(BASE_DIR, "results", "figures")

# Preferred order of representative countries for crowded plots
REPR_ORDER = ["US", "JP", "DE", "UK", "MX", "TR", "ZA", "AR", "BR", "IT", "KR", "RU"]
TENOR_COLORS = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c",
                5.0: "#d62728", 7.0: "#9467bd", 10.0: "#8c564b"}
DPI = 150


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, name: str, out_dir: str = FIGS_DIR, dpi: int = DPI) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if not fig.get_constrained_layout():
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.relpath(path, PARENT_DIR)}")


def _repr_countries(panel: pd.DataFrame, n_max: int = 8) -> list[str]:
    available = set(panel["country"].unique())
    ordered   = [c for c in REPR_ORDER if c in available]
    rest      = sorted(available - set(ordered))
    combined  = (ordered + rest)[:n_max]
    return combined


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Valuation vs implied intensity (dual-axis time series)
# ---------------------------------------------------------------------------

def plot_valuation_vs_intensity(
    panel:    pd.DataFrame,
    out_dir:  str = FIGS_DIR,
    countries: Sequence[str] | None = None,
) -> None:
    if "cape" not in panel.columns and "lambda_total" not in panel.columns:
        print("  SKIP valuation_vs_intensity: missing cape or lambda_total")
        return
    ctries = list(countries) if countries else _repr_countries(panel, 8)
    ctries = [c for c in ctries if c in panel["country"].unique()]
    if not ctries:
        return

    n  = len(ctries)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 3.5 * nr), squeeze=False)

    for idx, country in enumerate(ctries):
        ax1 = axes[idx // nc][idx % nc]
        grp = panel[panel["country"] == country].sort_values("date")

        ax1.set_title(country, fontsize=11, fontweight="bold")
        _style_ax(ax1)

        if "cape" in grp.columns and grp["cape"].notna().any():
            ax1.plot(grp["date"], grp["cape"], color="#1a6faf", lw=1.4, label="CAPE")
            ax1.set_ylabel("CAPE ratio", color="#1a6faf", fontsize=9)
            ax1.tick_params(axis="y", labelcolor="#1a6faf", labelsize=8)

            if "lambda_total" in grp.columns:
                ax2 = ax1.twinx()
                ax2.plot(grp["date"], grp["lambda_total"] * 100, color="#c0392b",
                         lw=1.2, ls="--", label="λ_total (%)")
                ax2.set_ylabel("λ_total (% p.a.)", color="#c0392b", fontsize=9)
                ax2.tick_params(axis="y", labelcolor="#c0392b", labelsize=8)
                ax2.spines["top"].set_visible(False)
        elif "lambda_total" in grp.columns:
            ax1.plot(grp["date"], grp["lambda_total"] * 100, color="#c0392b", lw=1.2)
            ax1.set_ylabel("λ_total (% p.a.)", fontsize=9)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        ax1.tick_params(axis="x", labelsize=8, rotation=30)

    # Hide unused axes
    for k in range(len(ctries), nr * nc):
        axes[k // nc][k % nc].set_visible(False)

    fig.suptitle("Valuation Ratio vs Implied Disaster Intensity", y=1.01,
                 fontsize=13, fontweight="bold")
    _savefig(fig, "valuation_vs_intensity.png", out_dir)


# ---------------------------------------------------------------------------
# 2. Global intensity time series
# ---------------------------------------------------------------------------

def plot_global_intensity_timeseries(
    panel:   pd.DataFrame,
    out_dir: str = FIGS_DIR,
) -> None:
    if "lambda_global" not in panel.columns:
        return
    global_ts = (panel.groupby("date")["lambda_global"]
                 .first()
                 .reset_index()
                 .sort_values("date"))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(global_ts["date"], global_ts["lambda_global"] * 100, color="#2c3e50", lw=1.6)
    ax.fill_between(global_ts["date"], 0, global_ts["lambda_global"] * 100,
                    alpha=0.15, color="#2c3e50")
    ax.set_ylabel("λ_global (% p.a.)", fontsize=11)
    ax.set_title("Global Disaster Intensity λ^g over Time", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    _style_ax(ax)
    _savefig(fig, "global_intensity_timeseries.png", out_dir)


# ---------------------------------------------------------------------------
# 3. Country-specific intensity panels
# ---------------------------------------------------------------------------

def plot_country_intensity_panels(
    panel:     pd.DataFrame,
    out_dir:   str = FIGS_DIR,
    countries: Sequence[str] | None = None,
) -> None:
    if "lambda_country" not in panel.columns:
        return
    ctries = list(countries) if countries else _repr_countries(panel, 8)
    ctries = [c for c in ctries if c in panel["country"].unique()]
    if not ctries:
        return

    n  = len(ctries)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    cmap  = matplotlib.colormaps["tab10"]
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 2.8 * nr), squeeze=False,
                             sharex=False)

    for idx, country in enumerate(ctries):
        ax  = axes[idx // nc][idx % nc]
        grp = panel[panel["country"] == country].sort_values("date")
        ax.plot(grp["date"], grp["lambda_country"] * 100, color=cmap(idx), lw=1.3)
        ax.fill_between(grp["date"], 0, grp["lambda_country"] * 100,
                        alpha=0.12, color=cmap(idx))
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_ylabel("λ^i (% p.a.)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        _style_ax(ax)

    for k in range(len(ctries), nr * nc):
        axes[k // nc][k % nc].set_visible(False)

    fig.suptitle("Country-Specific Disaster Intensity λ^i",
                 fontsize=13, fontweight="bold")
    _savefig(fig, "country_intensity_panels.png", out_dir)


# ---------------------------------------------------------------------------
# 4. Estimated parameters bar chart
# ---------------------------------------------------------------------------

def plot_estimated_parameters(
    params_df: pd.DataFrame,
    out_dir:   str = FIGS_DIR,
) -> None:
    df = params_df[params_df.get("converged", pd.Series(True, index=params_df.index))
                  .fillna(True)].copy()
    if df.empty:
        return
    df["eta_f"] = df["eta_i"]

    with np.errstate(invalid="ignore"):
        denom           = df["eta_f"] + df["eta_g"]
        df["glob_share"] = np.where(denom > 0, df["eta_g"] / denom, np.nan)

    df = df.sort_values("glob_share", ascending=False).reset_index(drop=True)
    countries = df["country"].tolist()
    x = np.arange(len(countries))
    w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        (axes[0, 0], "h0",         "h₀* (baseline hazard)",  "#2980b9"),
        (axes[0, 1], "eta_f",      "η_f (country loading)",  "#27ae60"),
        (axes[1, 0], "eta_g",      "η_g (global loading)",   "#e74c3c"),
        (axes[1, 1], "glob_share", "η_g / (η_f + η_g)",      "#8e44ad"),
    ]
    for ax, col, title, color in specs:
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, width=w, color=color, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        _style_ax(ax)
        finite = vals[np.isfinite(vals)]
        offset = np.nanmax(np.abs(finite)) * 0.03 if finite.size else 0.0
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                        f"{v:.3g}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Estimated Hazard Parameters by Country", fontsize=13, fontweight="bold")
    _savefig(fig, "estimated_parameters.png", out_dir)


# ---------------------------------------------------------------------------
# 5. Model vs observed scatter (colored by tenor)
# ---------------------------------------------------------------------------

def plot_model_vs_observed_scatter(
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
) -> None:
    df = model_vs_obs.dropna(subset=["spread_obs", "spread_model"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    tenors  = sorted(df["tenor"].unique())
    for tau in tenors:
        sub = df[df["tenor"] == tau]
        col = TENOR_COLORS.get(tau, "#333333")
        ax.scatter(sub["spread_obs"], sub["spread_model"],
                   s=5, alpha=0.35, color=col, label=f"{int(tau)}Y")

    lim = max(df["spread_obs"].quantile(0.99), df["spread_model"].quantile(0.99)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="45°")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed CDS spread (bps)", fontsize=11)
    ax.set_ylabel("Model CDS spread (bps)", fontsize=11)
    ax.set_title("Model vs Observed CDS Spreads", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=3, title="Tenor", title_fontsize=9)
    _style_ax(ax)
    _savefig(fig, "model_vs_observed_scatter.png", out_dir)


# ---------------------------------------------------------------------------
# 6. Model vs observed 5Y time series (per country)
# ---------------------------------------------------------------------------

def plot_model_vs_observed_5y_timeseries(
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
    countries:    Sequence[str] | None = None,
) -> None:
    df5 = model_vs_obs[model_vs_obs["tenor"] == 5.0].copy()
    if df5.empty:
        return

    avail  = sorted(df5["country"].unique())
    ctries = [c for c in (countries or REPR_ORDER) if c in avail]
    if not ctries:
        ctries = avail[:8]

    n  = len(ctries)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 3 * nr), squeeze=False)

    for idx, country in enumerate(ctries):
        ax  = axes[idx // nc][idx % nc]
        sub = df5[df5["country"] == country].sort_values("date")
        ax.plot(sub["date"], sub["spread_obs"],   color="#1a6faf", lw=1.4, label="Observed")
        ax.plot(sub["date"], sub["spread_model"], color="#c0392b", lw=1.2, ls="--", label="Model")
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_ylabel("5Y CDS (bps)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        if idx == 0:
            ax.legend(fontsize=8)
        _style_ax(ax)

    for k in range(len(ctries), nr * nc):
        axes[k // nc][k % nc].set_visible(False)

    fig.suptitle("Observed vs Model 5Y CDS Spread", y=1.01,
                 fontsize=13, fontweight="bold")
    _savefig(fig, "model_vs_observed_5y_timeseries.png", out_dir)


# ---------------------------------------------------------------------------
# 7. Residuals by tenor (boxplot)
# ---------------------------------------------------------------------------

def plot_residuals_by_tenor(
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
) -> None:
    df = model_vs_obs.dropna(subset=["residual_bps"])
    if df.empty:
        return
    tenors  = sorted(df["tenor"].unique())
    data    = [df[df["tenor"] == t]["residual_bps"].to_numpy() for t in tenors]
    labels  = [f"{int(t)}Y" for t in tenors]
    colors  = [TENOR_COLORS.get(t, "#555555") for t in tenors]

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4),
                    whiskerprops=dict(lw=1), capprops=dict(lw=1))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Tenor", fontsize=11)
    ax.set_ylabel("Residual (model − observed, bps)", fontsize=11)
    ax.set_title("Calibration Residuals by Tenor", fontsize=12, fontweight="bold")
    _style_ax(ax)
    _savefig(fig, "residuals_by_tenor.png", out_dir)


# ---------------------------------------------------------------------------
# 8. Residuals over time (rolling MAE)
# ---------------------------------------------------------------------------

def plot_residuals_over_time(
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
    window:       int = 12,
) -> None:
    df = model_vs_obs.dropna(subset=["residual_bps"]).copy()
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])
    mae_ts = (df.groupby("date")["residual_bps"]
               .apply(lambda x: np.mean(np.abs(x)))
               .reset_index()
               .sort_values("date"))
    mae_ts["rolling_mae"] = mae_ts["residual_bps"].rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(mae_ts["date"], mae_ts["residual_bps"], color="#bdc3c7", lw=0.8, alpha=0.7,
            label="Monthly MAE")
    ax.plot(mae_ts["date"], mae_ts["rolling_mae"], color="#c0392b", lw=1.6,
            label=f"{window}-month rolling MAE")
    ax.set_ylabel("MAE (bps)", fontsize=11)
    ax.set_title("Calibration Error over Time", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(fontsize=9)
    _style_ax(ax)
    _savefig(fig, "residuals_over_time.png", out_dir)


# ---------------------------------------------------------------------------
# 9. CDS data coverage heatmap
# ---------------------------------------------------------------------------

def plot_cds_data_coverage_heatmap(
    panel:   pd.DataFrame,
    out_dir: str = FIGS_DIR,
) -> None:
    TENOR_COLS = {
        "cds_spread_1y": "1Y", "cds_spread_2y": "2Y", "cds_spread_3y": "3Y",
        "cds_spread_5y": "5Y", "cds_spread_7y": "7Y", "cds_spread_10y": "10Y",
    }
    avail = {k: v for k, v in TENOR_COLS.items() if k in panel.columns}
    if not avail:
        return

    countries = sorted(panel["country"].unique())
    tenors    = list(avail.values())
    matrix    = np.zeros((len(countries), len(tenors)), dtype=int)
    for i, c in enumerate(countries):
        grp = panel[panel["country"] == c]
        for j, col in enumerate(avail.keys()):
            matrix[i, j] = int(grp[col].notna().sum())

    fig, ax = plt.subplots(figsize=(max(6, len(tenors) * 1.2),
                                    max(4, len(countries) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels(tenors, fontsize=10)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=9)
    ax.set_xlabel("Tenor", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    ax.set_title("CDS Data Coverage (monthly obs)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="# observations")
    for i in range(len(countries)):
        for j in range(len(tenors)):
            n = matrix[i, j]
            ax.text(j, i, str(n) if n > 0 else "—",
                    ha="center", va="center", fontsize=7,
                    color="white" if n > matrix.max() * 0.6 else "black")
    _savefig(fig, "cds_data_coverage_heatmap.png", out_dir)


# ---------------------------------------------------------------------------
# 10. RMSE by country × tenor heatmap
# ---------------------------------------------------------------------------

def plot_rmse_by_country_tenor(
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
) -> None:
    df = model_vs_obs.dropna(subset=["spread_obs", "spread_model"])
    if df.empty:
        return

    rmse_tbl = (df.groupby(["country", "tenor"])
                  .apply(lambda g: np.sqrt(np.mean((g["spread_model"] - g["spread_obs"]) ** 2)))
                  .reset_index(name="rmse_bps"))

    countries = sorted(rmse_tbl["country"].unique())
    tenors    = sorted(rmse_tbl["tenor"].unique())
    matrix    = np.full((len(countries), len(tenors)), np.nan)
    for _, row in rmse_tbl.iterrows():
        i = countries.index(row["country"])
        j = tenors.index(row["tenor"])
        matrix[i, j] = row["rmse_bps"]

    fig, ax = plt.subplots(figsize=(max(6, len(tenors) * 1.2),
                                    max(4, len(countries) * 0.45)))
    masked   = np.ma.masked_invalid(matrix)
    cmap_    = plt.cm.YlOrRd.copy()
    cmap_.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, aspect="auto", cmap=cmap_)
    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels([f"{int(t)}Y" for t in tenors], fontsize=10)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=9)
    ax.set_title("RMSE by Country and Tenor (bps)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="RMSE (bps)")
    for i in range(len(countries)):
        for j in range(len(tenors)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(matrix) * 0.7 else "black")
    _savefig(fig, "rmse_by_country_tenor.png", out_dir)


# ---------------------------------------------------------------------------
# 11. Latest CDS term structure fit
# ---------------------------------------------------------------------------

def plot_latest_cds_term_structure_fit(
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
    countries:    Sequence[str] | None = None,
) -> None:
    df = model_vs_obs.dropna(subset=["spread_obs", "spread_model"]).copy()
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])

    avail  = sorted(df["country"].unique())
    ctries = [c for c in (countries or REPR_ORDER) if c in avail]
    if not ctries:
        ctries = avail[:8]

    n  = len(ctries)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 3.5 * nr), squeeze=False)

    for idx, country in enumerate(ctries):
        ax  = axes[idx // nc][idx % nc]
        sub = df[df["country"] == country]
        if sub.empty:
            ax.set_visible(False)
            continue
        last_date = sub["date"].max()
        latest    = sub[sub["date"] == last_date].sort_values("tenor")
        tenors    = latest["tenor"].to_numpy()
        ax.plot(tenors, latest["spread_obs"],   "o-", color="#1a6faf", lw=1.5,
                markersize=5, label="Observed")
        ax.plot(tenors, latest["spread_model"], "s--", color="#c0392b", lw=1.3,
                markersize=5, label="Model")
        ax.set_title(f"{country}  ({last_date.date()})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=8)
        ax.set_ylabel("CDS spread (bps)", fontsize=8)
        ax.set_xticks(tenors)
        if idx == 0:
            ax.legend(fontsize=8)
        _style_ax(ax)

    for k in range(len(ctries), nr * nc):
        axes[k // nc][k % nc].set_visible(False)

    fig.suptitle("Model vs Observed CDS Term Structure (Latest Date)",
                 y=1.01, fontsize=13, fontweight="bold")
    _savefig(fig, "latest_cds_term_structure_fit.png", out_dir)


# ---------------------------------------------------------------------------
# 12. Hazard proxy scatter
# ---------------------------------------------------------------------------

def plot_hazard_proxy_scatter(
    panel:     pd.DataFrame,
    params_df: pd.DataFrame,
    out_dir:   str = FIGS_DIR,
) -> None:
    """Scatter of model hazard vs CDS-implied hazard proxy across all countries."""
    if "cds_spread_5y" not in panel.columns:
        return
    p_idx = params_df.set_index("country")
    all_h_cds, all_h_model, all_c = [], [], []

    for country, grp in panel.groupby("country"):
        if country not in p_idx.index:
            continue
        row = p_idx.loc[country]
        if not row.get("converged", True):
            continue
        h0    = float(row["h0"])
        eta_f = float(row["eta_i"])
        eta_g = float(row["eta_g"])
        R_i   = float(row["R"]) if "R" in row.index and pd.notna(row.get("R")) else 0.4
        sub = grp.dropna(subset=["cds_spread_5y", "lambda_country", "lambda_global"])
        if sub.empty:
            continue
        h_cds   = sub["cds_spread_5y"].to_numpy() / (1.0 - R_i)
        h_model = h0 + eta_f * sub["lambda_country"].to_numpy() \
                     + eta_g * sub["lambda_global"].to_numpy()
        all_h_cds.extend(h_cds)
        all_h_model.extend(h_model)
        all_c.extend([country] * len(h_cds))

    if not all_h_cds:
        return

    h_cds_a   = np.array(all_h_cds)
    h_model_a = np.array(all_h_model)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(h_cds_a * 100, h_model_a * 100, s=4, alpha=0.25, color="#2980b9")
    lim = np.nanpercentile(np.concatenate([h_cds_a, h_model_a]), 99) * 100 * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("CDS-implied hazard proxy  s(5Y)/(1−R)  (% p.a.)", fontsize=10)
    ax.set_ylabel("Model hazard h*(t)  (% p.a.)", fontsize=10)
    ax.set_title("Hazard Proxy Diagnostic (all countries)", fontsize=11, fontweight="bold")
    ax.annotate("Diagnostic approximation — not the formal pricing equation",
                xy=(0.02, 0.96), xycoords="axes fraction", fontsize=7,
                color="grey", va="top")
    _style_ax(ax)
    _savefig(fig, "hazard_proxy_scatter.png", out_dir)


# ---------------------------------------------------------------------------
# 13. Hazard proxy time series (selected countries)
# ---------------------------------------------------------------------------

def plot_hazard_proxy_timeseries(
    panel:     pd.DataFrame,
    params_df: pd.DataFrame,
    out_dir:   str = FIGS_DIR,
    countries: Sequence[str] | None = None,
) -> None:
    if "cds_spread_5y" not in panel.columns:
        return
    p_idx  = params_df.set_index("country")
    avail  = [c for c in panel["country"].unique() if c in p_idx.index]
    ctries = [c for c in (countries or REPR_ORDER) if c in avail]
    if not ctries:
        ctries = avail[:6]

    n  = len(ctries)
    nc = min(n, 2)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 3 * nr), squeeze=False)

    for idx, country in enumerate(ctries):
        ax  = axes[idx // nc][idx % nc]
        row = p_idx.loc[country]
        h0    = float(row["h0"])
        eta_f = float(row["eta_i"])
        eta_g = float(row["eta_g"])
        R_i   = float(row["R"]) if "R" in row.index and pd.notna(row.get("R")) else 0.4

        grp = panel[panel["country"] == country].sort_values("date")
        sub = grp.dropna(subset=["cds_spread_5y", "lambda_country", "lambda_global"])
        if sub.empty:
            ax.set_visible(False)
            continue
        h_cds   = sub["cds_spread_5y"] / (1.0 - R_i)
        h_model = h0 + eta_f * sub["lambda_country"] + eta_g * sub["lambda_global"]

        ax.plot(sub["date"], h_cds   * 100, color="#1a6faf", lw=1.3, label="CDS proxy")
        ax.plot(sub["date"], h_model * 100, color="#c0392b", lw=1.2, ls="--", label="Model h*")
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_ylabel("Hazard (% p.a.)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        if idx == 0:
            ax.legend(fontsize=8)
        _style_ax(ax)

    for k in range(len(ctries), nr * nc):
        axes[k // nc][k % nc].set_visible(False)

    fig.suptitle("Model Hazard vs CDS-Implied Proxy  (s(5Y)/(1−R))",
                 y=1.01, fontsize=12, fontweight="bold")
    ax_note = fig.add_axes([0.01, -0.02, 0.98, 0.02])
    ax_note.axis("off")
    ax_note.text(0.5, 0.5,
                 "Note: CDS-implied hazard is a diagnostic approximation, not the formal pricing equation.",
                 ha="center", va="center", fontsize=8, color="grey",
                 transform=ax_note.transAxes)
    _savefig(fig, "hazard_proxy_timeseries.png", out_dir)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def save_all_plots(
    panel:        pd.DataFrame,
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    out_dir:      str = FIGS_DIR,
    countries:    Sequence[str] | None = None,
) -> None:
    """Generate and save all calibration figures."""
    os.makedirs(out_dir, exist_ok=True)
    print("Generating calibration figures …")

    plot_valuation_vs_intensity(panel, out_dir, countries)
    plot_global_intensity_timeseries(panel, out_dir)
    plot_country_intensity_panels(panel, out_dir, countries)
    plot_estimated_parameters(params_df, out_dir)
    plot_model_vs_observed_scatter(model_vs_obs, out_dir)
    plot_model_vs_observed_5y_timeseries(model_vs_obs, out_dir, countries)
    plot_residuals_by_tenor(model_vs_obs, out_dir)
    plot_residuals_over_time(model_vs_obs, out_dir)
    plot_cds_data_coverage_heatmap(panel, out_dir)
    plot_rmse_by_country_tenor(model_vs_obs, out_dir)
    plot_latest_cds_term_structure_fit(model_vs_obs, out_dir, countries)
    plot_hazard_proxy_scatter(panel, params_df, out_dir)
    plot_hazard_proxy_timeseries(panel, params_df, out_dir, countries)

    print(f"Figures saved to {os.path.relpath(out_dir, PARENT_DIR)}/")
