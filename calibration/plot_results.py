"""
plot_results.py
===============
Thesis-ready calibration figures for Sections 7.6, 8.1–8.4.

Figures produced
----------------
7.6.1  model_vs_observed_5y_subset.png          — model vs observed 5Y CDS time series
7.6.2  cds_term_structure_low_high_disaster.png — CDS term structures in low/high disaster states
7.6.3  parameter_heterogeneity_groups.png        — eta_i / eta_g by country and group
8.1.1  simulated_paths_subset.png                — simulated λ, h*, CDS paths
8.2.1  empirical_vs_simulated_5y_distribution.png — KDE-style histogram comparison
8.3.1  flight_to_quality_global_shock.png        — risk-free vs defaultable yield curves
8.3.2  flight_to_quality_scatter.png             — rf yield and CDS spread vs λ^g
8.4.1  bond_yield_curves_turkey_vs_developed.png — long-maturity yield curves
8.4.2  cds_spread_term_structures_by_state.png   — CDS spread curves by disaster state
8.4.3  bond_spread_vs_cds_spread_check.png       — bond spread vs CDS spread consistency

CLI
---
    python -m calibration.plot_results
"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.parameters import DisasterModelParams
from core.closed_form import (
    b_star_cf, a_star_cf,
    defaultable_coeffs_cf,
)
from models.cds import cds_spread_batch

# ── Paths ────────────────────────────────────────────────────────────────────
PANEL_PATH     = os.path.join(BASE_DIR, "processed_data", "panel.csv")
PARAMS_PATH    = os.path.join(BASE_DIR, "results", "parameters.csv")
MODEL_OBS_PATH = os.path.join(BASE_DIR, "results", "model_vs_observed.csv")
SIM_PATHS_CSV  = os.path.join(BASE_DIR, "results", "simulation", "simulated_paths.csv")
SIM_5Y_CSV     = os.path.join(BASE_DIR, "results", "simulation", "simulated_5y_all.csv")
FIGS_DIR       = os.path.join(BASE_DIR, "results", "figures")

# ── Country groupings ────────────────────────────────────────────────────────
DEVELOPED = ["US", "DE", "UK", "JP", "FR", "NL", "SE"]
EMERGING  = ["BR", "CL", "ES", "ID", "MX", "MY", "TH", "TR"]
ALL_15    = DEVELOPED + EMERGING
CORE      = ["US", "DE", "UK", "CL", "ID", "TR"]
GROUP_MAP = {c: "Developed" for c in DEVELOPED}
GROUP_MAP.update({c: "Emerging" for c in EMERGING})

CORE_COLORS = {
    "US": "#1f77b4",   # blue
    "DE": "#4292c6",   # medium blue
    "UK": "#9ecae1",   # light blue
    "CL": "#fd8d3c",   # orange
    "ID": "#d62728",   # red
    "TR": "#843c39",   # dark maroon
}
ALL_COLORS = dict(CORE_COLORS)
ALL_COLORS.update({
    "JP": "#c6dbef", "FR": "#08519c", "NL": "#41ab5d", "SE": "#a1d99b",
    "BR": "#e6550d", "ES": "#fdae6b", "MX": "#e7ba52",
    "MY": "#756bb1", "TH": "#bcbddc",
})
TENOR_COLORS = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c",
                5.0: "#d62728", 7.0: "#9467bd", 10.0: "#8c564b"}

DPI = 150


# ── Shared helpers ────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGS_DIR, exist_ok=True)
    path = os.path.join(FIGS_DIR, name)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.relpath(path, PARENT_DIR)}")


def _build_base_params() -> DisasterModelParams:
    p = DisasterModelParams()
    p.compute_b_sdf()
    return p


def _build_country_params(
    base: DisasterModelParams,
    row: pd.Series,
    lam_bar_f: float,
    lam_bar_g: float,
) -> DisasterModelParams:
    return dataclasses.replace(
        base,
        h0_star   = float(row["h0"]),
        eta1      = float(row["eta_i"]),
        eta2      = float(row["eta_g"]),
        R         = float(row["R"]),
        lam_bar_f = float(lam_bar_f),
        lam_bar_g = float(lam_bar_g),
        b_sdf     = base.b_sdf,
    )


def _country_params_dict(
    params_df: pd.DataFrame,
    panel:     pd.DataFrame,
    base:      DisasterModelParams,
) -> dict[str, DisasterModelParams]:
    lam_bar_g = float(panel["lambda_global"].mean())
    out: dict[str, DisasterModelParams] = {}
    for _, row in params_df.iterrows():
        c = row["country"]
        grp = panel[panel["country"] == c]
        lam_bar_f = float(grp["lambda_country"].mean()) if not grp.empty else 1e-4
        out[c] = _build_country_params(base, row, lam_bar_f, lam_bar_g)
    return out


def _rf_yield_vectorised(
    p: DisasterModelParams,
    tau: float,
    lf: np.ndarray,
    lg: np.ndarray,
) -> np.ndarray:
    """Risk-free yield at tau for multiple (lf, lg) pairs."""
    a = a_star_cf(p, tau)
    b = b_star_cf(p, tau)
    return -(a + b * (lf + lg)) / tau


def _def_yield_vectorised(
    p: DisasterModelParams,
    tau: float,
    lf: np.ndarray,
    lg: np.ndarray,
) -> np.ndarray:
    """Defaultable yield at tau for multiple (lf, lg) pairs."""
    a_D, b_Df, b_Dg = defaultable_coeffs_cf(p, tau)
    return -(a_D + b_Df * lf + b_Dg * lg) / tau


def _safe_yieldcurve(
    p: DisasterModelParams,
    tau_grid: np.ndarray,
    lf: float,
    lg: float,
    kind: str = "rf",
) -> np.ndarray:
    """Yield curve with NaN at blow-up taus."""
    from core.riccati import blowup_time, discriminant
    from core.closed_form import K_const, _defaultable_Ai, _phi_sig2

    phi, sig2 = _phi_sig2(p)
    K = K_const(p)

    if kind == "rf":
        tau_star = blowup_time(phi, sig2, K)
        a_vals = np.array([a_star_cf(p, float(t)) for t in tau_grid])
        b_vals = np.array([b_star_cf(p, float(t)) for t in tau_grid])
        yields = -(a_vals + b_vals * (lf + lg)) / tau_grid
    else:
        A0, Af, Ag = _defaultable_Ai(p)
        from core.riccati import blowup_time as bt
        tau_star = min(bt(phi, sig2, -Af), bt(phi, sig2, -Ag))
        a_vals = np.zeros(len(tau_grid))
        yields = np.zeros(len(tau_grid))
        for i, t in enumerate(tau_grid):
            if t >= tau_star * 0.99:
                yields[i] = np.nan
                continue
            a_D, b_Df, b_Dg = defaultable_coeffs_cf(p, float(t))
            yields[i] = -(a_D + b_Df * lf + b_Dg * lg) / t

    # mask beyond blowup
    if np.isfinite(tau_star):
        yields[tau_grid >= tau_star * 0.99] = np.nan
    return yields


# ── Figure 7.6.1 ─────────────────────────────────────────────────────────────

def plot_model_vs_observed_5y_subset(
    model_vs_obs: pd.DataFrame,
    panel:        pd.DataFrame,
) -> None:
    """2×3 panel: model vs observed 5Y CDS time series for core 6."""
    mv5 = model_vs_obs[model_vs_obs["tenor"] == 5.0].copy()
    mv5["date"] = pd.to_datetime(mv5["date"])
    mv5 = mv5.sort_values(["country", "date"])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Model vs Observed 5-Year CDS Spreads", fontsize=13, fontweight="bold")

    legend_added = False
    for idx, c in enumerate(CORE):
        ax   = axes[idx // 2][idx % 2]
        sub  = mv5[mv5["country"] == c].dropna(subset=["spread_obs", "spread_model"])
        grp  = GROUP_MAP.get(c, "")
        col  = CORE_COLORS.get(c, "steelblue")

        if sub.empty:
            ax.set_visible(False)
            continue

        ax.plot(sub["date"], sub["spread_obs"],   color=col, lw=1.4,
                label="Observed", alpha=0.9)
        ax.plot(sub["date"], sub["spread_model"], color=col, lw=1.4,
                linestyle="--", label="Model", alpha=0.85)

        ax.set_title(f"{c}  ({grp})", fontsize=10, fontweight="bold")
        ax.set_ylabel("CDS spread (bps)", fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if not legend_added:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
            legend_added = True

    _savefig(fig, "model_vs_observed_5y_subset.png")


# ── Figure 7.6.2 ─────────────────────────────────────────────────────────────

def plot_cds_term_structure_low_high_disaster(
    model_vs_obs: pd.DataFrame,
    panel:        pd.DataFrame,
) -> None:
    """CDS term structures in low- vs high-disaster states for core 6."""
    tenors_avail = sorted(model_vs_obs["tenor"].unique())
    mv = model_vs_obs.copy()
    mv["date"] = pd.to_datetime(mv["date"])

    # Identify low/high lambda_global dates globally
    global_ts = panel.drop_duplicates("date").set_index("date")["lambda_global"].dropna().sort_index()
    lo_thresh = float(np.percentile(global_ts, 20))
    hi_thresh = float(np.percentile(global_ts, 80))
    lo_dates = set(global_ts.index[global_ts <= lo_thresh])
    hi_dates = set(global_ts.index[global_ts >= hi_thresh])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("CDS Term Structures: Low vs High Global Disaster State",
                 fontsize=13, fontweight="bold")

    for idx, c in enumerate(CORE):
        ax  = axes[idx // 2][idx % 2]
        sub = mv[mv["country"] == c]
        col = CORE_COLORS.get(c, "steelblue")
        grp = GROUP_MAP.get(c, "")

        # Low state: average observed and model by tenor
        lo_sub = sub[sub["date"].isin(lo_dates)]
        hi_sub = sub[sub["date"].isin(hi_dates)]

        def _mean_curve(df):
            return df.groupby("tenor").agg(
                obs=("spread_obs",   "mean"),
                mod=("spread_model", "mean"),
            ).reindex(tenors_avail)

        lo_curve = _mean_curve(lo_sub.dropna(subset=["spread_obs", "spread_model"]))
        hi_curve = _mean_curve(hi_sub.dropna(subset=["spread_obs", "spread_model"]))

        xt = tenors_avail
        if not lo_curve.empty and lo_curve["obs"].notna().any():
            ax.plot(xt, lo_curve["obs"].values,  color="steelblue", lw=1.6,
                    marker="o", ms=4, label="Obs low")
            ax.plot(xt, lo_curve["mod"].values,  color="steelblue", lw=1.6,
                    linestyle="--", marker="s", ms=4, label="Model low")
        if not hi_curve.empty and hi_curve["obs"].notna().any():
            ax.plot(xt, hi_curve["obs"].values,  color=col, lw=1.6,
                    marker="o", ms=4, label="Obs high")
            ax.plot(xt, hi_curve["mod"].values,  color=col, lw=1.6,
                    linestyle="--", marker="s", ms=4, label="Model high")

        ax.set_title(f"{c}  ({grp})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=8)
        ax.set_ylabel("CDS spread (bps)", fontsize=8)
        ax.set_xticks(tenors_avail)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    _savefig(fig, "cds_term_structure_low_high_disaster.png")


# ── Figure 7.6.3 ─────────────────────────────────────────────────────────────

def plot_parameter_heterogeneity_groups(params_df: pd.DataFrame) -> None:
    """Grouped bar chart of eta_i and eta_g by country, split by group."""
    order = DEVELOPED + EMERGING
    fitted = params_df.set_index("country").reindex(order).dropna(subset=["eta_i"])

    x       = np.arange(len(fitted))
    labels  = list(fitted.index)
    eta_i   = fitted["eta_i"].to_numpy()
    eta_g   = fitted["eta_g"].to_numpy()
    n_dev   = len(DEVELOPED)
    is_dev  = np.array([c in DEVELOPED for c in labels])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Calibrated Hazard Loadings by Country", fontsize=13, fontweight="bold")

    for ax, vals, title, ymax_pad in [
        (axes[0], eta_i, r"Country Loading $\eta_i$", 1.25),
        (axes[1], eta_g, r"Global Loading $\eta_{g,i}$", 1.15),
    ]:
        bar_colors = ["#2171b5" if d else "#d62728" for d in is_dev]
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.82, width=0.65, edgecolor="white")

        # Group separator
        ax.axvline(x=n_dev - 0.5, color="gray", linestyle="--", lw=1.2, alpha=0.6)
        ymax = float(np.nanmax(vals)) if len(vals) > 0 else 1.0
        ax.text(n_dev / 2 - 0.5, ymax * ymax_pad * 0.95,
                "Developed", ha="center", fontsize=8, color="#2171b5", style="italic")
        ax.text(n_dev + (len(labels) - n_dev) / 2 - 0.5, ymax * ymax_pad * 0.95,
                "Emerging", ha="center", fontsize=8, color="#d62728", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Parameter value", fontsize=9)
        ax.set_ylim(0, ymax * ymax_pad)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for xi, val in zip(x, vals):
            if not np.isfinite(val) or val < 1e-6:
                continue
            lbl = f"{val:.3f}"
            ax.annotate(lbl, (xi, val), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6)

    _savefig(fig, "parameter_heterogeneity_groups.png")


# ── Figure 8.1.1 ─────────────────────────────────────────────────────────────

def plot_simulated_paths_subset(sim_paths_csv: str = SIM_PATHS_CSV) -> None:
    """4-country, 4-variable simulated paths (single representative path)."""
    if not os.path.exists(sim_paths_csv):
        print(f"  SKIP simulated_paths: {sim_paths_csv} not found — "
              "run simulate_calibrated_model first.")
        return

    df = pd.read_csv(sim_paths_csv)
    countries_in = [c for c in ["US", "DE", "ID", "TR"] if c in df["country"].unique()]
    if not countries_in:
        print("  SKIP simulated_paths: no expected countries in CSV.")
        return

    variables = [
        ("lambda_global",  r"$\lambda^g$ (global intensity)",  "steelblue"),
        ("lambda_country", r"$\lambda^i$ (country intensity)",  "darkorange"),
        ("h_star",         r"Hazard rate $h^*$",                "mediumseagreen"),
        ("cds_5y_bps",     r"5Y CDS spread (bps)",              "tomato"),
    ]

    fig, axes = plt.subplots(4, len(countries_in),
                             figsize=(4 * len(countries_in), 9),
                             constrained_layout=True)
    fig.suptitle("Simulated Model Paths (Representative Path)", fontsize=12, fontweight="bold")

    if len(countries_in) == 1:
        axes = axes[:, np.newaxis]

    for col, c in enumerate(countries_in):
        sub = df[df["country"] == c].sort_values("t")
        t   = sub["t"].to_numpy()
        grp = GROUP_MAP.get(c, "")
        axes[0, col].set_title(f"{c}  ({grp})", fontsize=10, fontweight="bold")

        for row, (var, ylabel, color) in enumerate(variables):
            ax = axes[row, col]
            if var in sub.columns:
                vals = sub[var].to_numpy()
                # convert intensities to percentage for readability
                if var in ("lambda_global", "lambda_country", "h_star"):
                    vals = vals * 100
                    lbl  = ylabel.replace(")", " \\%)")
                else:
                    lbl = ylabel
                ax.plot(t, vals, color=color, lw=1.2)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 3:
                ax.set_xlabel("Month", fontsize=7)

    _savefig(fig, "simulated_paths_subset.png")


# ── Figure 8.2.1 ─────────────────────────────────────────────────────────────

def plot_empirical_vs_simulated_5y_distribution(
    panel:       pd.DataFrame,
    sim_5y_csv:  str = SIM_5Y_CSV,
) -> None:
    """Histogram comparison of observed vs simulated 5Y CDS distribution."""
    panel_5y = panel[["country", "cds_spread_5y"]].dropna()
    panel_5y = panel_5y.assign(cds_5y_bps=panel_5y["cds_spread_5y"] * 1e4)

    sim_df = None
    if os.path.exists(sim_5y_csv):
        sim_df = pd.read_csv(sim_5y_csv)
    else:
        warnings.warn(f"  Simulation file not found: {sim_5y_csv}; using model_vs_observed.")

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Empirical vs Simulated Distribution of 5-Year CDS Spreads",
                 fontsize=12, fontweight="bold")

    for idx, c in enumerate(CORE):
        ax  = axes[idx // 2][idx % 2]
        obs = panel_5y[panel_5y["country"] == c]["cds_5y_bps"].dropna().to_numpy()
        col = CORE_COLORS.get(c, "steelblue")
        grp = GROUP_MAP.get(c, "")

        if len(obs) < 5:
            ax.set_visible(False)
            continue

        # Observed histogram
        bins = np.histogram_bin_edges(obs, bins=25)
        ax.hist(obs, bins=bins, density=True, alpha=0.55, color=col, label="Observed",
                edgecolor="white", linewidth=0.5)

        # Simulated histogram
        if sim_df is not None:
            sim_s = sim_df[sim_df["country"] == c]["cds_5y_bps"].dropna().to_numpy()
            if len(sim_s) > 10:
                # clip extreme tail for visual clarity
                cap = np.percentile(obs, 99) * 2.5
                sim_s_clip = sim_s[sim_s <= cap]
                ax.hist(sim_s_clip, bins=bins, density=True, alpha=0.45,
                        color="gray", label="Simulated", edgecolor="white", linewidth=0.5)

        ax.set_title(f"{c}  ({grp})", fontsize=10, fontweight="bold")
        ax.set_xlabel("5Y CDS spread (bps)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.7)

    _savefig(fig, "empirical_vs_simulated_5y_distribution.png")


# ── Figure 8.3.1 ─────────────────────────────────────────────────────────────

def plot_flight_to_quality_global_shock(
    panel:        pd.DataFrame,
    country_p:    dict[str, DisasterModelParams],
) -> None:
    """RF yield, defaultable yield, and credit spread curves at 3 global disaster states."""
    countries_ftq = ["US", "DE", "TR"]
    tau_grid = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    # Global disaster state percentiles
    lg_all    = panel["lambda_global"].dropna().to_numpy()
    lg_states = {
        "Low (p20)":    float(np.percentile(lg_all, 20)),
        "Median (p50)": float(np.percentile(lg_all, 50)),
        "High (p80)":   float(np.percentile(lg_all, 80)),
    }
    state_colors = {"Low (p20)": "#4292c6", "Median (p50)": "#525252",
                    "High (p80)": "#d62728"}
    state_ls     = {"Low (p20)": ":", "Median (p50)": "--", "High (p80)": "-"}

    # Country median lambda_country
    lf_medians: dict[str, float] = {}
    for c in countries_ftq:
        grp = panel[panel["country"] == c]["lambda_country"].dropna()
        lf_medians[c] = float(grp.median()) if len(grp) > 0 else 0.01

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panel_titles = ["Risk-Free Yield Curve (%)", "Defaultable Yield Curve (%)",
                    "Credit Spread (bps)"]

    for state_name, lg in lg_states.items():
        col = state_colors[state_name]
        ls  = state_ls[state_name]
        for c_idx, c in enumerate(countries_ftq):
            p    = country_p.get(c)
            if p is None:
                continue
            lf   = lf_medians[c]
            col_c = CORE_COLORS.get(c, "steelblue")

            rf_y   = np.array([_safe_yieldcurve(p, tau_grid, lf, lg, "rf")]).flatten() * 100
            def_y  = np.array([_safe_yieldcurve(p, tau_grid, lf, lg, "def")]).flatten() * 100
            spread = (def_y - rf_y)   # already in %

            for ax_i, (ax, vals) in enumerate(zip(
                axes, [rf_y, def_y, spread * 100]   # spread → bps
            )):
                lbl = f"{c} — {state_name}" if c_idx == 0 else None
                ax.plot(tau_grid, vals,
                        color=col_c if len(countries_ftq) > 1 else col,
                        linestyle=ls, lw=1.6,
                        label=f"{c} {state_name}" if ax_i == 0 else None)

    # Cleaner legend: one per state
    axes[0].set_ylabel("Yield (%)", fontsize=9)
    axes[1].set_ylabel("Yield (%)", fontsize=9)
    axes[2].set_ylabel("Spread (bps)", fontsize=9)

    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=9)
        ax.set_xticks(tau_grid)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Build a clean legend combining country and state information
    from matplotlib.lines import Line2D
    legend_handles = []
    for c in countries_ftq:
        legend_handles.append(
            Line2D([0], [0], color=CORE_COLORS.get(c, "gray"), lw=2, label=c)
        )
    for state_name, ls in state_ls.items():
        legend_handles.append(
            Line2D([0], [0], color="gray", lw=1.5, linestyle=ls, label=state_name)
        )
    axes[0].legend(handles=legend_handles, fontsize=7, loc="best", framealpha=0.7,
                   ncol=2)

    _savefig(fig, "flight_to_quality_global_shock.png")


# ── Figure 8.3.2 ─────────────────────────────────────────────────────────────

def plot_flight_to_quality_scatter(
    panel:        pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    country_p:    dict[str, DisasterModelParams],
) -> None:
    """Scatter: lambda_global vs rf yield (left) and CDS spread (right)."""
    # Compute rf 5Y yield at observed states for all countries
    mv5 = model_vs_obs[model_vs_obs["tenor"] == 5.0].copy()
    mv5 = mv5.merge(
        panel[["country", "date", "lambda_global", "lambda_country"]],
        on=["country", "date"], how="left",
    ).dropna(subset=["lambda_global", "lambda_country", "spread_model"])

    rf_yield_all   = []
    spread_all     = []
    lg_all         = []
    country_labels = []

    for c, grp in mv5.groupby("country"):
        p = country_p.get(c)
        if p is None:
            continue
        lf = grp["lambda_country"].to_numpy()
        lg = grp["lambda_global"].to_numpy()
        rf_y = _rf_yield_vectorised(p, 5.0, lf, lg) * 100   # in %
        finite = np.isfinite(rf_y)
        rf_yield_all.extend(rf_y[finite].tolist())
        spread_all.extend(grp["spread_model"].to_numpy()[finite].tolist())
        lg_all.extend(lg[finite].tolist())
        country_labels.extend([c] * int(finite.sum()))

    lg_arr   = np.array(lg_all)
    rf_arr   = np.array(rf_yield_all)
    sp_arr   = np.array(spread_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Flight-to-Quality: Global Disaster Risk and Yields",
                 fontsize=12, fontweight="bold")

    scatter_kw = dict(s=5, alpha=0.35, edgecolors="none")
    ax1.scatter(lg_arr * 100, rf_arr,  color="steelblue", **scatter_kw)
    ax2.scatter(lg_arr * 100, sp_arr,  color="tomato",    **scatter_kw)

    # Regression lines
    def _regline(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            return
        m, b = np.polyfit(x[ok], y[ok], 1)
        xr = np.array([x[ok].min(), x[ok].max()])
        return xr, m * xr + b

    for ax, x, y in [(ax1, lg_arr * 100, rf_arr), (ax2, lg_arr * 100, sp_arr)]:
        res = _regline(x, y)
        if res:
            ax.plot(res[0], res[1], color="black", lw=1.5, linestyle="-",
                    label="OLS fit")

    ax1.set_xlabel(r"$\lambda^g$ (%)", fontsize=9)
    ax1.set_ylabel("Model RF 5Y yield (%)", fontsize=9)
    ax1.set_title("Risk-Free Yield vs Global Disaster Risk", fontsize=10)
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r"$\lambda^g$ (%)", fontsize=9)
    ax2.set_ylabel("Model 5Y CDS spread (bps)", fontsize=9)
    ax2.set_title("CDS Spread vs Global Disaster Risk", fontsize=10)
    ax2.legend(fontsize=8)

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _savefig(fig, "flight_to_quality_scatter.png")


# ── Figure 8.4.1 ─────────────────────────────────────────────────────────────

def plot_bond_yield_curves_turkey_vs_developed(
    panel:     pd.DataFrame,
    country_p: dict[str, DisasterModelParams],
) -> None:
    """Bond yield curves at median state for TR, US, DE, UK — up to 50 years."""
    countries = ["TR", "US", "DE", "UK"]
    tau_grid  = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50], dtype=float)

    fig, (ax_rf, ax_def) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Bond Yield Curves at Median Disaster State",
                 fontsize=12, fontweight="bold")

    for c in countries:
        p = country_p.get(c)
        if p is None:
            continue
        grp = panel[panel["country"] == c].dropna(
            subset=["lambda_global", "lambda_country"]
        )
        if grp.empty:
            continue
        lf   = float(grp["lambda_country"].median())
        lg   = float(grp["lambda_global"].median())
        col  = CORE_COLORS.get(c, "gray")
        lw   = 2.2 if c == "TR" else 1.4
        ls   = "-" if c == "TR" else "--"
        alpha = 1.0 if c == "TR" else 0.8

        rf_y  = _safe_yieldcurve(p, tau_grid, lf, lg, "rf")  * 100
        def_y = _safe_yieldcurve(p, tau_grid, lf, lg, "def") * 100

        ax_rf.plot(tau_grid, rf_y,  color=col, lw=lw, ls=ls, alpha=alpha, label=c)
        ax_def.plot(tau_grid, def_y, color=col, lw=lw, ls=ls, alpha=alpha, label=c)

    for ax, title, ylabel in [
        (ax_rf,  "Risk-Free Bond Yield Curve",   "Risk-free yield (%)"),
        (ax_def, "Defaultable Bond Yield Curve", "Defaultable yield (%)"),
    ]:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Annotate convergence note for TR
    ax_def.annotate("TR converges\n($\\delta(-A_i)>0$, $\\delta(-A_g)>0$)",
                    xy=(40, ax_def.get_ylim()[0] * 0.9),
                    fontsize=7, color=CORE_COLORS["TR"], style="italic")

    _savefig(fig, "bond_yield_curves_turkey_vs_developed.png")


# ── Figure 8.4.2 ─────────────────────────────────────────────────────────────

def plot_cds_spread_term_structures_by_state(
    panel:     pd.DataFrame,
    country_p: dict[str, DisasterModelParams],
) -> None:
    """CDS spread curves at low/median/high disaster states for US, DE, CL, ID, TR."""
    countries  = ["US", "DE", "CL", "ID", "TR"]
    tenors_arr = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    lg_all   = panel["lambda_global"].dropna().to_numpy()
    states   = {
        "Low (p20)":    float(np.percentile(lg_all, 20)),
        "Median (p50)": float(np.percentile(lg_all, 50)),
        "High (p80)":   float(np.percentile(lg_all, 80)),
    }
    state_colors = {"Low (p20)": "#4292c6", "Median (p50)": "#525252",
                    "High (p80)": "#d62728"}
    state_ls     = {"Low (p20)": ":", "Median (p50)": "--", "High (p80)": "-"}
    state_markers = {"Low (p20)": "o", "Median (p50)": "s", "High (p80)": "^"}

    ncols = min(3, len(countries))
    nrows = (len(countries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    fig.suptitle("Model CDS Spread Term Structures by Disaster State",
                 fontsize=12, fontweight="bold")
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for idx, c in enumerate(countries):
        ax  = axes_flat[idx]
        p   = country_p.get(c)
        grp = GROUP_MAP.get(c, "")
        col = CORE_COLORS.get(c, "steelblue")

        if p is None:
            ax.set_visible(False)
            continue

        lf_median = float(panel[panel["country"] == c]["lambda_country"].median())

        for state_name, lg in states.items():
            spreads = cds_spread_batch(
                p, tenors_arr,
                np.array([lf_median]),
                np.array([lg]),
                n_int=200,
            )[0] * 1e4   # bps

            ax.plot(tenors_arr, spreads,
                    color=state_colors[state_name],
                    linestyle=state_ls[state_name],
                    marker=state_markers[state_name], ms=5,
                    lw=1.6, label=state_name)

        ax.set_title(f"{c}  ({grp})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=8)
        ax.set_ylabel("CDS spread (bps)", fontsize=8)
        ax.set_xticks(tenors_arr)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.7)

    for idx in range(len(countries), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    _savefig(fig, "cds_spread_term_structures_by_state.png")


# ── Figure 8.4.3 ─────────────────────────────────────────────────────────────

def plot_bond_spread_vs_cds_spread_check(
    panel:        pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    country_p:    dict[str, DisasterModelParams],
) -> None:
    """Model bond credit spread vs model CDS spread, 5Y maturity, all countries pooled."""
    mv5 = model_vs_obs[model_vs_obs["tenor"] == 5.0].copy()
    mv5 = mv5.merge(
        panel[["country", "date", "lambda_global", "lambda_country"]],
        on=["country", "date"], how="left",
    ).dropna(subset=["lambda_global", "lambda_country", "spread_model"])

    cds_spreads: list[float]  = []
    bond_spreads: list[float] = []
    c_labels: list[str]       = []

    for c, grp in mv5.groupby("country"):
        p = country_p.get(c)
        if p is None:
            continue
        lf = grp["lambda_country"].to_numpy()
        lg = grp["lambda_global"].to_numpy()

        rf5  = _rf_yield_vectorised(p, 5.0, lf, lg)
        def5 = _def_yield_vectorised(p, 5.0, lf, lg)
        bond_sp = (def5 - rf5) * 1e4   # bps

        cds_s   = grp["spread_model"].to_numpy()
        finite  = np.isfinite(rf5) & np.isfinite(def5) & np.isfinite(cds_s)

        cds_spreads.extend(cds_s[finite].tolist())
        bond_spreads.extend(bond_sp[finite].tolist())
        c_labels.extend([c] * int(finite.sum()))

    x = np.array(cds_spreads)
    y = np.array(bond_spreads)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    fig.suptitle("Model Bond Credit Spread vs Model CDS Spread (5Y)",
                 fontsize=12, fontweight="bold")

    # Scatter coloured by country
    for c in ALL_15:
        mask = np.array(c_labels) == c
        if mask.sum() == 0:
            continue
        ax.scatter(x[mask], y[mask], s=8, alpha=0.5, edgecolors="none",
                   color=ALL_COLORS.get(c, "gray"), label=c)

    # 45° reference
    lo = min(x.min(), y.min()) if len(x) > 0 else 0
    hi = max(x.max(), y.max()) if len(x) > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7, label="45° line")

    ax.set_xlabel("Model CDS spread (bps)", fontsize=9)
    ax.set_ylabel("Model bond credit spread (bps)\n($y_D - y^*$)", fontsize=9)
    ax.legend(fontsize=6, ncol=4, loc="upper left", framealpha=0.6)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _savefig(fig, "bond_spread_vs_cds_spread_check.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for path in [PANEL_PATH, PARAMS_PATH, MODEL_OBS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    print("Loading data ...")
    panel        = pd.read_csv(PANEL_PATH,     parse_dates=["date"])
    params_df    = pd.read_csv(PARAMS_PATH)
    model_vs_obs = pd.read_csv(MODEL_OBS_PATH, parse_dates=["date"])

    # Run simulation if outputs missing
    if not os.path.exists(SIM_PATHS_CSV) or not os.path.exists(SIM_5Y_CSV):
        print("\nSimulation outputs not found — running simulate_calibrated_model ...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "calibration.simulate_calibrated_model"],
            cwd=PARENT_DIR,
        )
        if result.returncode != 0:
            warnings.warn("Simulation failed; paths/distribution plots may be incomplete.")

    base       = _build_base_params()
    country_p  = _country_params_dict(params_df, panel, base)

    os.makedirs(FIGS_DIR, exist_ok=True)

    print("\n1/10  model_vs_observed_5y_subset ...")
    plot_model_vs_observed_5y_subset(model_vs_obs, panel)

    print("2/10  cds_term_structure_low_high_disaster ...")
    plot_cds_term_structure_low_high_disaster(model_vs_obs, panel)

    print("3/10  parameter_heterogeneity_groups ...")
    plot_parameter_heterogeneity_groups(params_df)

    print("4/10  simulated_paths_subset ...")
    plot_simulated_paths_subset(SIM_PATHS_CSV)

    print("5/10  empirical_vs_simulated_5y_distribution ...")
    plot_empirical_vs_simulated_5y_distribution(panel, SIM_5Y_CSV)

    print("6/10  flight_to_quality_global_shock ...")
    plot_flight_to_quality_global_shock(panel, country_p)

    print("7/10  flight_to_quality_scatter ...")
    plot_flight_to_quality_scatter(panel, model_vs_obs, country_p)

    print("8/10  bond_yield_curves_turkey_vs_developed ...")
    plot_bond_yield_curves_turkey_vs_developed(panel, country_p)

    print("9/10  cds_spread_term_structures_by_state ...")
    plot_cds_spread_term_structures_by_state(panel, country_p)

    print("10/10 bond_spread_vs_cds_spread_check ...")
    plot_bond_spread_vs_cds_spread_check(panel, model_vs_obs, country_p)

    print("\nAll plots generated.")
    figs = sorted(f for f in os.listdir(FIGS_DIR) if f.endswith(".png"))
    for f in figs:
        size = os.path.getsize(os.path.join(FIGS_DIR, f))
        print(f"  {f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
