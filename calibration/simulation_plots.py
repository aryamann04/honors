"""
simulation_plots.py
===================
Thesis simulation figures for the calibrated model.

Figure inventory
----------------
 1. simulated_paths_representative_country.png  — λ^g, λ^i, h*, 5Y CDS path window
 2. flight_to_quality_mechanism.png             — rf yield / def yield / spread vs λ^g
 3. cds_term_structure_by_disaster_state.png    — CDS curves at low/med/high states
 4. yield_curves_by_disaster_state.png          — rf and def yield curves at three states
 5. spread_variance_decomposition.png           — global/country share bar chart
 6. hazard_contribution_decomposition.png       — baseline/local/global h* contribution

All intensities in % p.a. on plots; spreads in bps.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from calibration.simulation import (
    SimulationResult, spread_series, tenor_index, make_params_for_country,
    TENORS_DEFAULT,
)
from calibration.simulation_tables import build_variance_decomposition
from core.parameters import DisasterModelParams
from models.cds import cds_spread_batch, fair_cds_spread

FIGS_DIR    = os.path.join(BASE_DIR, "results", "figures")
REPR_ORDER  = ["US", "JP", "DE", "UK", "MX", "TR", "ZA", "AR", "BR", "IT", "KR", "RU"]
DPI         = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, name: str, out_dir: str = FIGS_DIR, dpi: int = DPI) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {os.path.relpath(path, PARENT_DIR)}")


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pick_representative(
    sim_results: dict[str, SimulationResult],
    panel:       pd.DataFrame,
) -> str | None:
    """Pick the country with most CDS observations as representative."""
    if "cds_spread_5y" not in panel.columns:
        return next(iter(sim_results), None)
    counts = (panel.groupby("country")["cds_spread_5y"]
              .apply(lambda s: s.notna().sum()))
    for c in REPR_ORDER:
        if c in sim_results and c in counts.index:
            return c
    best = counts.idxmax() if not counts.empty else None
    return best if best in sim_results else next(iter(sim_results), None)


# ---------------------------------------------------------------------------
# 1. Simulated paths
# ---------------------------------------------------------------------------

def plot_simulated_paths(
    sim_result: SimulationResult,
    out_dir:    str = FIGS_DIR,
    n_show:     int = 240,   # months to display (20 years)
    start:      int = 0,
) -> None:
    sim   = sim_result
    end   = min(start + n_show, len(sim.lam_g))
    t_yrs = np.arange(end - start) / 12.0

    s5 = sim.spreads[start:end, tenor_index(sim, 5.0)] * 1e4   # bps

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    specs = [
        (axes[0], sim.lam_g[start:end] * 100, "λ^g (% p.a.)",     "#2c3e50"),
        (axes[1], sim.lam_i[start:end] * 100, "λ^i (% p.a.)",     "#27ae60"),
        (axes[2], sim.h_star[start:end] * 100, "h* (% p.a.)",      "#8e44ad"),
        (axes[3], s5,                           "5Y CDS (bps)",      "#c0392b"),
    ]
    for ax, y, ylabel, color in specs:
        ax.plot(t_yrs, y, color=color, lw=1.0)
        ax.fill_between(t_yrs, 0, y, alpha=0.1, color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        _style_ax(ax)

    axes[0].set_title(
        f"Simulated Model Paths — {sim.country}  "
        f"(h₀={sim.h0:.3f}, η_f={sim.eta_f:.2f}, η_g={sim.eta_g:.2f})",
        fontsize=11, fontweight="bold",
    )
    axes[-1].set_xlabel("Years (simulated)", fontsize=10)
    _savefig(fig, "simulated_paths_representative_country.png", out_dir)


# ---------------------------------------------------------------------------
# 2. Flight-to-quality mechanism
# ---------------------------------------------------------------------------

def plot_flight_to_quality(
    base_params: DisasterModelParams,
    params_df:   pd.DataFrame,
    panel:       pd.DataFrame,
    out_dir:     str = FIGS_DIR,
    country:     str | None = None,
    tau:         float = 5.0,
    n_grid:      int   = 80,
    n_int:       int   = 200,
) -> None:
    """Plot rf yield, defaultable yield, credit spread, CDS spread vs λ^g.

    λ^i is held at its sample median.
    """
    # Choose country
    if country is None:
        for c in REPR_ORDER:
            if c in params_df["country"].values:
                country = c
                break
        if country is None:
            country = params_df["country"].iloc[0]

    row   = params_df.set_index("country").loc[country]
    grp   = panel[panel["country"] == country]
    lam_i_med = float(grp["lambda_country"].median()) if not grp.empty else 0.02
    lam_g_med = float(grp["lambda_global"].median())  if not grp.empty else 0.03

    lam_g_grid = np.linspace(0.001, max(lam_g_med * 4, 0.12), n_grid)
    lam_bar_f  = float(grp["lambda_country"].mean()) if not grp.empty else 0.01
    lam_bar_g  = float(grp["lambda_global"].mean())  if not grp.empty else 0.03
    R_i        = float(row["R"]) if "R" in row.index and pd.notna(row.get("R")) else 0.4

    c_params = make_params_for_country(
        base_params, float(row["h0"]), float(row["eta_i"]), float(row["eta_g"]),
        lam_bar_f, lam_bar_g, R_i,
    )

    from models.risk_free  import rf_yield
    from models.defaultable import def_yield, credit_spread

    rf_yields, def_yields, credit_spreads, cds_spreads = [], [], [], []
    for lg in lam_g_grid:
        rf_yields.append(rf_yield(base_params, tau, lam_i_med, lg) * 100)
        def_yields.append(def_yield(c_params, tau, lam_i_med, lg) * 100)
        credit_spreads.append(credit_spread(c_params, tau, lam_i_med, lg) * 1e4)
        try:
            cds_spreads.append(fair_cds_spread(c_params, tau, lam_i_med, lg, n_int=n_int) * 1e4)
        except Exception:
            cds_spreads.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(lam_g_grid * 100, rf_yields,    color="#1a6faf", lw=2, label=f"Risk-free yield y*({tau:.0f}Y)")
    ax1.plot(lam_g_grid * 100, def_yields,   color="#c0392b", lw=2, ls="--", label=f"Defaultable yield y_D*({tau:.0f}Y)")
    ax1.set_ylabel("Yield (% p.a.)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Flight-to-Quality Mechanism — {country}  (λ^i = {lam_i_med*100:.2f}%)",
                  fontsize=11, fontweight="bold")
    _style_ax(ax1)

    ax2.plot(lam_g_grid * 100, credit_spreads, color="#8e44ad", lw=2, label="Credit spread (bps)")
    ax2.plot(lam_g_grid * 100, cds_spreads,    color="#27ae60", lw=2, ls=":", label="CDS spread (bps)")
    ax2.set_xlabel("Global disaster intensity λ^g (% p.a.)", fontsize=11)
    ax2.set_ylabel("Spread (bps)", fontsize=11)
    ax2.legend(fontsize=9)
    _style_ax(ax2)

    _savefig(fig, "flight_to_quality_mechanism.png", out_dir)


# ---------------------------------------------------------------------------
# 3. CDS term structure by disaster state
# ---------------------------------------------------------------------------

def plot_cds_term_structure_by_state(
    sim_results: dict[str, SimulationResult],
    base_params: DisasterModelParams,
    params_df:   pd.DataFrame,
    panel:       pd.DataFrame,
    out_dir:     str = FIGS_DIR,
    n_int:       int = 200,
) -> None:
    """CDS term structure at 10th/50th/90th percentiles of simulated (λ^g, λ^i)."""
    avail  = [c for c in REPR_ORDER if c in sim_results]
    if not avail:
        avail = list(sim_results.keys())[:4]
    ctries = avail[:4]

    nc = min(len(ctries), 2)
    nr = (len(ctries) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 4 * nr), squeeze=False)

    state_pcts  = [10, 50, 90]
    state_labels = ["Low (p10)", "Median (p50)", "High (p90)"]
    state_colors = ["#2980b9", "#27ae60", "#c0392b"]
    state_ls     = ["-", "--", ":"]
    tenor_arr    = np.array(TENORS_DEFAULT)

    p_idx = params_df.set_index("country")

    for idx, country in enumerate(ctries):
        ax  = axes[idx // nc][idx % nc]
        sim = sim_results[country]
        if country not in p_idx.index:
            ax.set_visible(False)
            continue
        row       = p_idx.loc[country]
        grp       = panel[panel["country"] == country]
        lam_bar_f = float(grp["lambda_country"].mean()) if not grp.empty else sim.lam_bar_f
        lam_bar_g = float(grp["lambda_global"].mean())  if not grp.empty else sim.lam_bar_g
        R_i       = float(row["R"]) if "R" in row.index and pd.notna(row.get("R")) else 0.4
        c_params  = make_params_for_country(
            base_params, float(row["h0"]), float(row["eta_i"]), float(row["eta_g"]),
            lam_bar_f, lam_bar_g, R_i,
        )

        for pct, label, color, ls in zip(state_pcts, state_labels, state_colors, state_ls):
            lg_p = float(np.percentile(sim.lam_g, pct))
            li_p = float(np.percentile(sim.lam_i, pct))
            try:
                s_arr = cds_spread_batch(
                    c_params, tenor_arr,
                    np.array([li_p]), np.array([lg_p]), n_int=n_int,
                )[0] * 1e4
            except Exception:
                continue
            ax.plot(tenor_arr, s_arr, color=color, lw=1.8, ls=ls, label=label)

        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=9)
        ax.set_ylabel("CDS spread (bps)", fontsize=9)
        ax.set_xticks(tenor_arr)
        if idx == 0:
            ax.legend(fontsize=8)
        _style_ax(ax)

    for k in range(len(ctries), nr * nc):
        axes[k // nc][k % nc].set_visible(False)

    fig.suptitle("Model CDS Term Structure by Disaster State",
                 y=1.01, fontsize=13, fontweight="bold")
    _savefig(fig, "cds_term_structure_by_disaster_state.png", out_dir)


# ---------------------------------------------------------------------------
# 4. Yield curves by disaster state
# ---------------------------------------------------------------------------

def plot_yield_curves_by_state(
    base_params: DisasterModelParams,
    params_df:   pd.DataFrame,
    panel:       pd.DataFrame,
    sim_results: dict[str, SimulationResult],
    out_dir:     str = FIGS_DIR,
    country:     str | None = None,
    tau_max:     float = 30.0,
    n_tau:       int   = 60,
) -> None:
    """Risk-free and defaultable yield curves at low/med/high disaster states."""
    try:
        from models.risk_free   import rf_yield_curve
        from models.defaultable import def_yield_curve
    except ImportError:
        print("  SKIP yield_curves_by_disaster_state: model functions unavailable")
        return

    if country is None:
        for c in REPR_ORDER:
            if c in sim_results and c in params_df["country"].values:
                country = c
                break
    if country is None or country not in sim_results:
        print("  SKIP yield_curves_by_disaster_state: no suitable country")
        return

    sim   = sim_results[country]
    row   = params_df.set_index("country").loc[country]
    grp   = panel[panel["country"] == country]
    lam_bar_f = float(grp["lambda_country"].mean()) if not grp.empty else sim.lam_bar_f
    lam_bar_g = float(grp["lambda_global"].mean())  if not grp.empty else sim.lam_bar_g
    R_i       = float(row["R"]) if "R" in row.index and pd.notna(row.get("R")) else 0.4
    c_params  = make_params_for_country(
        base_params, float(row["h0"]), float(row["eta_i"]), float(row["eta_g"]),
        lam_bar_f, lam_bar_g, R_i,
    )

    tau_grid  = np.linspace(0.25, tau_max, n_tau)
    state_pcts   = [10, 50, 90]
    state_labels = ["Low (p10)", "Median (p50)", "High (p90)"]
    state_colors = ["#2980b9", "#27ae60", "#c0392b"]
    state_ls     = ["-", "--", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for pct, label, color, ls in zip(state_pcts, state_labels, state_colors, state_ls):
        lg_p = float(np.percentile(sim.lam_g, pct))
        li_p = float(np.percentile(sim.lam_i, pct))
        try:
            y_rf  = rf_yield_curve(base_params, tau_grid, li_p, lg_p) * 100
            y_def = def_yield_curve(c_params,   tau_grid, li_p, lg_p) * 100
        except Exception as exc:
            print(f"  WARNING: yield curve computation failed ({label}): {exc}")
            continue
        ax1.plot(tau_grid, y_rf,  color=color, lw=1.8, ls=ls, label=label)
        ax2.plot(tau_grid, y_def, color=color, lw=1.8, ls=ls, label=label)

    ax1.set_title(f"Risk-Free Yield Curve — {country}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Maturity (years)", fontsize=10)
    ax1.set_ylabel("Yield (% p.a.)", fontsize=10)
    ax1.legend(fontsize=9)
    _style_ax(ax1)

    ax2.set_title(f"Defaultable Yield Curve — {country}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Maturity (years)", fontsize=10)
    ax2.set_ylabel("Yield (% p.a.)", fontsize=10)
    _style_ax(ax2)

    _savefig(fig, "yield_curves_by_disaster_state.png", out_dir)


# ---------------------------------------------------------------------------
# 5. Spread variance decomposition bar chart
# ---------------------------------------------------------------------------

def plot_spread_variance_decomposition(
    sim_results: dict[str, SimulationResult],
    base_params: DisasterModelParams,
    out_dir:     str = FIGS_DIR,
) -> None:
    var_df = build_variance_decomposition(sim_results, base_params)
    if var_df.empty:
        return

    var_df = var_df.sort_values("global_share", ascending=False).reset_index(drop=True)
    countries     = var_df["country"].tolist()
    global_share  = var_df["global_share"].fillna(0).to_numpy()
    country_share = var_df["country_share"].fillna(0).to_numpy()

    x = np.arange(len(countries))
    w = 0.6
    fig, ax = plt.subplots(figsize=(max(8, len(countries) * 0.8), 4.5))
    ax.bar(x, global_share,  w, label="Global λ^g share",  color="#2980b9", alpha=0.85)
    ax.bar(x, country_share, w, bottom=global_share, label="Country λ^i share",
           color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of 5Y spread variance", fontsize=11)
    ax.set_title("5Y CDS Spread Variance Decomposition (Option-B Counterfactual)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")
    _style_ax(ax)

    for i, (g, c) in enumerate(zip(global_share, country_share)):
        if g > 0.04:
            ax.text(i, g / 2, f"{g:.0%}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        if c > 0.04:
            ax.text(i, g + c / 2, f"{c:.0%}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    _savefig(fig, "spread_variance_decomposition.png", out_dir)

    # Also save the numeric table
    _tbl_path = os.path.join(out_dir.replace("figures", "tables"),
                             "variance_decomposition_from_plot.csv")
    os.makedirs(os.path.dirname(_tbl_path), exist_ok=True)
    var_df.to_csv(_tbl_path, index=False)


# ---------------------------------------------------------------------------
# 6. Hazard loading decomposition (using historical panel)
# ---------------------------------------------------------------------------

def plot_hazard_contribution_decomposition(
    sim_results: dict[str, SimulationResult],
    panel:       pd.DataFrame,
    params_df:   pd.DataFrame,
    out_dir:     str = FIGS_DIR,
) -> None:
    """Bar chart of mean h* decomposition into h0, η_f λ^i, η_g λ^g per country."""
    p_idx = params_df.set_index("country")
    countries, h_base, h_local, h_global = [], [], [], []

    for country in sorted(sim_results.keys()):
        if country not in p_idx.index:
            continue
        row = p_idx.loc[country]
        if not row.get("converged", True):
            continue
        grp = panel[panel["country"] == country]
        if grp.empty:
            continue
        h0    = float(row["h0"])
        eta_f = float(row["eta_i"])
        eta_g = float(row["eta_g"])
        mean_li = float(grp["lambda_country"].mean()) if "lambda_country" in grp.columns else 0.0
        mean_lg = float(grp["lambda_global"].mean())  if "lambda_global"  in grp.columns else 0.0

        countries.append(country)
        h_base.append(h0)
        h_local.append(eta_f * mean_li)
        h_global.append(eta_g * mean_lg)

    if not countries:
        return

    x = np.arange(len(countries))
    w = 0.6
    h_base_a   = np.array(h_base)   * 100   # % p.a.
    h_local_a  = np.array(h_local)  * 100
    h_global_a = np.array(h_global) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(countries) * 0.8), 5))
    ax.bar(x, h_base_a,  w, label="Baseline h₀*",     color="#95a5a6", alpha=0.9)
    ax.bar(x, h_local_a, w, bottom=h_base_a,
           label="η_f · λ̄^i  (country)", color="#27ae60", alpha=0.85)
    ax.bar(x, h_global_a, w, bottom=h_base_a + h_local_a,
           label="η_g · λ̄^g  (global)", color="#2980b9", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean hazard contribution (% p.a.)", fontsize=11)
    ax.set_title("Decomposition of Mean Hazard Rate h*(t)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    _style_ax(ax)
    _savefig(fig, "hazard_contribution_decomposition.png", out_dir)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def save_all_simulation_plots(
    sim_results: dict[str, SimulationResult],
    base_params: DisasterModelParams,
    params_df:   pd.DataFrame,
    panel:       pd.DataFrame,
    out_dir:     str = FIGS_DIR,
    countries:   Sequence[str] | None = None,
) -> None:
    """Generate and save all simulation figures."""
    os.makedirs(out_dir, exist_ok=True)
    print("Generating simulation figures …")

    if not sim_results:
        print("  No simulation results — skipping simulation plots.")
        return

    # 1. Simulated paths — pick representative country
    rep = _pick_representative(sim_results, panel)
    if rep:
        plot_simulated_paths(sim_results[rep], out_dir)

    # 2. Flight to quality
    plot_flight_to_quality(base_params, params_df, panel, out_dir,
                           country=rep)

    # 3. CDS term structure by state
    plot_cds_term_structure_by_state(sim_results, base_params, params_df, panel, out_dir)

    # 4. Yield curves by state
    plot_yield_curves_by_state(base_params, params_df, panel, sim_results, out_dir,
                               country=rep)

    # 5. Variance decomposition
    plot_spread_variance_decomposition(sim_results, base_params, out_dir)

    # 6. Hazard contribution decomposition
    plot_hazard_contribution_decomposition(sim_results, panel, params_df, out_dir)

    print(f"Simulation figures saved to {os.path.relpath(out_dir, PARENT_DIR)}/")
