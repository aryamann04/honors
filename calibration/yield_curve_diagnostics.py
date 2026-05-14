"""
yield_curve_diagnostics.py
==========================
Diagnose yield-curve slope conditions derived from three LaTeX propositions:

  Prop. risk-free slope   : short-end upward slope of y*(τ)
  Prop. defaultable slope : short-end upward slope of y_D*(τ)
  Prop. long-maturity     : convergence of yields as τ → ∞

Sign conventions (matching closed_form.py):
  φ   = b_sdf · σ_λ² − κ          (< 0 for stability)
  K   = exp(−γZ)(1 − exp Z) > 0   (risk-free Riccati constant)
  A_f = −K + (1−R)η_f              (defaultable bond constant, WITH (1−R))
  A_g = −K + (1−R)η_g
  Note: Â_f = −K + η_f from _cds_Ahat OMITS the (1−R) factor and is NOT used here.

All Riccati / limit computations delegate to existing core / closed_form functions.

Outputs
-------
  calibration/results/tables/
    yield_curve_condition_by_date.csv
    yield_curve_condition_summary.csv
    long_maturity_convergence.csv

  calibration/results/figures/
    yield_curve_threshold_ratios.png
    upward_slope_frequency_by_country.png
    long_maturity_discriminants.png
    long_maturity_yield_limits.png
    condition_gap_vs_actual_slope.png   (only if yield pricers available)

CLI
---
  python -m calibration.yield_curve_diagnostics
  python -m calibration.yield_curve_diagnostics --no-finite-slopes
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.parameters  import DisasterModelParams
from core.closed_form import (
    K_const, _phi_sig2, _defaultable_Ai,
    y_star_inf, y_D_star_inf, b_star_inf, b_Df_inf, b_Dg_inf,
)
from core.riccati import discriminant

PANEL_PATH  = os.path.join(BASE_DIR, "processed_data", "panel.csv")
PARAMS_PATH = os.path.join(BASE_DIR, "results", "parameters.csv")
TABLES_DIR  = os.path.join(BASE_DIR, "results", "tables")
FIGS_DIR    = os.path.join(BASE_DIR, "results", "figures")

# ── Optional finite-maturity slope pricers ────────────────────────────────────
try:
    from models.risk_free   import rf_yield
    from models.defaultable import def_yield
    _HAVE_YIELD_PRICERS = True
except ImportError:
    _HAVE_YIELD_PRICERS = False
    warnings.warn(
        "models.risk_free / models.defaultable not importable; "
        "finite-maturity slope validation will be skipped.",
        ImportWarning, stacklevel=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parameter construction
# ─────────────────────────────────────────────────────────────────────────────

def _make_params(
    base_params: DisasterModelParams,
    row_p:       pd.Series,
    lam_bar_f:   float,
    lam_bar_g:   float,
) -> DisasterModelParams:
    """Build per-country DisasterModelParams from a parameters.csv row."""
    return dataclasses.replace(
        base_params,
        h0_star   = float(row_p["h0"]),
        eta1      = float(row_p["eta_i"]),
        eta2      = float(row_p["eta_g"]),
        R         = float(row_p["R"]) if "R" in row_p.index and pd.notna(row_p["R"]) else base_params.R,
        lam_bar_f = float(lam_bar_f),
        lam_bar_g = float(lam_bar_g),
        b_sdf     = base_params.b_sdf,   # cached; does not depend on η/h0/R
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-date (state-dependent) conditions  —  vectorised over dates
# ─────────────────────────────────────────────────────────────────────────────

def _per_date_conditions(
    lam_f_arr: np.ndarray,
    lam_g_arr: np.ndarray,
    params:    DisasterModelParams,
) -> dict:
    """
    Vectorised computation of Prop. risk-free / defaultable short-end conditions.

    Risk-free (Prop. risk-free slope):
      slope  = -(K/2) [κ(λ̄_f+λ̄_g) + φ(λ_f+λ_g)]
      upward ⟺  λ_f+λ_g  >  κ(λ̄_f+λ̄_g)/(κ−bσ²)   [threshold_rf]

    Defaultable (Prop. defaultable slope):
      slope  = (1/2) [κ(λ̄_f A_f + λ̄_g A_g) + φ(λ_f A_f + λ_g A_g)]
      upward ⟺  λ_f A_f + λ_g A_g  >  κ(λ̄_f A_f + λ̄_g A_g)/(κ−bσ²)   [threshold_d]

    A_f = −K + (1−R)η_f,  A_g = −K + (1−R)η_g  (from _defaultable_Ai, WITH (1-R))
    """
    phi, sig2  = _phi_sig2(params)
    kappa      = params.kappa
    K          = K_const(params)
    _, Af, Ag  = _defaultable_Ai(params)     # includes (1-R) factor

    lam_bar_f  = params.lam_bar_f
    lam_bar_g  = params.lam_bar_g

    # κ − bσ² = −φ  (> 0 when stable)
    kap_mb = kappa - float(params.b_sdf) * sig2   # = -phi

    # ── Risk-free ─────────────────────────────────────────────────────────────
    lam_total    = lam_f_arr + lam_g_arr
    lam_bar_tot  = lam_bar_f + lam_bar_g
    threshold_rf = kappa * lam_bar_tot / kap_mb if kap_mb != 0 else np.inf
    gap_rf       = lam_total - threshold_rf
    slope_rf     = -(K / 2.0) * (kappa * lam_bar_tot + phi * lam_total)
    upward_rf    = gap_rf > 0

    # ── Defaultable ───────────────────────────────────────────────────────────
    weighted_bar_d = lam_bar_f * Af + lam_bar_g * Ag
    condition_d    = lam_f_arr * Af + lam_g_arr * Ag
    threshold_d    = kappa * weighted_bar_d / kap_mb if kap_mb != 0 else np.inf
    gap_d          = condition_d - threshold_d
    slope_d        = 0.5 * (kappa * weighted_bar_d + phi * condition_d)
    upward_d       = gap_d < 0

    return dict(
        phi=phi, sig2=sig2, K=K, A_f=Af, A_g=Ag,
        kap_mb=kap_mb,
        # risk-free
        threshold_rf=threshold_rf,
        condition_rf=lam_total,
        gap_rf=gap_rf, slope_rf=slope_rf, upward_rf=upward_rf,
        # defaultable
        weighted_bar_d=weighted_bar_d,
        threshold_d=threshold_d,
        condition_d=condition_d,
        gap_d=gap_d, slope_d=slope_d, upward_d=upward_d,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Long-maturity convergence  —  parameter-dependent, one value per country
# ─────────────────────────────────────────────────────────────────────────────

def _long_run(params: DisasterModelParams) -> dict:
    """
    Prop. long-maturity: conditions and limits for τ → ∞.

    Conditions:  φ < 0,  δ(K) > 0,  δ(−A_f) > 0,  δ(−A_g) > 0.

    Limiting yields are computed by reusing closed_form.y_star_inf
    and closed_form.y_D_star_inf (which already implement the proposition
    formula exactly).
    """
    phi, sig2  = _phi_sig2(params)
    K          = K_const(params)
    _, Af, Ag  = _defaultable_Ai(params)

    disc_K      = discriminant(K,   phi, sig2)   # δ(K)
    disc_neg_Af = discriminant(-Af, phi, sig2)   # δ(−A_f)
    disc_neg_Ag = discriminant(-Ag, phi, sig2)   # δ(−A_g)

    phi_stable  = phi < 0.0
    conv_rf     = phi_stable and (disc_K > 0)
    conv_d      = phi_stable and (disc_neg_Af > 0) and (disc_neg_Ag > 0)

    # Limiting loadings (lower fixed points)
    bsi  = b_star_inf(params) if conv_rf else None     # b*_∞
    bdfi = b_Df_inf(params)   if conv_d  else None     # b_{D,f,∞}
    bdgi = b_Dg_inf(params)   if conv_d  else None     # b_{D,g,∞}

    # Limiting yields — reuse existing closed_form functions
    yrf_inf  = y_star_inf(params)   if conv_rf else None   # already uses bsi
    yd_inf   = y_D_star_inf(params) if conv_d  else None   # already uses bdfi, bdgi

    spread_inf = (
        (yd_inf - yrf_inf)
        if (yrf_inf is not None and yd_inf is not None)
        else None
    )

    def _nan(v):
        return float(v) if v is not None else np.nan

    return dict(
        phi=phi, sig2=sig2, K=K, A_f=Af, A_g=Ag,
        A0=_defaultable_Ai(params)[0],
        phi_stable=phi_stable,
        disc_K=disc_K, disc_neg_Af=disc_neg_Af, disc_neg_Ag=disc_neg_Ag,
        conv_rf=conv_rf, conv_d=conv_d,
        b_star_inf=_nan(bsi),
        b_Df_inf=_nan(bdfi), b_Dg_inf=_nan(bdgi),
        y_rf_inf=_nan(yrf_inf),
        y_d_inf=_nan(yd_inf),
        spread_inf=_nan(spread_inf),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional: finite-difference slopes  y(τ_hi) − y(τ_lo)
# ─────────────────────────────────────────────────────────────────────────────

def _finite_slopes_single(
    params: DisasterModelParams,
    lam_f:  float,
    lam_g:  float,
    tau_lo: float = 1.0,
    tau_hi: float = 2.0,
) -> tuple[float, float]:
    """
    (2Y−1Y) finite-difference slopes for rf and defaultable yields.
    Returns (nan, nan) if pricers unavailable or evaluation fails.
    """
    if not _HAVE_YIELD_PRICERS:
        return np.nan, np.nan
    try:
        s_rf = (rf_yield(params, tau_hi, lam_f, lam_g)
                - rf_yield(params, tau_lo, lam_f, lam_g)) / (tau_hi - tau_lo)
        s_d  = (def_yield(params, tau_hi, lam_f, lam_g)
                - def_yield(params, tau_lo, lam_f, lam_g)) / (tau_hi - tau_lo)
        return float(s_rf), float(s_d)
    except Exception:
        return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(
    panel:                pd.DataFrame,
    params_df:            pd.DataFrame,
    base_params:          DisasterModelParams,
    include_finite_slopes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all diagnostics for every (country, date) in the panel.

    Returns
    -------
    by_date_df     : one row per (country, date)
    summary_df     : one row per country
    convergence_df : one row per country (long-run analysis)
    """
    lam_bar_g_global = float(panel["lambda_global"].mean())
    do_finite = include_finite_slopes and _HAVE_YIELD_PRICERS
    params_idx = params_df.set_index("country")

    by_date_rows   : list[dict] = []
    convergence_rows: list[dict] = []

    for country, grp in panel.groupby("country"):
        if country not in params_idx.index:
            warnings.warn(f"{country}: not in parameters.csv, skipping.")
            continue
        row_p = params_idx.loc[country]
        if "converged" in row_p.index and not bool(row_p["converged"]):
            warnings.warn(f"{country}: calibration did not converge, skipping.")
            continue

        lam_bar_f = float(grp["lambda_country"].mean())
        params    = _make_params(base_params, row_p, lam_bar_f, lam_bar_g_global)

        grp_clean = grp.dropna(subset=["lambda_country", "lambda_global"]).copy()
        if grp_clean.empty:
            continue

        lam_f = grp_clean["lambda_country"].to_numpy()
        lam_g = grp_clean["lambda_global"].to_numpy()
        dates = grp_clean["date"].to_numpy()

        # ── Per-date vectorised conditions ─────────────────────────────────
        res = _per_date_conditions(lam_f, lam_g, params)

        for i in range(len(grp_clean)):
            row: dict = {
                "country":          country,
                "date":             dates[i],
                "lambda_country":   lam_f[i],
                "lambda_global":    lam_g[i],
                # risk-free
                "threshold_rf":     res["threshold_rf"],
                "condition_rf":     res["condition_rf"][i],
                "gap_rf":           res["gap_rf"][i],
                "slope_rf_analytic": res["slope_rf"][i],
                "upward_rf":        bool(res["upward_rf"][i]),
                # defaultable (includes A constants for context)
                "A0":               res["A_f"],   # kept for traceability
                "A_f":              res["A_f"],
                "A_g":              res["A_g"],
                "threshold_d":      res["threshold_d"],
                "condition_d":      res["condition_d"][i],
                "gap_d":            res["gap_d"][i],
                "slope_d_analytic": res["slope_d"][i],
                "upward_d":         bool(res["upward_d"][i]),
            }

            if do_finite:
                s_rf, s_d = _finite_slopes_single(params, lam_f[i], lam_g[i])
                row["slope_rf_2y1y"]  = s_rf
                row["slope_d_2y1y"]   = s_d
                if np.isfinite(s_rf):
                    row["sign_match_rf"] = bool((s_rf > 0) == bool(res["upward_rf"][i]))
                if np.isfinite(s_d):
                    row["sign_match_d"]  = bool((s_d  > 0) == bool(res["upward_d"][i]))

            by_date_rows.append(row)

        # ── Long-run (per country) ─────────────────────────────────────────
        lr = _long_run(params)
        convergence_rows.append({
            "country":        country,
            "b_sdf":          float(params.b_sdf),
            "phi":            lr["phi"],
            "kappa_minus_bsig2": -lr["phi"],   # = κ - bσ²; positive when stable
            "K":              lr["K"],
            "A_f":            lr["A_f"],
            "A_g":            lr["A_g"],
            "A0":             lr["A0"],
            "eta_i":          float(params.eta1),
            "eta_g":          float(params.eta2),
            "R":              float(params.R),
            "phi_stable":     lr["phi_stable"],
            "disc_K":         lr["disc_K"],
            "disc_neg_Af":    lr["disc_neg_Af"],
            "disc_neg_Ag":    lr["disc_neg_Ag"],
            "converges_rf":   lr["conv_rf"],
            "converges_d":    lr["conv_d"],
            "b_star_inf":     lr["b_star_inf"],
            "b_Df_inf":       lr["b_Df_inf"],
            "b_Dg_inf":       lr["b_Dg_inf"],
            "y_rf_inf_pct":   lr["y_rf_inf"] * 100 if np.isfinite(lr["y_rf_inf"]) else np.nan,
            "y_d_inf_pct":    lr["y_d_inf"]  * 100 if np.isfinite(lr["y_d_inf"])  else np.nan,
            "spread_inf_bps": lr["spread_inf"] * 1e4 if np.isfinite(lr["spread_inf"]) else np.nan,
        })

    by_date_df    = pd.DataFrame(by_date_rows)
    convergence_df = pd.DataFrame(convergence_rows)

    # ── Summary by country ────────────────────────────────────────────────
    summary_rows: list[dict] = []
    for country, grp in by_date_df.groupby("country"):
        n   = len(grp)
        row = {
            "country":       country,
            "n_obs":         n,
            "pct_upward_rf": grp["upward_rf"].mean() * 100,
            "pct_upward_d":  grp["upward_d"].mean()  * 100,
        }
        for col, prefix in [("gap_rf", "gap_rf"), ("gap_d", "gap_d")]:
            row[f"{prefix}_mean"]   = grp[col].mean()
            row[f"{prefix}_median"] = grp[col].median()
            row[f"{prefix}_p10"]    = grp[col].quantile(0.10)
            row[f"{prefix}_p90"]    = grp[col].quantile(0.90)
        row["slope_rf_mean"] = grp["slope_rf_analytic"].mean()
        row["slope_d_mean"]  = grp["slope_d_analytic"].mean()
        if "slope_rf_2y1y" in grp.columns:
            row["slope_rf_2y1y_mean"] = grp["slope_rf_2y1y"].mean()
            row["slope_d_2y1y_mean"]  = grp["slope_d_2y1y"].mean()
            row["sign_match_rf_pct"]  = grp.get("sign_match_rf", pd.Series(dtype=float)).mean() * 100
            row["sign_match_d_pct"]   = grp.get("sign_match_d",  pd.Series(dtype=float)).mean() * 100
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return by_date_df, summary_df, convergence_df


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_threshold_ratios(by_date_df: pd.DataFrame, out_dir: str) -> None:
    """Histogram of condition / threshold ratio for RF and D conditions (vline @ 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = [
        (axes[0], "condition_rf", "threshold_rf", "Risk-free"),
        (axes[1], "condition_d",  "threshold_d",  "Defaultable"),
    ]
    for ax, cond_col, thr_col, label in pairs:
        thr  = by_date_df[thr_col].replace(0, np.nan)
        ratio = (by_date_df[cond_col] / thr).dropna()
        p99   = np.percentile(ratio, 99) if len(ratio) > 0 else 5
        ratio = ratio.clip(upper=p99)
        ax.hist(ratio, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(1.0, color="crimson", lw=1.5, linestyle="--", label="Ratio = 1")
        pct = (ratio > 1).mean() * 100
        ax.set_title(f"{label}\n({pct:.1f}% of obs above threshold)")
        ax.set_xlabel("Condition value / Threshold")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Threshold Ratios — Short-End Upward-Slope Conditions", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "yield_curve_threshold_ratios.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def _plot_upward_frequency(summary_df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: % of dates satisfying upward-slope conditions, by country."""
    df = summary_df.sort_values("pct_upward_rf", ascending=False).reset_index(drop=True)
    x  = np.arange(len(df))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9 + 1), 4))
    ax.bar(x - w / 2, df["pct_upward_rf"], w, label="Risk-free",   color="steelblue",  alpha=0.85)
    ax.bar(x + w / 2, df["pct_upward_d"],  w, label="Defaultable", color="darkorange", alpha=0.85)
    ax.axhline(50, color="grey", lw=0.8, linestyle=":", label="50%")
    ax.set_xticks(x)
    ax.set_xticklabels(df["country"], rotation=45, ha="right")
    ax.set_ylabel("% of dates with upward short-end slope")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9)
    ax.set_title("Frequency of Upward Short-End Slope by Country")
    plt.tight_layout()
    path = os.path.join(out_dir, "upward_slope_frequency_by_country.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def _plot_discriminants(convergence_df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: three discriminants per country with zero line."""
    df = convergence_df.sort_values("country").reset_index(drop=True)
    x  = np.arange(len(df))
    w  = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9 + 1), 4))
    ax.bar(x - w,   df["disc_K"],       w, label=r"$\delta(K)$",      color="steelblue",   alpha=0.85)
    ax.bar(x,       df["disc_neg_Af"],  w, label=r"$\delta(-A_f)$",   color="darkorange",  alpha=0.85)
    ax.bar(x + w,   df["disc_neg_Ag"],  w, label=r"$\delta(-A_g)$",   color="forestgreen", alpha=0.85)
    ax.axhline(0, color="black", lw=1.0, linestyle="--", label="zero")
    ax.set_xticks(x)
    ax.set_xticklabels(df["country"], rotation=45, ha="right")
    ax.set_ylabel("Discriminant value")
    ax.set_title("Long-Maturity Convergence Discriminants\n(all must be > 0 for convergence)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "long_maturity_discriminants.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def _plot_yield_limits(convergence_df: pd.DataFrame, out_dir: str) -> None:
    """Long-run RF yield, defaultable yield, and spread by country."""
    df = convergence_df.dropna(subset=["y_rf_inf_pct"]).copy()
    if df.empty:
        warnings.warn("No countries with convergent RF yield; skipping long_maturity_yield_limits.png")
        return
    df = df.sort_values("y_rf_inf_pct").reset_index(drop=True)
    x  = np.arange(len(df))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9 + 1), 4))
    ax.bar(x - w / 2, df["y_rf_inf_pct"], w,
           label=r"$y^*_\infty$ (%)", color="steelblue", alpha=0.85)

    has_d = df["y_d_inf_pct"].notna().any()
    if has_d:
        ax.bar(x + w / 2, df["y_d_inf_pct"].fillna(0), w,
               label=r"$y^*_{D,\infty}$ (%)", color="darkorange", alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(df["country"], rotation=45, ha="right")
    ax.set_ylabel("Limiting yield (%)")
    ax.set_title("Long-Maturity Limiting Yields")
    ax.legend(fontsize=9)

    if has_d and df["spread_inf_bps"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(x, df["spread_inf_bps"].to_numpy(), "kD", ms=5,
                 label="Long-run spread (bps)")
        ax2.set_ylabel("Limiting spread (bps)")
        ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "long_maturity_yield_limits.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def _plot_gap_vs_actual_slope(by_date_df: pd.DataFrame, out_dir: str) -> None:
    """Scatter: analytical condition gap vs finite-difference (2Y−1Y) slope."""
    if "slope_rf_2y1y" not in by_date_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pairs = [
        (axes[0], "gap_rf", "slope_rf_2y1y", "Risk-free"),
        (axes[1], "gap_d",  "slope_d_2y1y",  "Defaultable"),
    ]
    for ax, gap_col, slope_col, label in pairs:
        if slope_col not in by_date_df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        sub = by_date_df[[gap_col, slope_col]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        sign_match = (np.sign(sub[gap_col]) == np.sign(sub[slope_col])).mean() * 100
        ax.scatter(sub[gap_col], sub[slope_col], s=4, alpha=0.35, color="steelblue",
                   rasterized=True)
        ax.axhline(0, color="grey", lw=0.8, linestyle=":")
        ax.axvline(0, color="grey", lw=0.8, linestyle=":")
        ax.set_xlabel("Condition gap (state − threshold)")
        ax.set_ylabel("FD slope y(2Y)−y(1Y)")
        ax.set_title(f"{label} — sign match: {sign_match:.1f}%")

    fig.suptitle("Condition Gap vs. Finite-Difference Short-End Slope", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "condition_gap_vs_actual_slope.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_tables(
    by_date_df:     pd.DataFrame,
    summary_df:     pd.DataFrame,
    convergence_df: pd.DataFrame,
    tables_dir:     str = TABLES_DIR,
) -> None:
    os.makedirs(tables_dir, exist_ok=True)
    for df, name in [
        (by_date_df,     "yield_curve_condition_by_date.csv"),
        (summary_df,     "yield_curve_condition_summary.csv"),
        (convergence_df, "long_maturity_convergence.csv"),
    ]:
        path = os.path.join(tables_dir, name)
        df.to_csv(path, index=False)
        print(f"  → {path}")


def save_figures(
    by_date_df:     pd.DataFrame,
    summary_df:     pd.DataFrame,
    convergence_df: pd.DataFrame,
    figs_dir:       str = FIGS_DIR,
) -> None:
    os.makedirs(figs_dir, exist_ok=True)
    _plot_threshold_ratios(by_date_df,  figs_dir)
    _plot_upward_frequency(summary_df,  figs_dir)
    _plot_discriminants(convergence_df, figs_dir)
    _plot_yield_limits(convergence_df,  figs_dir)
    _plot_gap_vs_actual_slope(by_date_df, figs_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    by_date_df:     pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  YIELD CURVE DIAGNOSTICS  —  SUMMARY")
    print(sep)

    n_total = len(by_date_df)
    pct_rf = by_date_df["upward_rf"].mean() * 100
    pct_d  = by_date_df["upward_d"].mean()  * 100
    print(f"  Country-date observations:          {n_total:,}")
    print(f"  RF upward short-end condition:      {pct_rf:.1f}% of obs")
    print(f"  Defaultable upward short-end:       {pct_d:.1f}% of obs")

    print(f"\n  Stability check  φ = bσ²−κ < 0:")
    for _, row in convergence_df.iterrows():
        sym = "✓" if row["phi_stable"] else "✗  ← VIOLATION"
        print(f"    {row['country']:4s}  φ = {row['phi']:+.5f}  {sym}")

    n_conv_rf = int(convergence_df["converges_rf"].sum())
    n_total_c = len(convergence_df)
    print(f"\n  Long-run RF convergence:  {n_conv_rf}/{n_total_c} countries")

    d_yes = convergence_df.loc[convergence_df["converges_d"], "country"].tolist()
    d_no  = convergence_df.loc[~convergence_df["converges_d"], "country"].tolist()
    print(f"  Long-run D  convergence:  {len(d_yes)}/{n_total_c} countries")
    if d_yes:
        print(f"    Converging:     {d_yes}")
    if d_no:
        print(f"    Non-converging: {d_no}")

    # Note about interpretation
    print()
    print("  Note: the upward short-end condition is state-dependent and can")
    print("  be satisfied even when the curve is downward-sloping elsewhere.")
    print("  Long-maturity convergence is parameter-dependent; a curve can be")
    print("  downward-sloping in simulation while still converging to a finite")
    print("  limit as τ → ∞.")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_yield_diagnostics(
    panel:       pd.DataFrame,
    params_df:   pd.DataFrame,
    base_params: DisasterModelParams,
    tables_dir:  str  = TABLES_DIR,
    figs_dir:    str  = FIGS_DIR,
    include_finite_slopes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Public entry point for integration with results_pipeline.py.

    Returns (by_date_df, summary_df, convergence_df).
    """
    by_date_df, summary_df, convergence_df = compute_diagnostics(
        panel, params_df, base_params,
        include_finite_slopes=include_finite_slopes,
    )
    print(f"  {len(by_date_df):,} country-date observations processed.")
    print("\nSaving yield-curve diagnostic tables …")
    save_tables(by_date_df, summary_df, convergence_df, tables_dir)
    print("Saving yield-curve diagnostic figures …")
    save_figures(by_date_df, summary_df, convergence_df, figs_dir)
    print_summary(by_date_df, convergence_df)
    return by_date_df, summary_df, convergence_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yield-curve slope and long-maturity convergence diagnostics."
    )
    parser.add_argument("--panel",   default=PANEL_PATH,  help="Path to panel.csv")
    parser.add_argument("--params",  default=PARAMS_PATH, help="Path to parameters.csv")
    parser.add_argument("--tables",  default=TABLES_DIR,  help="Output dir for tables")
    parser.add_argument("--figures", default=FIGS_DIR,    help="Output dir for figures")
    parser.add_argument(
        "--no-finite-slopes", action="store_true",
        help="Skip finite-maturity (2Y−1Y) slope validation (faster).",
    )
    args = parser.parse_args()

    for p in [args.panel, args.params]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "Run  python -m calibration.run_calibration  first."
            )

    print("Loading inputs …")
    panel     = pd.read_csv(args.panel,  parse_dates=["date"])
    params_df = pd.read_csv(args.params)
    print(f"  Panel:      {len(panel):,} rows, {panel['country'].nunique()} countries")
    print(f"  Parameters: {len(params_df)} countries")

    base_params = DisasterModelParams()
    base_params.compute_b_sdf()
    print(f"  b_sdf = {base_params.b_sdf:.6f}")

    include_finite = not args.no_finite_slopes
    if include_finite and not _HAVE_YIELD_PRICERS:
        print("  Note: finite-slope validation skipped (yield pricers not importable).")
        include_finite = False

    run_yield_diagnostics(
        panel, params_df, base_params,
        tables_dir=args.tables,
        figs_dir=args.figures,
        include_finite_slopes=include_finite,
    )


if __name__ == "__main__":
    main()
