"""
simulation_tables.py
====================
Simulation moment tables and variance decompositions.

Table inventory
---------------
1. moment_comparison              — data vs simulation moments pooled + per country
2. simulated_spread_moments_by_tenor — per-tenor moments with data/simulation columns
3. variance_decomposition         — Option-B counterfactual variance decomposition
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from calibration.simulation import SimulationResult, spread_series, counterfactual_spreads
from core.parameters import DisasterModelParams

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

TAU_5Y = 5.0   # reference tenor for pooled moments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, name: str, out_dir: str = TABLES_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"  → {os.path.relpath(path, PARENT_DIR)}")


def _moments(x: np.ndarray, lag: int = 1) -> dict:
    """Compute mean, std, autocorrelation at *lag* for a 1-D array."""
    x   = x[np.isfinite(x)]
    if len(x) < 4:
        return {"mean": np.nan, "std": np.nan, "autocorr": np.nan}
    ac  = float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if len(x) > lag + 1 else np.nan
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "autocorr": ac}


def _corr_with(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


# ---------------------------------------------------------------------------
# 1. Moment comparison (data vs simulation)
# ---------------------------------------------------------------------------

def build_moment_comparison(
    sim_results: dict[str, SimulationResult],
    panel:       pd.DataFrame,
    tau:         float = TAU_5Y,
) -> pd.DataFrame:
    """Compare key moments between observed panel and simulated model.

    Pooled row first, then per country.
    """
    records = []

    def _data_moments(sdf: pd.DataFrame) -> dict:
        col_5y = "cds_spread_5y"
        out    = {}
        if col_5y in sdf.columns and sdf[col_5y].notna().any():
            s5  = sdf[col_5y].dropna().to_numpy() * 1e4   # bps
            m   = _moments(s5)
            out.update({f"data_mean_5y_bps":   round(m["mean"], 2),
                        f"data_std_5y_bps":    round(m["std"],  2),
                        f"data_autocorr_5y":   round(m["autocorr"], 4)})
        if "lambda_global" in sdf.columns:
            lg   = sdf["lambda_global"].dropna().to_numpy()
            h_s5 = sdf[col_5y].dropna() if col_5y in sdf.columns else pd.Series(dtype=float)
            if len(lg) > 2 and col_5y in sdf.columns:
                # align lengths
                common = sdf.dropna(subset=[col_5y, "lambda_global"])
                out["data_corr_s5y_lambda_g"] = round(
                    _corr_with(common[col_5y].to_numpy(), common["lambda_global"].to_numpy()), 4)
        if "lambda_country" in sdf.columns and col_5y in sdf.columns:
            common = sdf.dropna(subset=[col_5y, "lambda_country"])
            out["data_corr_s5y_lambda_i"] = round(
                _corr_with(common[col_5y].to_numpy(), common["lambda_country"].to_numpy()), 4)
        return out

    def _sim_moments(sim: SimulationResult) -> dict:
        s5  = spread_series(sim, tau) * 1e4   # bps
        m   = _moments(s5)
        out = {
            "sim_mean_5y_bps":      round(m["mean"], 2),
            "sim_std_5y_bps":       round(m["std"],  2),
            "sim_autocorr_5y":      round(m["autocorr"], 4),
            "sim_corr_s5y_lambda_g": round(_corr_with(s5, sim.lam_g * 1e4), 4),
            "sim_corr_s5y_lambda_i": round(_corr_with(s5, sim.lam_i * 1e4), 4),
            "sim_corr_s5y_hstar":    round(_corr_with(s5, sim.h_star * 1e4), 4),
        }
        return out

    # Pooled data moments
    pooled_data = _data_moments(panel)
    # Pooled simulation: average across countries
    sim_pool = {}
    for key in ["sim_mean_5y_bps", "sim_std_5y_bps", "sim_autocorr_5y",
                "sim_corr_s5y_lambda_g", "sim_corr_s5y_lambda_i", "sim_corr_s5y_hstar"]:
        vals = [_sim_moments(s).get(key, np.nan) for s in sim_results.values()]
        sim_pool[key] = round(float(np.nanmean(vals)), 4) if vals else np.nan

    records.append({"country": "POOLED", **pooled_data, **sim_pool})

    # Per-country rows
    for country, sim in sim_results.items():
        grp      = panel[panel["country"] == country]
        row_data = _data_moments(grp)
        row_sim  = _sim_moments(sim)
        records.append({"country": country, **row_data, **row_sim})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Simulated spread moments by tenor
# ---------------------------------------------------------------------------

def build_simulated_spread_moments_by_tenor(
    sim_results: dict[str, SimulationResult],
    panel:       pd.DataFrame,
) -> pd.DataFrame:
    """Per-tenor moments averaged across all simulated countries, with data comparison."""
    TENOR_MAP = {1.0: "cds_spread_1y", 2.0: "cds_spread_2y", 3.0: "cds_spread_3y",
                 5.0: "cds_spread_5y", 7.0: "cds_spread_7y", 10.0: "cds_spread_10y"}

    if not sim_results:
        return pd.DataFrame()

    # Determine available tenors from first sim result
    sample_sim  = next(iter(sim_results.values()))
    sim_tenors  = sample_sim.tenors

    records = []
    for tau in sim_tenors:
        col = TENOR_MAP.get(tau)

        # Data moments (pool all countries that have this tenor)
        data_vals = []
        if col and col in panel.columns:
            data_vals = panel[col].dropna().to_numpy() * 1e4
        dm = _moments(data_vals) if len(data_vals) > 4 else {}

        # Simulation moments (average across countries)
        sim_means, sim_stds, sim_acs = [], [], []
        sim_corr_g, sim_corr_i = [], []
        for sim in sim_results.values():
            try:
                j   = sim.tenors.index(tau)
            except ValueError:
                continue
            s    = sim.spreads[:, j] * 1e4
            m    = _moments(s)
            sim_means.append(m["mean"])
            sim_stds.append(m["std"])
            sim_acs.append(m["autocorr"])
            sim_corr_g.append(_corr_with(s, sim.lam_g * 1e4))
            sim_corr_i.append(_corr_with(s, sim.lam_i * 1e4))

        def _avg(lst):
            arr = [v for v in lst if v is not None and np.isfinite(v)]
            return round(float(np.mean(arr)), 4) if arr else np.nan

        records.append({
            "tenor_y":             float(tau),
            "data_mean_bps":       round(float(np.mean(data_vals)), 2) if len(data_vals) > 0 else np.nan,
            "data_std_bps":        round(float(np.std(data_vals, ddof=1)), 2) if len(data_vals) > 1 else np.nan,
            "data_autocorr":       round(dm.get("autocorr", np.nan), 4),
            "sim_mean_bps":        _avg(sim_means),
            "sim_std_bps":         _avg(sim_stds),
            "sim_autocorr":        _avg(sim_acs),
            "sim_corr_lambda_g":   _avg(sim_corr_g),
            "sim_corr_lambda_i":   _avg(sim_corr_i),
        })

    return pd.DataFrame(records).sort_values("tenor_y").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Variance decomposition (Option B — counterfactual)
# ---------------------------------------------------------------------------

def build_variance_decomposition(
    sim_results: dict[str, SimulationResult],
    base_params: DisasterModelParams,
    tau:         float = TAU_5Y,
    n_int:       int   = 100,
) -> pd.DataFrame:
    """Option-B counterfactual variance decomposition at 5Y tenor.

    For each country:
        full_spread_t     = f(λ^i_t, λ^g_t)
        global_only_t     = f(λ̄^i,   λ^g_t)   ← only global varies
        country_only_t    = f(λ^i_t, λ̄^g)    ← only country varies

    Shares:
        global_share  = Var(global_only) / (Var(global_only) + Var(country_only))
        country_share = 1 − global_share
    """
    records = []
    for country, sim in sim_results.items():
        s_full = spread_series(sim, tau) * 1e4   # bps

        try:
            s_global_only, s_country_only = counterfactual_spreads(
                sim, base_params, tau=tau, n_int=n_int
            )
            s_global_only  = s_global_only  * 1e4
            s_country_only = s_country_only * 1e4
        except Exception as exc:
            print(f"  WARNING: variance decomp failed for {country}: {exc}")
            continue

        var_full    = float(np.var(s_full))
        var_global  = float(np.var(s_global_only))
        var_country = float(np.var(s_country_only))
        denom       = var_global + var_country

        records.append({
            "country":          country,
            "var_full_bps2":    round(var_full, 2),
            "var_global_bps2":  round(var_global, 2),
            "var_country_bps2": round(var_country, 2),
            "global_share":     round(var_global  / denom, 4) if denom > 0 else np.nan,
            "country_share":    round(var_country / denom, 4) if denom > 0 else np.nan,
            "mean_full_bps":    round(float(np.mean(s_full)), 2),
            "mean_global_only_bps":  round(float(np.mean(s_global_only)), 2),
            "mean_country_only_bps": round(float(np.mean(s_country_only)), 2),
        })

    return pd.DataFrame(records).sort_values("country").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def save_all_simulation_tables(
    sim_results: dict[str, SimulationResult],
    panel:       pd.DataFrame,
    base_params: DisasterModelParams,
    out_dir:     str = TABLES_DIR,
) -> None:
    """Build and save all simulation tables to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    print("Building simulation tables …")

    if not sim_results:
        print("  No simulation results — skipping simulation tables.")
        return

    t1 = build_moment_comparison(sim_results, panel)
    _save(t1, "moment_comparison.csv", out_dir)

    t2 = build_simulated_spread_moments_by_tenor(sim_results, panel)
    _save(t2, "simulated_spread_moments_by_tenor.csv", out_dir)

    t3 = build_variance_decomposition(sim_results, base_params)
    if not t3.empty:
        _save(t3, "variance_decomposition.csv", out_dir)

    print(f"Simulation tables saved to {os.path.relpath(out_dir, PARENT_DIR)}/")
