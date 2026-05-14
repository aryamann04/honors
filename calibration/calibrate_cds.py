"""
calibrate_cds.py
================
Per-country NLS calibration of (h0_i, η_i, η_{g,i}).

For each country i, minimise
    Σ_{t,τ} (s^model_{i,t}(τ) − s^obs_{i,t}(τ))²
over (h0_i, η_i, η_{g,i}), where
    h^i_t = h0_i + η_i λ^i_t + η_{g,i} λ^g_t.

Speed improvements over the naïve implementation
-------------------------------------------------
1. **Vectorised batch pricer** (`models.cds.cds_spread_batch`): affine coefficients
   are computed once per optimizer call (not once per date×tenor); G and K are
   then broadcast over all lambda pairs via numpy, giving ~100× speedup in the
   inner loop.

2. **Parallel country calibration**: countries are independent so they are
   dispatched to a `ProcessPoolExecutor` (default: all CPUs).

3. **Looser tolerances and fewer iterations**: sufficient for practical accuracy
   given the data quality.

Inputs
------
  calibration/processed_data/panel.csv

Outputs
-------
  calibration/results/parameters.csv
  calibration/results/model_vs_observed.csv

CLI
---
    python -m calibration.calibrate_cds
"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.parameters import DisasterModelParams
from models.cds import cds_spread_batch, fair_cds_spread

PANEL_PATH    = os.path.join(BASE_DIR, "processed_data", "panel.csv")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
PARAMS_OUT    = os.path.join(RESULTS_DIR, "parameters.csv")
MODEL_OBS_OUT = os.path.join(RESULTS_DIR, "model_vs_observed.csv")

# Optimisation settings
X0_DEFAULT  = np.array([0.02,  2.0,  1.0])     # [h0, eta_i, eta_g]
LOWER_BOUND = np.array([0.0,  0.0,  0.0])
UPPER_BOUND = np.array([0.50, 100.0, 100.0])

N_INT_FAST  = 100   # integration points during optimisation (vectorised; fast+accurate)
N_INT_FINAL = 400   # integration points for final model_vs_observed table

MIN_OBS     = 24    # minimum valid (date × tenor) residuals per country


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"country", "date", "lambda_global", "lambda_country"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"panel.csv missing columns: {missing}")
    return df


def get_available_tenors(panel: pd.DataFrame) -> list[float]:
    """Return tenor floats for columns with at least MIN_OBS non-null rows globally."""
    tenor_map = {
        "cds_spread_1y": 1.0,  "cds_spread_2y": 2.0,
        "cds_spread_3y": 3.0,  "cds_spread_5y": 5.0,
        "cds_spread_7y": 7.0,  "cds_spread_10y": 10.0,
    }
    return sorted(
        tau for col, tau in tenor_map.items()
        if col in panel.columns and panel[col].notna().sum() >= MIN_OBS
    )


def make_params_for_country(
    base_params: DisasterModelParams,
    h0:          float,
    eta_i:       float,
    eta_g:       float,
    lam_bar_f:   float,
    lam_bar_g:   float,
    R:           float,
) -> DisasterModelParams:
    """Return modified params with country-specific hazard parameters."""
    return dataclasses.replace(
        base_params,
        h0_star   = float(h0),
        eta1      = float(eta_i),
        eta2      = float(eta_g),
        lam_bar_f = float(lam_bar_f),
        lam_bar_g = float(lam_bar_g),
        R         = float(R),
        b_sdf     = base_params.b_sdf,   # reuse cached value — avoids recomputation
    )


# ---------------------------------------------------------------------------
# Observation matrix builder
# ---------------------------------------------------------------------------

def _spread_col(tau: float) -> str:
    return f"cds_spread_{int(tau)}y" if tau != 10.0 else "cds_spread_10y"


def _build_obs_matrix(country_df: pd.DataFrame, tenors: list[float]) -> np.ndarray:
    """Build (N_rows, 2 + len(tenors)) matrix.

    Columns: [lambda_country, lambda_global, spread_tau1, spread_tau2, ...]
    Rows where both lambdas are finite and at least one tenor spread is finite.
    Spreads stored in decimal (not bps).
    """
    spread_cols = [_spread_col(t) for t in tenors]
    sub = country_df[
        ["lambda_country", "lambda_global"]
        + [c for c in spread_cols if c in country_df.columns]
    ].copy()
    sub = sub.dropna(subset=["lambda_country", "lambda_global"])

    present_cds = [c for c in spread_cols if c in sub.columns]
    if not present_cds:
        return np.empty((0, 2 + len(tenors)))

    sub = sub[sub[present_cds].notna().any(axis=1)]

    matrix = np.full((len(sub), 2 + len(tenors)), np.nan)
    matrix[:, 0] = sub["lambda_country"].to_numpy()
    matrix[:, 1] = sub["lambda_global"].to_numpy()
    for j, col in enumerate(spread_cols):
        if col in sub.columns:
            matrix[:, 2 + j] = sub[col].to_numpy()
    return matrix


# ---------------------------------------------------------------------------
# Vectorised residuals using cds_spread_batch
# ---------------------------------------------------------------------------

def compute_residuals_batch(
    theta:       np.ndarray,
    obs_matrix:  np.ndarray,
    tenors:      list[float],
    base_params: DisasterModelParams,
    lam_bar_f:   float,
    lam_bar_g:   float,
    R:           float,
) -> np.ndarray:
    """Return residual vector for least_squares (all dates × tenors in one numpy call)."""
    h0, eta_i, eta_g = float(theta[0]), float(theta[1]), float(theta[2])
    params = make_params_for_country(base_params, h0, eta_i, eta_g, lam_bar_f, lam_bar_g, R)

    lam_f_arr = obs_matrix[:, 0]
    lam_g_arr = obs_matrix[:, 1]
    obs_block = obs_matrix[:, 2:]      # (M, T)

    try:
        model_block = cds_spread_batch(
            params, np.array(tenors), lam_f_arr, lam_g_arr, n_int=N_INT_FAST,
        )                              # (M, T)
    except Exception:
        return np.full(int(np.isfinite(obs_block).sum()), 1e6)

    mask = np.isfinite(obs_block)
    if not mask.any():
        return np.array([1e6])

    model_finite = np.where(np.isfinite(model_block), model_block, 0.0)
    return (model_finite - obs_block)[mask]


# ---------------------------------------------------------------------------
# Per-country calibration (module-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _calibrate_country_worker(
    country:     str,
    obs_matrix:  np.ndarray,
    tenors:      list[float],
    base_params: DisasterModelParams,
    lam_bar_f:   float,
    lam_bar_g:   float,
    R:           float,
) -> dict:
    """Worker function: runs least_squares for one country.  Must be module-level."""
    n_obs = int(np.isfinite(obs_matrix[:, 2:]).sum()) if obs_matrix.shape[0] > 0 else 0

    if n_obs < MIN_OBS:
        return {
            "country": country, "h0": np.nan, "eta_i": np.nan, "eta_g": np.nan,
            "R": R, "rmse": np.inf, "num_obs": n_obs, "converged": False,
        }

    def _residuals(theta):
        return compute_residuals_batch(
            theta, obs_matrix, tenors, base_params, lam_bar_f, lam_bar_g, R,
        )

    try:
        result = least_squares(
            _residuals,
            x0      = X0_DEFAULT,
            bounds  = (LOWER_BOUND, UPPER_BOUND),
            method  = "trf",
            ftol    = 1e-6,
            xtol    = 1e-6,
            gtol    = 1e-6,
            max_nfev = 500,
            verbose  = 0,
        )
        converged = result.success or result.cost < 1e-2
        h0, eta_i, eta_g = float(result.x[0]), float(result.x[1]), float(result.x[2])
        rmse = float(np.sqrt(np.mean(result.fun ** 2))) if len(result.fun) > 0 else np.inf
    except Exception as exc:
        warnings.warn(f"{country}: optimisation failed — {exc}")
        return {
            "country": country, "h0": np.nan, "eta_i": np.nan, "eta_g": np.nan,
            "R": R, "rmse": np.inf, "num_obs": n_obs, "converged": False,
        }

    return {
        "country":   country,
        "h0":        h0,
        "eta_i":     eta_i,
        "eta_g":     eta_g,
        "R":         R,
        "rmse":      rmse,
        "num_obs":   n_obs,
        "converged": bool(converged),
    }


def calibrate_one_country(
    country:     str,
    country_df:  pd.DataFrame,
    tenors:      list[float],
    base_params: DisasterModelParams,
    lam_bar_g:   float,
    R:           float,
) -> dict:
    """Prepare obs_matrix and call the picklable worker."""
    lam_bar_f  = float(country_df["lambda_country"].mean())
    obs_matrix = _build_obs_matrix(country_df, tenors)
    result = _calibrate_country_worker(
        country, obs_matrix, tenors, base_params, lam_bar_f, lam_bar_g, R,
    )
    rmse_bps = result["rmse"] * 1e4 if np.isfinite(result["rmse"]) else float("inf")
    status = "OK" if result["converged"] else "WARN"
    print(
        f"  {country:4s}: h0={result['h0']:.5f}  "
        f"eta_i={result['eta_i']:.4f}  eta_g={result['eta_g']:.4f}  "
        f"R={R:.2f}  RMSE={rmse_bps:.2f}bps  n={result['num_obs']}  {status}"
    )
    return result


# ---------------------------------------------------------------------------
# Model-vs-observed table (also vectorised)
# ---------------------------------------------------------------------------

def build_model_vs_observed(
    panel:       pd.DataFrame,
    params_df:   pd.DataFrame,
    base_params: DisasterModelParams,
    tenors:      list[float],
    lam_bar_g:   float,
) -> pd.DataFrame:
    """Compute model spreads at fitted parameters using the batch pricer."""
    records     = []
    params_idx  = params_df.set_index("country")
    tenor_arr   = np.array(tenors, dtype=float)

    for country, grp in panel.groupby("country"):
        if country not in params_idx.index:
            continue
        row_p = params_idx.loc[country]
        if not row_p["converged"]:
            continue

        h0, eta_i, eta_g = float(row_p["h0"]), float(row_p["eta_i"]), float(row_p["eta_g"])
        R = float(row_p["R"]) if "R" in row_p.index else base_params.R
        lam_bar_f = float(grp["lambda_country"].mean())
        p = make_params_for_country(base_params, h0, eta_i, eta_g, lam_bar_f, lam_bar_g, R)

        grp_clean = grp.dropna(subset=["lambda_country", "lambda_global"]).copy()
        if grp_clean.empty:
            continue

        lam_f_arr = grp_clean["lambda_country"].to_numpy()
        lam_g_arr = grp_clean["lambda_global"].to_numpy()

        try:
            model_block = cds_spread_batch(
                p, tenor_arr, lam_f_arr, lam_g_arr, n_int=N_INT_FINAL,
            )                          # (M, T)
        except Exception:
            continue

        for row_idx, (_, row) in enumerate(grp_clean.iterrows()):
            for j, tau in enumerate(tenors):
                col_obs = _spread_col(tau)
                if col_obs not in row.index:
                    continue
                obs = row[col_obs]
                if not np.isfinite(obs):
                    continue
                model_s = model_block[row_idx, j]
                records.append({
                    "country":      country,
                    "date":         row["date"],
                    "tenor":        tau,
                    "spread_obs":   obs    * 1e4,
                    "spread_model": model_s * 1e4 if np.isfinite(model_s) else np.nan,
                    "residual_bps": (model_s - obs) * 1e4 if np.isfinite(model_s) else np.nan,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full calibration run (parallel across countries)
# ---------------------------------------------------------------------------

def run_calibration(
    panel:       pd.DataFrame,
    base_params: DisasterModelParams,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tenors    = get_available_tenors(panel)
    lam_bar_g = float(panel["lambda_global"].mean())
    print(f"Tenors: {tenors}   global λ̄^g = {lam_bar_g:.5f}")

    # Pre-build obs matrices (in the main process — avoids double-pickling large DataFrames)
    default_R   = base_params.R
    has_recovery = "recovery_rate" in panel.columns
    country_jobs: list[tuple] = []
    for country, grp in panel.groupby("country"):
        lam_bar_f  = float(grp["lambda_country"].mean())
        obs_matrix = _build_obs_matrix(grp, tenors)
        if has_recovery and grp["recovery_rate"].notna().any():
            R = float(grp["recovery_rate"].median())
        else:
            R = default_R
        country_jobs.append((country, obs_matrix, tenors, base_params, lam_bar_f, lam_bar_g, R))

    n_countries = len(country_jobs)
    workers = min(max_workers or os.cpu_count() or 1, n_countries)
    print(f"Calibrating {n_countries} countries with {workers} workers …\n")

    results_list: list[dict] = [None] * n_countries   # type: ignore[list-item]

    if workers == 1:
        # Single-process path: simpler, easier to debug
        for i, (country, obs_matrix, tenors_, bp, lbf, lbg, R) in enumerate(country_jobs):
            result = _calibrate_country_worker(country, obs_matrix, tenors_, bp, lbf, lbg, R)
            rmse_bps = result["rmse"] * 1e4 if np.isfinite(result["rmse"]) else float("inf")
            status   = "OK" if result["converged"] else "WARN"
            print(f"  {country:4s}: h0={result['h0']:.5f}  "
                  f"eta_i={result['eta_i']:.4f}  eta_g={result['eta_g']:.4f}  "
                  f"R={R:.2f}  RMSE={rmse_bps:.2f}bps  n={result['num_obs']}  {status}")
            results_list[i] = result
    else:
        # Multi-process path
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_calibrate_country_worker, *job): i
                for i, job in enumerate(country_jobs)
            }
            for future in as_completed(futures):
                i      = futures[future]
                result = future.result()
                rmse_bps = result["rmse"] * 1e4 if np.isfinite(result["rmse"]) else float("inf")
                status   = "OK" if result["converged"] else "WARN"
                print(f"  {result['country']:4s}: h0={result['h0']:.5f}  "
                      f"eta_i={result['eta_i']:.4f}  eta_g={result['eta_g']:.4f}  "
                      f"R={result['R']:.2f}  RMSE={rmse_bps:.2f}bps  n={result['num_obs']}  {status}")
                results_list[i] = result

    params_df = pd.DataFrame(results_list)

    print("\nBuilding model-vs-observed table …")
    model_vs_obs = build_model_vs_observed(
        panel, params_df, base_params, tenors, lam_bar_g,
    )
    return params_df, model_vs_obs


def save_results(
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    results_dir:  str = RESULTS_DIR,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    params_df.to_csv(PARAMS_OUT, index=False)
    model_vs_obs.to_csv(MODEL_OBS_OUT, index=False)
    print(f"Saved → {PARAMS_OUT}")
    print(f"Saved → {MODEL_OBS_OUT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(PANEL_PATH):
        raise FileNotFoundError(
            f"Panel not found: {PANEL_PATH}\n"
            "Run  python -m calibration.build_panel  first."
        )

    print("Loading panel …")
    panel = load_panel(PANEL_PATH)
    print(f"  {panel['country'].nunique()} countries, {len(panel):,} rows")

    base_params = DisasterModelParams()
    base_params.compute_b_sdf()
    print(f"  b_sdf = {base_params.b_sdf:.6f}")

    print("\nCalibrating …")
    params_df, model_vs_obs = run_calibration(panel, base_params)

    print("\nParameter table:")
    print(params_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    save_results(params_df, model_vs_obs)


if __name__ == "__main__":
    main()
