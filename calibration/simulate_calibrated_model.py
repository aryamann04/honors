"""
simulate_calibrated_model.py
============================
Run the calibrated-model CIR simulation for all countries and save outputs
to calibration/results/simulation/.

Builds on calibration.simulation (CIR engine + vectorised batch CDS pricer).

Outputs
-------
  calibration/results/simulation/simulated_paths.csv
      First 120 monthly steps for US, DE, ID, TR (for Figure 8.1.1).
  calibration/results/simulation/simulated_5y_all.csv
      Full 5Y CDS spread paths for all countries (for moments / distribution).

CLI
---
    python -m calibration.simulate_calibrated_model
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

from core.parameters import DisasterModelParams
from calibration.simulation import run_all_countries, SimulationResult, tenor_index

PANEL_PATH  = os.path.join(BASE_DIR, "processed_data", "panel.csv")
PARAMS_PATH = os.path.join(BASE_DIR, "results", "parameters.csv")
SIM_DIR     = os.path.join(BASE_DIR, "results", "simulation")

PATHS_COUNTRIES = ["US", "DE", "ID", "TR"]
PATHS_SHOW      = 120    # months shown in paths figure (10 years)
N_MONTHS        = 2000   # post-burn simulation length
BURN_IN         = 120    # 10-year monthly burn-in
SEED            = 42
N_INT           = 100    # integration points for batch CDS pricer


def run_and_save(
    panel:       pd.DataFrame,
    params_df:   pd.DataFrame,
    base_params: DisasterModelParams,
) -> dict[str, SimulationResult]:
    """Run calibrated simulation for all converged countries and save outputs."""
    os.makedirs(SIM_DIR, exist_ok=True)

    if "converged" in params_df.columns:
        run_df = params_df[params_df["converged"].fillna(False).astype(bool)]
    else:
        run_df = params_df
    countries = run_df["country"].tolist()

    print(
        f"Simulating {len(countries)} countries  "
        f"(N_MONTHS={N_MONTHS}, BURN_IN={BURN_IN}, SEED={SEED}) ..."
    )
    results = run_all_countries(
        params_df, panel, base_params,
        n_months=N_MONTHS, burn_in=BURN_IN,
        seed=SEED, n_int=N_INT,
        countries=countries,
    )

    # ── simulated_paths.csv: first PATHS_SHOW months for 4 countries ──────────
    records: list[dict] = []
    for c in PATHS_COUNTRIES:
        sim = results.get(c)
        if sim is None:
            print(f"  WARNING: {c} not in simulation results; skipping.")
            continue
        n   = min(PATHS_SHOW, len(sim.lam_g))
        ti5 = tenor_index(sim, 5.0)
        for t in range(n):
            records.append({
                "country":        c,
                "t":              t,
                "lambda_global":  float(sim.lam_g[t]),
                "lambda_country": float(sim.lam_i[t]),
                "h_star":         float(sim.h_star[t]),
                "cds_5y_bps":     float(sim.spreads[t, ti5] * 1e4),
            })
    out = os.path.join(SIM_DIR, "simulated_paths.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"  Saved → {out}")

    # ── simulated_5y_all.csv: full paths for all countries ────────────────────
    records5: list[dict] = []
    for c, sim in results.items():
        ti5   = tenor_index(sim, 5.0)
        lam_g = sim.lam_g
        lam_i = sim.lam_i
        s5    = sim.spreads[:, ti5] * 1e4
        for t in range(len(lam_g)):
            records5.append({
                "country":        c,
                "t":              t,
                "lambda_global":  float(lam_g[t]),
                "lambda_country": float(lam_i[t]),
                "cds_5y_bps":     float(s5[t]),
            })
    out5 = os.path.join(SIM_DIR, "simulated_5y_all.csv")
    pd.DataFrame(records5).to_csv(out5, index=False)
    print(f"  Saved → {out5}")

    print(f"\nSimulation complete. Outputs in {SIM_DIR}")
    return results


def main() -> None:
    for path in [PANEL_PATH, PARAMS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    print("Loading panel and parameters ...")
    panel     = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    params_df = pd.read_csv(PARAMS_PATH)

    base_params = DisasterModelParams()
    base_params.compute_b_sdf()
    print(f"  b_sdf = {base_params.b_sdf:.6f}")

    run_and_save(panel, params_df, base_params)


if __name__ == "__main__":
    main()
