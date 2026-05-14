"""
simulation.py
=============
Calibrated-model CIR simulation engine.

Simulates λ^g (global) and λ^i (country-specific) as independent CIR processes
under the physical measure using the full-truncation Euler scheme, then evaluates
model-implied CDS spreads at every simulated state via the vectorised batch pricer.

Physical-measure CIR:
    dλ = κ(λ̄ − λ) dt + σ_λ √λ dW

Full-truncation Euler (preserves non-negativity):
    λ_{t+1} = max{ λ_t + κ(λ̄ − λ_t)dt + σ_λ √max(λ_t,0) √dt ε_t , 0 }

Note: plugging simulated states into the affine CDS pricer gives the *model-implied*
spread conditional on the current intensity state, consistently combining physical
simulation with risk-neutral pricing.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.parameters import DisasterModelParams
from models.cds import cds_spread_batch

TENORS_DEFAULT  = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
N_MONTHS        = 20_000          # post-burn-in simulation length
BURN_IN         = 1_200           # 100 years of monthly burn-in
DT              = 1.0 / 12.0     # monthly time-step
SEED            = 42
N_INT_SIM       = 100             # integration points in batch pricer during simulation


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Holds all outputs from one country's calibrated simulation."""
    country:    str
    lam_g:      np.ndarray   # (n_months,) global intensity path
    lam_i:      np.ndarray   # (n_months,) country-specific intensity path
    h_star:     np.ndarray   # (n_months,) total hazard rate
    spreads:    np.ndarray   # (n_months, n_tenors) CDS spreads in decimal
    tenors:     list         # list of float tenor years
    h0:         float
    eta_f:      float        # loading on λ^i  (= params.eta1)
    eta_g:      float        # loading on λ^g  (= params.eta2)
    R:          float
    lam_bar_f:  float        # long-run mean of λ^i
    lam_bar_g:  float        # long-run mean of λ^g


# ---------------------------------------------------------------------------
# CIR simulator
# ---------------------------------------------------------------------------

def simulate_cir(
    lam_bar:   float,
    kappa:     float,
    sigma_lam: float,
    n_total:   int,
    dt:        float = DT,
    rng:       Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Full-truncation Euler scheme for one CIR factor.

    Returns an array of length *n_total* starting from lam_bar.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    sqrt_dt = np.sqrt(dt)
    eps     = rng.standard_normal(n_total - 1)
    path    = np.empty(n_total)
    path[0] = max(lam_bar, 0.0)
    for t in range(n_total - 1):
        drift    = kappa * (lam_bar - path[t]) * dt
        vol      = sigma_lam * np.sqrt(max(path[t], 0.0)) * sqrt_dt
        path[t + 1] = max(path[t] + drift + vol * eps[t], 0.0)
    return path


# ---------------------------------------------------------------------------
# Single-country simulation
# ---------------------------------------------------------------------------

def make_params_for_country(
    base_params: DisasterModelParams,
    h0:          float,
    eta_f:       float,
    eta_g:       float,
    lam_bar_f:   float,
    lam_bar_g:   float,
    R:           float,
) -> DisasterModelParams:
    return dataclasses.replace(
        base_params,
        h0_star   = float(h0),
        eta1      = float(eta_f),
        eta2      = float(eta_g),
        lam_bar_f = float(lam_bar_f),
        lam_bar_g = float(lam_bar_g),
        R         = float(R),
        b_sdf     = base_params.b_sdf,
    )


def run_simulation(
    country:   str,
    h0:        float,
    eta_f:     float,
    eta_g:     float,
    R:         float,
    lam_bar_f: float,
    lam_bar_g: float,
    base_params: DisasterModelParams,
    tenors:    list = TENORS_DEFAULT,
    n_months:  int  = N_MONTHS,
    burn_in:   int  = BURN_IN,
    dt:        float = DT,
    seed:      int  = SEED,
    n_int:     int  = N_INT_SIM,
) -> SimulationResult:
    """Simulate one country's calibrated model and return paths + model spreads."""
    rng     = np.random.default_rng(seed)
    n_total = n_months + burn_in

    lam_g_full = simulate_cir(lam_bar_g, base_params.kappa, base_params.sigma_lambda,
                              n_total, dt, rng)
    lam_i_full = simulate_cir(lam_bar_f, base_params.kappa, base_params.sigma_lambda,
                              n_total, dt, rng)

    lam_g = lam_g_full[burn_in:]
    lam_i = lam_i_full[burn_in:]

    h_star = h0 + eta_f * lam_i + eta_g * lam_g

    params      = make_params_for_country(base_params, h0, eta_f, eta_g,
                                         lam_bar_f, lam_bar_g, R)
    tenors_arr  = np.array(tenors, dtype=float)
    spreads     = cds_spread_batch(params, tenors_arr, lam_i, lam_g, n_int=n_int)

    return SimulationResult(
        country   = country,
        lam_g     = lam_g,
        lam_i     = lam_i,
        h_star    = h_star,
        spreads   = spreads,
        tenors    = list(tenors),
        h0        = h0,
        eta_f     = eta_f,
        eta_g     = eta_g,
        R         = R,
        lam_bar_f = lam_bar_f,
        lam_bar_g = lam_bar_g,
    )


# ---------------------------------------------------------------------------
# Multi-country runner
# ---------------------------------------------------------------------------

def run_all_countries(
    params_df:   pd.DataFrame,
    panel:       pd.DataFrame,
    base_params: DisasterModelParams,
    tenors:      list = TENORS_DEFAULT,
    n_months:    int  = N_MONTHS,
    burn_in:     int  = BURN_IN,
    seed:        int  = SEED,
    n_int:       int  = N_INT_SIM,
    countries:   Optional[list] = None,
) -> dict[str, SimulationResult]:
    """Run simulation for all converged countries. Returns country→SimulationResult."""
    lam_bar_g  = float(panel["lambda_global"].mean())
    converged  = params_df[params_df.get("converged", pd.Series(True, index=params_df.index))
                           .fillna(False).astype(bool)].copy()
    if countries:
        converged = converged[converged["country"].isin(countries)]

    results: dict[str, SimulationResult] = {}
    for _, row in converged.iterrows():
        country = row["country"]
        grp     = panel[panel["country"] == country]
        if grp.empty:
            continue
        lam_bar_f = float(grp["lambda_country"].mean())
        h0    = float(row["h0"])
        eta_f = float(row["eta_i"])
        eta_g = float(row["eta_g"])
        R_val = float(row["R"]) if "R" in row.index and pd.notna(row["R"]) else 0.4

        print(f"  Simulating {country} …", flush=True)
        results[country] = run_simulation(
            country   = country,
            h0        = h0, eta_f = eta_f, eta_g = eta_g, R = R_val,
            lam_bar_f = lam_bar_f, lam_bar_g = lam_bar_g,
            base_params = base_params,
            tenors    = tenors, n_months = n_months,
            burn_in   = burn_in, seed = seed, n_int = n_int,
        )

    return results


# ---------------------------------------------------------------------------
# Convenience helpers used by table/plot modules
# ---------------------------------------------------------------------------

def tenor_index(sim: SimulationResult, tau: float) -> int:
    """Return the column index for a given tenor in sim.spreads."""
    tenors = np.array(sim.tenors)
    idx    = np.argmin(np.abs(tenors - tau))
    return int(idx)


def spread_series(sim: SimulationResult, tau: float) -> np.ndarray:
    """Simulated spread path at a given tenor (decimal)."""
    return sim.spreads[:, tenor_index(sim, tau)]


def counterfactual_spreads(
    sim:         SimulationResult,
    base_params: DisasterModelParams,
    tau:         float = 5.0,
    n_int:       int   = N_INT_SIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Option-B counterfactual spreads for variance decomposition.

    Returns:
        global_only : spreads with λ^i held at lam_bar_f   (only λ^g varies)
        country_only: spreads with λ^g held at lam_bar_g   (only λ^i varies)
    """
    params     = make_params_for_country(base_params, sim.h0, sim.eta_f, sim.eta_g,
                                        sim.lam_bar_f, sim.lam_bar_g, sim.R)
    tenors_arr = np.array([tau])
    M          = len(sim.lam_g)

    lam_i_const = np.full(M, sim.lam_bar_f)
    lam_g_const = np.full(M, sim.lam_bar_g)

    global_only  = cds_spread_batch(params, tenors_arr, lam_i_const, sim.lam_g,
                                    n_int=n_int)[:, 0]
    country_only = cds_spread_batch(params, tenors_arr, sim.lam_i, lam_g_const,
                                    n_int=n_int)[:, 0]
    return global_only, country_only
