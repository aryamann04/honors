"""Foreign defaultable bond pricing, credit spreads, and sensitivities.

Reduced model (v = 0) only — uses closed-form functions from core.closed_form.
"""
import numpy as np
from core.parameters import DisasterModelParams
from core.closed_form import (
    defaultable_coeffs_cf,
    b_Df_inf, b_Dg_inf, y_D_star_inf,
)
from models.risk_free import rf_yield


# ──────────────────────────────────────────────────────────────────────────────
# Prices and yields
# ──────────────────────────────────────────────────────────────────────────────

def def_log_price(params: DisasterModelParams, tau: float,
                  lam_f: float, lam_g: float) -> float:
    """log B_D*(t, T) = a_D*(τ) + b_{D,f}*(τ)λ^f + b_{D,g}*(τ)λ^g."""
    a_D, b_Df, b_Dg = defaultable_coeffs_cf(params, tau)
    return a_D + b_Df * lam_f + b_Dg * lam_g


def def_yield(params: DisasterModelParams, tau: float,
              lam_f: float, lam_g: float) -> float:
    """y_D*(τ) = −log B_D*(t, T) / τ."""
    return -def_log_price(params, tau, lam_f, lam_g) / tau


def def_yield_curve(params: DisasterModelParams, tau_grid: np.ndarray,
                    lam_f: float, lam_g: float) -> np.ndarray:
    return np.array([def_yield(params, float(t), lam_f, lam_g) for t in tau_grid])


# ──────────────────────────────────────────────────────────────────────────────
# Credit spread  s*(τ) = y_D*(τ) − y*(τ)
# ──────────────────────────────────────────────────────────────────────────────

def credit_spread(params: DisasterModelParams, tau: float,
                  lam_f: float, lam_g: float) -> float:
    """s*(τ) = y_D*(τ) − y*(τ)."""
    return def_yield(params, tau, lam_f, lam_g) - rf_yield(params, tau, lam_f, lam_g)


def credit_spread_curve(params: DisasterModelParams, tau_grid: np.ndarray,
                        lam_f: float, lam_g: float) -> np.ndarray:
    return np.array([credit_spread(params, float(t), lam_f, lam_g) for t in tau_grid])


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivities  ∂s*/∂λ_f  and  ∂s*/∂λ_g  (central differences)
# ──────────────────────────────────────────────────────────────────────────────

def spread_sensitivity(params: DisasterModelParams, tau_grid: np.ndarray,
                       lam_f: float, lam_g: float, d_lam: float = 1e-5):
    """Return (ds/dlam_f, ds/dlam_g) as arrays over tau_grid."""
    def _s(t, lf, lg):
        return credit_spread(params, float(t), lf, lg)

    dsf = np.array([(_s(t, lam_f + d_lam, lam_g) - _s(t, lam_f - d_lam, lam_g)) / (2.0 * d_lam)
                    for t in tau_grid])
    dsg = np.array([(_s(t, lam_f, lam_g + d_lam) - _s(t, lam_f, lam_g - d_lam)) / (2.0 * d_lam)
                    for t in tau_grid])
    return dsf, dsg
