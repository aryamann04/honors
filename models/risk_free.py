"""Foreign and domestic risk-free bond pricing.

Reduced model (v = 0): use closed-form functions from core.closed_form.
General model (v > 0): use ODE solvers from core.odes.
"""
import numpy as np
from core.parameters import DisasterModelParams
from core.closed_form import (
    b_star_cf, a_star_cf, a_dom_cf,
    b_star_inf, y_star_inf,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reduced model  (v = 0, closed-form)
# ──────────────────────────────────────────────────────────────────────────────

def rf_log_price(params: DisasterModelParams, tau: float,
                 lam_f: float, lam_g: float) -> float:
    """log B*(t, T) = a*(τ) + b*(τ)(λ^f + λ^g)."""
    return a_star_cf(params, tau) + b_star_cf(params, tau) * (lam_f + lam_g)


def rf_yield(params: DisasterModelParams, tau: float,
             lam_f: float, lam_g: float) -> float:
    """y*(τ) = −log B*(t, T) / τ."""
    return -rf_log_price(params, tau, lam_f, lam_g) / tau


def rf_yield_curve(params: DisasterModelParams, tau_grid: np.ndarray,
                   lam_f: float, lam_g: float) -> np.ndarray:
    """y*(τ) evaluated over a maturity grid."""
    return np.array([rf_yield(params, float(t), lam_f, lam_g) for t in tau_grid])


def dom_log_price(params: DisasterModelParams, tau: float,
                  lam_h: float, lam_g: float) -> float:
    """log B(t, T) = a(τ) + b*(τ)(λ^h + λ^g)  for domestic risk-free bond."""
    return a_dom_cf(params, tau) + b_star_cf(params, tau) * (lam_h + lam_g)


def dom_yield(params: DisasterModelParams, tau: float,
              lam_h: float, lam_g: float) -> float:
    return -dom_log_price(params, tau, lam_h, lam_g) / tau
