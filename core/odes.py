"""ODE solvers for the general v ≥ 0 model.

All bond-price affine coefficients are obtained by integrating the
Feynman-Kac ODEs from τ = 0 using the Radau method (stiff solver).

Hazard forcing in the defaultable bond uses the corrected constants:
    A_f = (1−R) η_f,   A_g = (1−R) η_g
which is the fix for the Bug-1 double-counting that existed in draft2/.
The term exp(b·v − γZ)(exp(b_D·v) − exp Z) in the ODE already provides
the risk-premium contribution; adding C = exp(−γZ)(exp Z − 1) into A_f
(as draft2 did) double-counted that contribution at v = 0.
"""
import numpy as np
from scipy.integrate import solve_ivp
from core.parameters import DisasterModelParams


def _solve_ode(ode, tau: float, y0):
    sol = solve_ivp(
        ode, (0.0, tau), y0=list(y0),
        method="Radau", rtol=1e-9, atol=1e-12,
    )
    return sol.y[:, -1]


# ──────────────────────────────────────────────────────────────────────────────
# Foreign risk-free bond  B*(t, T) = exp(a*(τ) + b*(τ)(λ^f + λ^g))
# ──────────────────────────────────────────────────────────────────────────────

def riskfree_ode(params: DisasterModelParams, tau: float):
    """(a*(τ), b*(τ)) for the foreign risk-free bond."""
    params.compute_b_sdf()
    b_bar   = params.b_sdf
    C       = np.exp(b_bar * params.v - params.gamma * params.Z)
    lam_bar = params.lam_bar_f + params.lam_bar_g

    def ode(_, y):
        a, b = y
        da = (params.kappa * lam_bar * b
              - params.beta - params.mu + params.gamma * params.sigma_c**2)
        db = ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b
              + 0.5 * params.sigma_lambda**2 * b**2
              + C * (np.exp(b * params.v) - np.exp(params.Z)))
        return [da, db]

    if tau == 0.0:
        return 0.0, 0.0
    a_tau, b_tau = _solve_ode(ode, tau, [0.0, 0.0])
    return float(a_tau), float(b_tau)


# ──────────────────────────────────────────────────────────────────────────────
# Domestic risk-free bond
# ──────────────────────────────────────────────────────────────────────────────

def domestic_riskfree_ode(params: DisasterModelParams, tau: float):
    """(a(τ), b(τ)) for the domestic risk-free bond."""
    params.compute_b_sdf()
    b_bar   = params.b_sdf
    C       = np.exp(b_bar * params.v - params.gamma * params.Z)
    lam_bar = params.lam_bar_h + params.lam_bar_g

    def ode(_, y):
        a, b = y
        da = (params.kappa * lam_bar * b
              - params.beta - params.mu + params.gamma * params.sigma_c**2)
        db = ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b
              + 0.5 * params.sigma_lambda**2 * b**2
              + C * (np.exp(b * params.v) - np.exp(params.Z)))
        return [da, db]

    if tau == 0.0:
        return 0.0, 0.0
    a_tau, b_tau = _solve_ode(ode, tau, [0.0, 0.0])
    return float(a_tau), float(b_tau)


# ──────────────────────────────────────────────────────────────────────────────
# Foreign defaultable bond  B_D*(t, T) = exp(a_D*(τ) + b_{D,f}*(τ)λ^f + b_{D,g}*(τ)λ^g)
# ──────────────────────────────────────────────────────────────────────────────

def defaultable_ode(params: DisasterModelParams, tau: float):
    """(a_D*(τ), b_{D,f}*(τ), b_{D,g}*(τ)) for the foreign defaultable bond.

    ODE for loading i ∈ {f, g}:
        db_{D,i}/dτ = (b σ_λ² − κ − v) b_{D,i}
                     + ½ σ_λ² b_{D,i}²
                     + exp(bv − γZ)(exp(b_{D,i} v) − exp Z)
                     − (1−R) η_i

    Bug-1 fix: A_i = (1−R) η_i only (no C-term; it is already in the ODE).
    """
    params.compute_b_sdf()
    b_bar = params.b_sdf
    C     = np.exp(b_bar * params.v - params.gamma * params.Z)
    A0    = params.beta + params.mu - params.gamma * params.sigma_c**2 + (1.0 - params.R) * params.h0_star
    Af    = (1.0 - params.R) * params.eta1
    Ag    = (1.0 - params.R) * params.eta2

    def ode(_, y):
        a, b_f, b_g = y
        db_f = ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b_f
                + 0.5 * params.sigma_lambda**2 * b_f**2
                + C * (np.exp(b_f * params.v) - np.exp(params.Z)) - Af)
        db_g = ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b_g
                + 0.5 * params.sigma_lambda**2 * b_g**2
                + C * (np.exp(b_g * params.v) - np.exp(params.Z)) - Ag)
        da   = params.kappa * (params.lam_bar_f * b_f + params.lam_bar_g * b_g) - A0
        return [da, db_f, db_g]

    if tau == 0.0:
        return 0.0, 0.0, 0.0
    a_tau, b_f_tau, b_g_tau = _solve_ode(ode, tau, [0.0, 0.0, 0.0])
    return float(a_tau), float(b_f_tau), float(b_g_tau)
