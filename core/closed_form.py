"""Closed-form (v = 0) affine bond and CDS coefficients.

All Riccati evaluations delegate to core.riccati.psi / phi_integral.

Notation (consistent with the LaTeX draft):
  φ   = b_sdf · σ_λ² − κ
  K   = exp(−γZ)(1 − exp Z)  > 0         risk-free Riccati constant
  A_f = −K + (1−R)η_f  (< 0 typically)   defaultable bond constant
  A_g = −K + (1−R)η_g
  x_f = −A_f = K − (1−R)η_f              Riccati forcing for defaultable bond
  Â_f = −K + η_f                          CDS G-object constant (no (1−R))
  −η_f                                    Riccati forcing for CDS c-coefficient (Bug-2 fix)
"""
import numpy as np
from core.parameters import DisasterModelParams
from core.riccati import psi, phi_integral, lower_fixed_point


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _phi_sig2(params: DisasterModelParams):
    """Return (φ, σ²) after ensuring b_sdf is computed."""
    params.compute_b_sdf()
    sig2 = params.sigma_lambda ** 2
    phi  = float(params.b_sdf) * sig2 - params.kappa
    return phi, sig2


def K_const(params: DisasterModelParams) -> float:
    """K = exp(−γZ)(1 − exp Z) > 0.  Risk-free Riccati constant under v = 0."""
    return np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))


# ──────────────────────────────────────────────────────────────────────────────
# Risk-free foreign bond
# ──────────────────────────────────────────────────────────────────────────────

def b_star_cf(params: DisasterModelParams, tau: float) -> float:
    """b*(τ) = Ψ(τ; K)."""
    phi, sig2 = _phi_sig2(params)
    return psi(tau, K_const(params), phi, sig2)


def a_star_cf(params: DisasterModelParams, tau: float) -> float:
    """a*(τ) = −(β + μ − γσ²)τ + κ(λ̄^f + λ̄^g) Φ(τ; K)."""
    phi, sig2 = _phi_sig2(params)
    A0      = params.beta + params.mu - params.gamma * params.sigma_c ** 2
    lam_bar = params.lam_bar_f + params.lam_bar_g
    Phi     = phi_integral(tau, K_const(params), phi, sig2)
    return -A0 * tau + params.kappa * lam_bar * Phi


def b_star_inf(params: DisasterModelParams):
    """b*_∞ = lower fixed point of Ψ(·; K).  None if trigonometric branch."""
    phi, sig2 = _phi_sig2(params)
    return lower_fixed_point(K_const(params), phi, sig2)


def y_star_inf(params: DisasterModelParams):
    """Long-run foreign risk-free yield  y*_∞ = A0 − κ(λ̄^f+λ̄^g) b*_∞.

    Note: b*_∞ is independent of η_f / η_g, so y*_∞ is flat in those parameters.
    Returns None if the risk-free loading is on the trigonometric branch (no finite limit).
    """
    b_inf   = b_star_inf(params)
    if b_inf is None:
        return None
    A0      = params.beta + params.mu - params.gamma * params.sigma_c ** 2
    lam_bar = params.lam_bar_f + params.lam_bar_g
    return A0 - params.kappa * lam_bar * b_inf


# ──────────────────────────────────────────────────────────────────────────────
# Domestic risk-free bond
# ──────────────────────────────────────────────────────────────────────────────

def a_dom_cf(params: DisasterModelParams, tau: float) -> float:
    """a(τ) for the domestic risk-free bond (λ̄^h + λ̄^g instead of λ̄^f + λ̄^g)."""
    phi, sig2 = _phi_sig2(params)
    A0      = params.beta + params.mu - params.gamma * params.sigma_c ** 2
    lam_bar = params.lam_bar_g + params.lam_bar_h
    Phi     = phi_integral(tau, K_const(params), phi, sig2)
    return -A0 * tau + params.kappa * lam_bar * Phi


# ──────────────────────────────────────────────────────────────────────────────
# Defaultable foreign bond
# ──────────────────────────────────────────────────────────────────────────────

def _defaultable_Ai(params: DisasterModelParams):
    """A0, A_f, A_g constants for the defaultable bond.

    A_f = exp(−γZ)(exp Z − 1) + (1−R)η_f = −K + (1−R)η_f
    x_f = −A_f = K − (1−R)η_f      (Riccati forcing: x > 0 → exponential branch)
    """
    K  = K_const(params)
    A0 = params.beta + params.mu - params.gamma * params.sigma_c**2 + (1.0 - params.R) * params.h0_star
    Af = -K + (1.0 - params.R) * params.eta1
    Ag = -K + (1.0 - params.R) * params.eta2
    return A0, Af, Ag


def defaultable_coeffs_cf(params: DisasterModelParams, tau: float):
    """(a_D*(τ), b_{D,f}*(τ), b_{D,g}*(τ)) via the reduced-model closed form.

        b_{D,i}*(τ) = Ψ(τ; −A_i)
        a_D*(τ)     = −A_0 τ + κ λ̄^f Φ(τ; −A_f) + κ λ̄^g Φ(τ; −A_g)
    """
    phi, sig2 = _phi_sig2(params)
    A0, Af, Ag = _defaultable_Ai(params)

    x_f = -Af
    x_g = -Ag

    b_Df  = psi(tau, x_f, phi, sig2)
    b_Dg  = psi(tau, x_g, phi, sig2)
    Phi_f = phi_integral(tau, x_f, phi, sig2)
    Phi_g = phi_integral(tau, x_g, phi, sig2)

    a_D = (-A0 * tau
           + params.kappa * params.lam_bar_f * Phi_f
           + params.kappa * params.lam_bar_g * Phi_g)
    return a_D, b_Df, b_Dg


def b_Df_inf(params: DisasterModelParams):
    """b_{D,f,∞}* = lower fixed point of Ψ(·; −A_f)."""
    phi, sig2 = _phi_sig2(params)
    _, Af, _ = _defaultable_Ai(params)
    return lower_fixed_point(-Af, phi, sig2)


def b_Dg_inf(params: DisasterModelParams):
    """b_{D,g,∞}* = lower fixed point of Ψ(·; −A_g)."""
    phi, sig2 = _phi_sig2(params)
    _, _, Ag = _defaultable_Ai(params)
    return lower_fixed_point(-Ag, phi, sig2)


def y_D_star_inf(params: DisasterModelParams):
    """Long-run defaultable yield y_{D,∞}* = A0 − κλ̄^f b_{D,f,∞} − κλ̄^g b_{D,g,∞}.

    Returns None if either loading is on the trigonometric branch (no finite limit).
    """
    A0, _, _ = _defaultable_Ai(params)
    bDf = b_Df_inf(params)
    bDg = b_Dg_inf(params)
    if bDf is None or bDg is None:
        return None
    return A0 - params.kappa * params.lam_bar_f * bDf - params.kappa * params.lam_bar_g * bDg


# ──────────────────────────────────────────────────────────────────────────────
# CDS affine objects  G(t, t+τ)  and  K(t, t+τ)
# ──────────────────────────────────────────────────────────────────────────────

def _cds_Ahat(params: DisasterModelParams):
    """Â constants for G(t, t+τ): full hazard rate, no (1−R) factor.

        Â_f = −K + η_f,   Â_0 = β + μ − γσ² + h_0*
    """
    K       = K_const(params)
    A0_hat  = params.beta + params.mu - params.gamma * params.sigma_c**2 + params.h0_star
    Af_hat  = -K + params.eta1
    Ag_hat  = -K + params.eta2
    return A0_hat, Af_hat, Ag_hat


def cds_G_coeffs_cf(params: DisasterModelParams, tau: float):
    """(a_G(τ), b_{G,f}(τ), b_{G,g}(τ)) for G(t, t+τ) = exp(a_G + b_Gf λ_f + b_Gg λ_g).

    G is the discounted survival probability.  Uses Â_i (full hazard, no (1−R)).
    """
    phi, sig2 = _phi_sig2(params)
    A0_hat, Af_hat, Ag_hat = _cds_Ahat(params)

    x_f = -Af_hat
    x_g = -Ag_hat

    b_Gf  = psi(tau, x_f, phi, sig2)
    b_Gg  = psi(tau, x_g, phi, sig2)
    Phi_f = phi_integral(tau, x_f, phi, sig2)
    Phi_g = phi_integral(tau, x_g, phi, sig2)

    a_G = (-A0_hat * tau
           + params.kappa * params.lam_bar_f * Phi_f
           + params.kappa * params.lam_bar_g * Phi_g)
    return a_G, b_Gf, b_Gg


def cds_c_coeff(params: DisasterModelParams, tau: float, eta_i: float) -> float:
    """c_f(τ) or c_g(τ) coefficient for K(t, t+τ).

    BUG-2 FIX: the Riccati forcing for the integrating factor is x = −η_i,
    NOT x = −Â_i.  The old code used Â_i = −K + η_i in the discriminant,
    which incorrectly mixed the jump-risk-premium K-term with the pure hazard
    loading η_i.  The discriminant is now δ(−η_i) = φ² + 2σ²η_i.

    Formula:  c_i(τ) = η_i · exp(φτ + σ² Φ(τ; −η_i))
    """
    phi, sig2 = _phi_sig2(params)
    x   = -eta_i                               # Riccati forcing: −η_i
    Phi = phi_integral(tau, x, phi, sig2)      # Φ(τ; −η_i)
    return eta_i * np.exp(phi * tau + sig2 * Phi)


def cds_K_coeffs_cf(params: DisasterModelParams, tau: float, n_int: int = 400):
    """(c_0(τ), c_f(τ), c_g(τ)) for K(t, t+τ) = [c_0 + c_f λ_f + c_g λ_g] G(t, t+τ).

    c_0(τ) = h_0* + κ ∫₀^τ [λ̄^f c_f(s) + λ̄^g c_g(s)] ds
    """
    cf_tau = cds_c_coeff(params, tau, params.eta1)
    cg_tau = cds_c_coeff(params, tau, params.eta2)

    if tau == 0.0:
        return params.h0_star, cf_tau, cg_tau

    grid    = np.linspace(0.0, tau, n_int)
    cf_grid = np.array([cds_c_coeff(params, s, params.eta1) for s in grid])
    cg_grid = np.array([cds_c_coeff(params, s, params.eta2) for s in grid])
    intgd   = params.lam_bar_f * cf_grid + params.lam_bar_g * cg_grid
    c0_tau  = params.h0_star + params.kappa * np.trapz(intgd, grid)

    return c0_tau, cf_tau, cg_tau
