"""Reduced-model Riccati lemma — single source of truth for Ψ, Φ, δ.

For the scalar ODE

    dy/dτ = (σ²/2) y² + φ y + x,   y(0) = 0,

define the discriminant

    δ(x) = φ² − 2σ²x.

Three branches:
  δ > 0  →  exponential  (globally defined; y → lower fixed point y₋ = (−φ−δ)/σ²)
  δ < 0  →  trigonometric (finite blowup at τ* = (π − 2 arctan(φ/|δ|^½)) / |δ|^½)
  δ = 0  →  degenerate parabolic

Φ(τ; x) = ∫₀^τ Ψ(s; x) ds  is needed for the affine constant coefficients a*(τ).

Sign conventions used throughout:
  φ = b_sdf · σ_λ² − κ          (linear coefficient; φ < 0 required for stability)
  K  = exp(−γZ)(1 − exp Z) > 0   (risk-free Riccati constant, v = 0)
  x  = K            for the risk-free bond
  x  = −A_f, −A_g   for the defaultable bond (A_i > 0, so x < 0, giving δ > 0)
  x  = −η_i         for the CDS c-coefficient (η_i > 0, so x < 0)
"""
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core scalar functions
# ──────────────────────────────────────────────────────────────────────────────

def discriminant(x: float, phi: float, sig2: float) -> float:
    """δ(x) = φ² − 2σ²x."""
    return phi * phi - 2.0 * sig2 * x


def blowup_time(phi: float, sig2: float, x: float) -> float:
    """Finite blowup time τ* for the trigonometric branch (δ(x) < 0).

    Returns inf if δ(x) ≥ 0 (no blowup).
    """
    disc = discriminant(x, phi, sig2)
    if disc >= 0.0:
        return np.inf
    delta_t = np.sqrt(-disc)
    a0 = np.arctan(phi / delta_t)
    return (np.pi - 2.0 * a0) / delta_t


def psi(tau: float, x: float, phi: float, sig2: float) -> float:
    """Ψ(τ; x): closed-form solution of the reduced-model Riccati ODE, y(0)=0."""
    disc = discriminant(x, phi, sig2)

    if disc > 0.0:
        delta = np.sqrt(disc)
        # Numerically stable: write (1 − e^{−δτ}) / e^{δτ} = (e^{δτ}−1)/e^{δτ}
        u = delta * tau
        if u > 500.0:
            frac = 1.0
        else:
            frac = (np.exp(u) - 1.0) / np.exp(u)
        d = (delta - phi) + (phi + delta) * np.exp(-u)
        return 2.0 * x * frac / d

    if disc < 0.0:
        delta_t = np.sqrt(-disc)
        a0  = np.arctan(phi / delta_t)
        ang = 0.5 * delta_t * tau + a0
        return (delta_t * np.tan(ang) - phi) / sig2

    # Degenerate: δ = 0, φ² = 2σ²x
    denom = 1.0 - phi * tau
    if abs(denom) < 1e-14:
        return np.inf
    return (x * tau) / denom


def phi_integral(tau: float, x: float, phi: float, sig2: float) -> float:
    """Φ(τ; x) = ∫₀^τ Ψ(s; x) ds.

    Exponential branch: closed-form log expression (= stable_log_term / σ²
    in the legacy code).
    Trigonometric branch: closed-form arctan-log expression.
    Degenerate: numerical fallback.
    """
    disc = discriminant(x, phi, sig2)

    if disc > 0.0:
        delta = np.sqrt(disc)
        u  = delta * tau
        d  = (delta - phi) + (phi + delta) * np.exp(-u)
        # log_term = (δ−φ)τ + 2[ln(2δ) − ln(d) − δτ]
        log_term = (delta - phi) * tau + 2.0 * (np.log(2.0 * delta) - np.log(d) - u)
        return log_term / sig2

    if disc < 0.0:
        delta_t = np.sqrt(-disc)
        a0  = np.arctan(phi / delta_t)
        ang = 0.5 * delta_t * tau + a0
        log_term = (-phi) * tau - 2.0 * np.log(np.cos(ang) / np.cos(a0))
        return log_term / sig2

    # Degenerate: numerical fallback (rare in practice)
    grid = np.linspace(0.0, tau, 4001)
    vals = np.array([psi(float(s), x, phi, sig2) for s in grid])
    return float(np.trapz(vals, grid))


def lower_fixed_point(x: float, phi: float, sig2: float):
    """Stable (lower) fixed point y₋ = (−φ − δ)/σ², exponential branch only.

    This is the limit Ψ(τ; x) → y₋ as τ → ∞.
    Returns None if δ(x) ≤ 0 (no finite limit).
    """
    disc = discriminant(x, phi, sig2)
    if disc <= 0.0:
        return None
    delta = np.sqrt(disc)
    return (-phi - delta) / sig2


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised helpers
# ──────────────────────────────────────────────────────────────────────────────

def psi_vec(tau_arr, x: float, phi: float, sig2: float) -> np.ndarray:
    """Ψ evaluated at each element of tau_arr."""
    return np.array([psi(float(t), x, phi, sig2) for t in tau_arr])


def phi_integral_vec(tau_arr, x: float, phi: float, sig2: float) -> np.ndarray:
    """Φ evaluated at each element of tau_arr."""
    return np.array([phi_integral(float(t), x, phi, sig2) for t in tau_arr])


# ──────────────────────────────────────────────────────────────────────────────
# Truly vectorised helpers — O(n) numpy ops over a tau array, scalar x/φ/σ²
# ──────────────────────────────────────────────────────────────────────────────

def psi_arr(tau_arr: np.ndarray, x: float, phi: float, sig2: float) -> np.ndarray:
    """Ψ(τ; x) evaluated over an ndarray of τ values with a single branch.

    Much faster than psi_vec for large arrays because it uses numpy broadcasting
    instead of a Python loop.  The branch (exponential / trigonometric /
    degenerate) is determined once by the scalar discriminant.
    """
    tau_arr = np.asarray(tau_arr, dtype=float)
    disc    = discriminant(x, phi, sig2)

    if disc > 0.0:
        delta = np.sqrt(disc)
        u     = delta * tau_arr
        # 1 − exp(−u):  expm1 is numerically exact for small u; no overflow
        frac  = -np.expm1(-u)
        d     = (delta - phi) + (phi + delta) * np.exp(-u)
        return 2.0 * x * frac / d

    if disc < 0.0:
        delta_t = np.sqrt(-disc)
        a0      = np.arctan(phi / delta_t)
        ang     = 0.5 * delta_t * tau_arr + a0
        return (delta_t * np.tan(ang) - phi) / sig2

    # Degenerate: fall back to scalar loop (rare)
    return psi_vec(tau_arr, x, phi, sig2)


def phi_integral_arr(tau_arr: np.ndarray, x: float, phi: float, sig2: float) -> np.ndarray:
    """Φ(τ; x) = ∫₀^τ Ψ(s; x) ds evaluated over an ndarray of τ values."""
    tau_arr = np.asarray(tau_arr, dtype=float)
    disc    = discriminant(x, phi, sig2)

    if disc > 0.0:
        delta    = np.sqrt(disc)
        u        = delta * tau_arr
        d        = (delta - phi) + (phi + delta) * np.exp(-u)
        log_term = (delta - phi) * tau_arr + 2.0 * (np.log(2.0 * delta) - np.log(d) - u)
        return log_term / sig2

    if disc < 0.0:
        delta_t  = np.sqrt(-disc)
        a0       = np.arctan(phi / delta_t)
        ang      = 0.5 * delta_t * tau_arr + a0
        log_term = (-phi) * tau_arr - 2.0 * np.log(np.cos(ang) / np.cos(a0))
        return log_term / sig2

    # Degenerate: fall back to scalar loop (rare)
    return phi_integral_vec(tau_arr, x, phi, sig2)
