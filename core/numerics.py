"""Numerical blow-up detection for affine bond-price loadings.

Ported and cleaned from draft4/ode_solver.py.  The utility estimates the
finite maturity τ* at which a bond-price loading b(τ) diverges to +∞.
"""
import numpy as np
from scipy.integrate import solve_ivp
from core.parameters import DisasterModelParams


def _hit_time_to_level(F, b0: float, level: float, tau_max: float):
    """Integrate dy/dτ = F(y) until y hits `level`.  Returns (τ_hit, did_hit)."""
    def event(t, y):
        return y[0] - level
    event.terminal = True
    event.direction = 1.0

    sol = solve_ivp(
        lambda t, y: [F(y[0])],
        (0.0, tau_max), y0=[b0],
        method="Radau", rtol=1e-10, atol=1e-13, max_step=0.05,
        events=event,
    )
    if sol.status == 1 and sol.t_events and len(sol.t_events[0]) > 0:
        return float(sol.t_events[0][0]), True
    return float(tau_max), False


def _integral_remaining_time(F, b_start: float, b_end: float, n: int = 200_000) -> float:
    """Estimate ∫_{b_start}^{b_end} db/F(b) (time to travel from b_start to b_end)."""
    grid = np.linspace(b_start, b_end, int(n))
    vals = np.array([F(b) for b in grid], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("F(b) not finite on integration grid.")
    if np.min(vals) <= 0.0:
        raise RuntimeError("F(b) not strictly positive on integration grid.")
    return float(np.trapz(1.0 / vals, grid))


def _estimate_blowup_time(F, b0: float = 0.0, tau_max: float = 500.0,
                          b_danger: float = 200.0, b_big: float = 4000.0):
    """Return [τ_lo, τ_hi] bracket for blow-up time, or None if no blow-up detected."""
    tau_hit, hit = _hit_time_to_level(F, b0=b0, level=b_danger, tau_max=tau_max)
    if not hit:
        return None
    F_d = float(F(b_danger))
    if not np.isfinite(F_d) or F_d <= 0.0:
        return None
    try:
        dt = _integral_remaining_time(F, b_danger, b_big)
    except RuntimeError:
        return None
    return tau_hit, tau_hit + dt


# ──────────────────────────────────────────────────────────────────────────────
# Bond-specific wrappers
# ──────────────────────────────────────────────────────────────────────────────

def blowup_riskfree(params: DisasterModelParams, tau_max: float = 500.0):
    """Blow-up bracket for the foreign risk-free bond loading b*(τ)."""
    params.compute_b_sdf()
    b_bar = params.b_sdf
    C = np.exp(b_bar * params.v - params.gamma * params.Z)

    def F(b):
        return ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b
                + 0.5 * params.sigma_lambda**2 * b**2
                + C * (np.exp(b * params.v) - np.exp(params.Z)))

    return _estimate_blowup_time(F, tau_max=tau_max)


def blowup_defaultable(params: DisasterModelParams, tau_max: float = 500.0):
    """Blow-up bracket for the defaultable bond loading (earliest of f, g)."""
    params.compute_b_sdf()
    b_bar = params.b_sdf
    C  = np.exp(b_bar * params.v - params.gamma * params.Z)
    Af = (1.0 - params.R) * params.eta1
    Ag = (1.0 - params.R) * params.eta2

    def F_f(b):
        return ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b
                + 0.5 * params.sigma_lambda**2 * b**2
                + C * (np.exp(b * params.v) - np.exp(params.Z)) - Af)

    def F_g(b):
        return ((b_bar * params.sigma_lambda**2 - params.kappa - params.v) * b
                + 0.5 * params.sigma_lambda**2 * b**2
                + C * (np.exp(b * params.v) - np.exp(params.Z)) - Ag)

    est_f = _estimate_blowup_time(F_f, tau_max=tau_max)
    est_g = _estimate_blowup_time(F_g, tau_max=tau_max)
    candidates = [x for x in [est_f, est_g] if x is not None]
    if not candidates:
        return None
    return min(c[0] for c in candidates), min(c[1] for c in candidates)


def print_blowup_estimates(params: DisasterModelParams, tau_max: float = 500.0):
    rf   = blowup_riskfree(params, tau_max)
    deflt = blowup_defaultable(params, tau_max)

    def _fmt(x):
        if x is None:
            return f"> {tau_max:.1f}y (no blow-up detected)"
        return f"[{x[0]:.4f}, {x[1]:.4f}] years"

    print("Estimated blow-up maturity brackets:")
    print(f"  b*(τ)        foreign risk-free:   {_fmt(rf)}")
    print(f"  b_D*(τ)      foreign defaultable: {_fmt(deflt)}")
