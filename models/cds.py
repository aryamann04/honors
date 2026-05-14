"""CDS pricing under the reduced model (v = 0).

Fair spread:
    s_CDS(τ) = (1−R) ∫₀^τ K(t, t+u) du
               ────────────────────────────────────────────────────────────────
               Σ_j α_j G(t, t_j) + Σ_j ∫_{t_{j-1}}^{t_j} (u−t_{j-1}) K(t,u) du

where
    G(t, t+u) = exp(a_G(u) + b_{G,f}(u) λ^f + b_{G,g}(u) λ^g)
    K(t, t+u) = [c_0(u) + c_f(u) λ^f + c_g(u) λ^g] · G(t, t+u)

The c_f / c_g coefficients use the Bug-2-fixed cds_c_coeff (Riccati forcing = −η_i).
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from core.parameters import DisasterModelParams
from core.closed_form import cds_G_coeffs_cf, cds_K_coeffs_cf, _phi_sig2, K_const, _cds_Ahat
from core.riccati import psi_arr, phi_integral_arr


# ──────────────────────────────────────────────────────────────────────────────
# G  and  K  point evaluations
# ──────────────────────────────────────────────────────────────────────────────

def G_value(params: DisasterModelParams, tau: float,
            lam_f: float, lam_g: float) -> float:
    """G(t, t+τ) = discounted survival probability."""
    a_G, b_Gf, b_Gg = cds_G_coeffs_cf(params, tau)
    return np.exp(a_G + b_Gf * lam_f + b_Gg * lam_g)


def K_value(params: DisasterModelParams, tau: float,
            lam_f: float, lam_g: float) -> float:
    """K(t, t+τ) = [c_0 + c_f λ^f + c_g λ^g] · G(t, t+τ)."""
    a_G, b_Gf, b_Gg = cds_G_coeffs_cf(params, tau)
    c0, cf, cg = cds_K_coeffs_cf(params, tau)
    poly = c0 + cf * lam_f + cg * lam_g
    return poly * np.exp(a_G + b_Gf * lam_f + b_Gg * lam_g)


# ──────────────────────────────────────────────────────────────────────────────
# Fair CDS spread
# ──────────────────────────────────────────────────────────────────────────────

def fair_cds_spread(params: DisasterModelParams, maturity: float,
                    lam_f: float, lam_g: float,
                    payment_interval: float = 0.25,
                    n_int: int = 3000) -> float:
    """Fair CDS spread (annualised decimal) for a given maturity."""
    T    = float(maturity)
    grid = np.linspace(0.0, T, n_int)
    K_g  = np.array([K_value(params, u, lam_f, lam_g) for u in grid])

    protection = (1.0 - params.R) * np.trapz(K_g, grid)

    payment_dates = np.arange(payment_interval, T + 1e-12, payment_interval)
    if len(payment_dates) == 0 or payment_dates[-1] < T - 1e-12:
        payment_dates = np.append(payment_dates, T)

    premium = 0.0
    t_prev  = 0.0
    for t_j in payment_dates:
        alpha_j  = t_j - t_prev
        premium += alpha_j * G_value(params, t_j, lam_f, lam_g)
        idx = (grid >= t_prev) & (grid <= t_j)
        if idx.sum() >= 2:
            premium += np.trapz((grid[idx] - t_prev) * K_g[idx], grid[idx])
        t_prev = t_j

    return protection / premium


def cds_spread_curve(params: DisasterModelParams, tau_grid: np.ndarray,
                     lam_f: float, lam_g: float,
                     payment_interval: float = 0.25) -> np.ndarray:
    """Fair CDS spread evaluated over a maturity grid."""
    return np.array([fair_cds_spread(params, float(t), lam_f, lam_g, payment_interval)
                     for t in tau_grid])


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised batch pricer
# ──────────────────────────────────────────────────────────────────────────────

def precompute_cds_coefficients(
    params: DisasterModelParams,
    u_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute (a_G, b_Gf, b_Gg, c0, cf, cg) on *u_grid* in O(n) vectorised ops.

    Uses psi_arr / phi_integral_arr so the loop over grid points is replaced by
    numpy broadcasting.  The c0 coefficient is computed via a single cumulative
    trapezoid integral rather than one integral per grid point.

    Returns six arrays of shape ``(len(u_grid),)``.
    """
    params.compute_b_sdf()
    phi_p, sig2 = _phi_sig2(params)   # φ = b_sdf·σ² − κ,  σ² = σ_λ²

    K = K_const(params)
    A0_hat, Af_hat, Ag_hat = _cds_Ahat(params)

    x_f = -Af_hat   # Riccati forcing for G-coefficient (f-factor)
    x_g = -Ag_hat

    # ── G coefficients ─────────────────────────────────────────────────────────
    b_Gf  = psi_arr(u_grid, x_f, phi_p, sig2)
    b_Gg  = psi_arr(u_grid, x_g, phi_p, sig2)
    Phi_f = phi_integral_arr(u_grid, x_f, phi_p, sig2)
    Phi_g = phi_integral_arr(u_grid, x_g, phi_p, sig2)
    a_G   = (-A0_hat * u_grid
             + params.kappa * params.lam_bar_f * Phi_f
             + params.kappa * params.lam_bar_g * Phi_g)

    # ── c coefficients ─────────────────────────────────────────────────────────
    # cf(u) = η₁ · exp(φu + σ² Φ(u; −η₁))      (Bug-2-fixed Riccati forcing = −η_i)
    Phi_cf = phi_integral_arr(u_grid, -params.eta1, phi_p, sig2)
    Phi_cg = phi_integral_arr(u_grid, -params.eta2, phi_p, sig2)
    cf     = params.eta1 * np.exp(phi_p * u_grid + sig2 * Phi_cf)
    cg     = params.eta2 * np.exp(phi_p * u_grid + sig2 * Phi_cg)

    # c0(τ) = h0* + κ ∫₀^τ [λ̄^f cf(s) + λ̄^g cg(s)] ds  — one cumtrapz, O(n)
    integrand = params.lam_bar_f * cf + params.lam_bar_g * cg
    c0        = params.h0_star + params.kappa * cumulative_trapezoid(
        integrand, u_grid, initial=0.0
    )

    return a_G, b_Gf, b_Gg, c0, cf, cg


def cds_spread_batch(
    params:           DisasterModelParams,
    tenors:           np.ndarray,
    lam_f_arr:        np.ndarray,
    lam_g_arr:        np.ndarray,
    payment_interval: float = 0.25,
    n_int:            int   = 200,
) -> np.ndarray:
    """Vectorised fair CDS spread for multiple (λ^f, λ^g) pairs and multiple tenors.

    Evaluates all (date × tenor) combinations in a single pass by:
      1. Precomputing affine coefficients on a shared u-grid (O(n_int) numpy ops).
      2. Broadcasting G and K over all lambda pairs simultaneously.
      3. Integrating with numpy trapz — no Python loops over dates.

    Parameters
    ----------
    params    : DisasterModelParams — h0_star, eta1, eta2, lam_bar_f, lam_bar_g, R must be set.
    tenors    : shape (T,) — maturities in years.
    lam_f_arr : shape (M,) — country-specific intensity per observation.
    lam_g_arr : shape (M,) — global intensity per observation.
    n_int     : base integration points (payment dates are always included exactly).

    Returns
    -------
    spreads : ndarray of shape (M, T)
    """
    tenors    = np.asarray(tenors,    dtype=float)
    lam_f_arr = np.asarray(lam_f_arr, dtype=float)
    lam_g_arr = np.asarray(lam_g_arr, dtype=float)
    M         = len(lam_f_arr)
    T_max     = float(tenors.max())

    # Build u-grid that includes every payment date exactly (exact G at coupon times)
    payment_pts: list[float] = [0.0]
    for T in tenors:
        pmts = list(np.arange(payment_interval, T + 1e-12, payment_interval))
        if not pmts or pmts[-1] < T - 1e-12:
            pmts.append(float(T))
        payment_pts.extend(pmts)
    base_grid = np.linspace(0.0, T_max, n_int)
    u_grid    = np.unique(np.concatenate([base_grid, payment_pts]))
    n         = len(u_grid)

    # ── Precompute affine coefficients ────────────────────────────────────────
    a_G, b_Gf, b_Gg, c0, cf, cg = precompute_cds_coefficients(params, u_grid)

    # ── G(u, λ) and K(u, λ) for all lambda pairs: shape (n, M) ──────────────
    log_G = a_G[:, None] + np.outer(b_Gf, lam_f_arr) + np.outer(b_Gg, lam_g_arr)
    G_mat = np.exp(log_G)
    K_mat = (c0[:, None] + np.outer(cf, lam_f_arr) + np.outer(cg, lam_g_arr)) * G_mat

    R       = params.R
    spreads = np.empty((M, len(tenors)))

    for j, T in enumerate(tenors):
        mask  = u_grid <= T + 1e-12
        u_sub = u_grid[mask]
        K_sub = K_mat[mask, :]       # (n_sub, M)

        # Protection leg: (1−R) ∫ K du
        protection = (1.0 - R) * np.trapz(K_sub, u_sub, axis=0)   # (M,)

        # Premium leg
        pmts = list(np.arange(payment_interval, T + 1e-12, payment_interval))
        if not pmts or pmts[-1] < T - 1e-12:
            pmts.append(float(T))

        premium = np.zeros(M)
        t_prev  = 0.0
        for t_j in pmts:
            alpha_j = t_j - t_prev

            # G at t_j — t_j is in u_grid by construction, so searchsorted is exact
            idx_j   = int(np.searchsorted(u_grid, t_j))
            idx_j   = min(idx_j, n - 1)
            premium += alpha_j * G_mat[idx_j, :]          # (M,)

            # Accrued-interest integral over (t_prev, t_j)
            int_mask = (u_sub >= t_prev - 1e-12) & (u_sub <= t_j + 1e-12)
            if int_mask.sum() >= 2:
                u_int = u_sub[int_mask]
                intgd = (u_int[:, None] - t_prev) * K_sub[int_mask, :]
                premium += np.trapz(intgd, u_int, axis=0)  # (M,)

            t_prev = t_j

        spreads[:, j] = protection / np.maximum(premium, 1e-30)

    return spreads
