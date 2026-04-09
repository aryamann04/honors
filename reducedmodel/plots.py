import os
import sys
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from params.modelparams import DisasterModelParams

FIGDIR = os.path.join(BASE_DIR, "figures_v0_closed_form")


# =========================
# Closed-form helper blocks
# =========================

def stable_delta(phi, delta, tau):
    exp_neg = np.exp(-delta * tau)
    return (delta - phi) + (phi + delta) * exp_neg


def stable_log_term(phi, delta, tau):
    d = stable_delta(phi, delta, tau)
    return (delta - phi) * tau + 2.0 * (np.log(2.0 * delta) - np.log(d) - delta * tau)


def stable_frac_exp(delta, tau):
    u = delta * tau
    if u > 50.0:
        return 1.0
    return (np.exp(u) - 1.0) / np.exp(u)


def stable_delta_plusminus(delta, kappa, tau):
    exp_neg = np.exp(-delta * tau)
    return (delta + kappa) + (delta - kappa) * exp_neg


def stable_log_term_plus(delta, kappa, tau):
    d = stable_delta_plusminus(delta, kappa, tau)
    return (delta + kappa) * tau + 2.0 * (np.log(2.0 * delta) - np.log(d) - delta * tau)


def stable_frac_exp_plus(delta, tau):
    u = delta * tau
    if u > 50.0:
        return 1.0
    return (np.exp(u) - 1.0) / np.exp(u)


def trig_b(phi, sig2, delta_trig, tau):
    a0 = np.arctan(phi / delta_trig)
    ang = 0.5 * delta_trig * tau + a0
    return (delta_trig * np.tan(ang) - phi) / sig2


def trig_log_term(phi, delta_trig, tau):
    a0 = np.arctan(phi / delta_trig)
    ang = 0.5 * delta_trig * tau + a0
    c0 = np.cos(a0)
    c1 = np.cos(ang)
    return (-phi) * tau - 2.0 * np.log(c1 / c0)


def delta_base_sq(params, b):
    sig2 = params.sigma_lambda ** 2
    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
    phi = b * sig2 - params.kappa
    return phi * phi - 2.0 * sig2 * K0


def delta_i_sq(params, b, A_i):
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa
    return phi * phi - 2.0 * sig2 * (-A_i)


def classify_branch(delta_sq):
    if delta_sq > 0.0:
        return "exponential"
    if delta_sq < 0.0:
        return "trigonometric"
    return "degenerate"


# =========================
# Parameter checks
# =========================

def check_parameter_admissibility(params, tau_probe=100.0):
    beta = float(params.beta)
    kappa = float(params.kappa)
    sigl = float(params.sigma_lambda)
    sig2 = sigl ** 2
    gamma = float(params.gamma)
    Z = float(params.Z)
    R = float(params.R)
    eta_f = float(params.eta1)
    eta_g = float(params.eta2)

    failures = []

    def req(cond, msg):
        if not cond:
            failures.append(msg)

    req(np.isfinite(beta) and beta > 0.0, f"beta must be > 0. Got {beta}")
    req(np.isfinite(kappa) and kappa > 0.0, f"kappa must be > 0. Got {kappa}")
    req(np.isfinite(sigl) and sigl > 0.0, f"sigma_lambda must be > 0. Got {sigl}")
    req(np.isfinite(gamma) and gamma > 0.0, f"gamma must be > 0. Got {gamma}")
    req(np.isfinite(Z) and Z < 0.0, f"Z must be < 0. Got {Z}")
    req(np.isfinite(R) and 0.0 <= R <= 1.0, f"R must be in [0,1]. Got {R}")
    req(np.isfinite(eta_f) and eta_f >= 0.0, f"eta1 must be >= 0. Got {eta_f}")
    req(np.isfinite(eta_g) and eta_g >= 0.0, f"eta2 must be >= 0. Got {eta_g}")

    D0 = np.exp((1.0 - gamma) * Z) - 1.0
    disc_b = (beta + kappa) ** 2 - 2.0 * sig2 * D0
    req(np.isfinite(disc_b) and disc_b > 0.0, f"b discriminant must be > 0. Got {disc_b}")

    params.compute_b_sdf()
    b = getattr(params, "b_sdf", None)
    req(b is not None and np.isfinite(b), f"b_sdf must be finite. Got {b}")

    if failures:
        raise ValueError("Model parameter restrictions violated: " + " | ".join(failures))

    b = float(b)
    phi = b * sig2 - kappa
    req(np.isfinite(phi), f"phi must be finite. Got {phi}")
    req(phi < 0.0, f"phi=b*sigma_lambda^2-kappa must be < 0. Got phi={phi}")

    K0 = np.exp(-gamma * Z) * (1.0 - np.exp(Z))
    req(np.isfinite(K0) and K0 > 0.0, f"K0 must be finite and > 0. Got {K0}")

    Ah = np.exp(-gamma * Z) * (np.exp(Z) - 1.0)
    Ag_def = Ah + (1.0 - R) * eta_g
    Af_def = Ah + (1.0 - R) * eta_f
    Af_quanto = (1.0 - R) * eta_f

    disc_base = phi * phi - 2.0 * sig2 * K0
    disc_h = phi * phi - 2.0 * sig2 * (-Ah)
    disc_g = phi * phi - 2.0 * sig2 * (-Ag_def)
    disc_f = phi * phi - 2.0 * sig2 * (-Af_def)

    disc_qf = kappa * kappa + 2.0 * sig2 * Af_quanto
    req(np.isfinite(disc_qf) and disc_qf > 0.0, f"tilde_delta_f^2 must be > 0. Got {disc_qf}")

    branches = {
        "base (RF)": classify_branch(disc_base),
        "quanto h": classify_branch(disc_h),
        "def g": classify_branch(disc_g),
        "def f": classify_branch(disc_f),
        "quanto f": "exponential",
    }

    print("\nBranch requirements from parameters:")
    for k, v in branches.items():
        print(f" - {k}: {v}")

    def denom_checks(label, delta2):
        if not (np.isfinite(delta2) and delta2 > 0.0):
            return
        delta = float(np.sqrt(delta2))
        d0 = stable_delta(phi, delta, 0.0)
        dT = stable_delta(phi, delta, float(tau_probe))
        req(np.isfinite(d0) and d0 > 0.0, f"{label} denom must be >0 at tau=0. Got {d0}")
        req(np.isfinite(dT) and dT > 0.0, f"{label} denom must be >0 at tau={tau_probe}. Got {dT}")

    denom_checks("base", disc_base)
    denom_checks("h", disc_h)
    denom_checks("g", disc_g)
    denom_checks("f", disc_f)

    if failures:
        raise ValueError("Model parameter restrictions violated: " + " | ".join(failures))


# =========================
# Reduced-model bond pricing
# =========================

def a_star(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2)
    lam_bar = params.lam_bar_g + params.lam_bar_f

    disc = delta_base_sq(params, b)
    if disc > 0.0:
        delta = float(np.sqrt(disc))
        term = stable_log_term(phi, delta, tau)
    elif disc < 0.0:
        delta_trig = float(np.sqrt(-disc))
        term = trig_log_term(phi, delta_trig, tau)
    else:
        term = (-phi) * tau

    return -A0 * tau + (params.kappa * lam_bar / sig2) * term


def b_star(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
    disc = delta_base_sq(params, b)

    if disc > 0.0:
        delta = float(np.sqrt(disc))
        d = stable_delta(phi, delta, tau)
        frac = stable_frac_exp(delta, tau)
        num = 2.0 * K0 * frac
        return num / d

    if disc < 0.0:
        delta_trig = float(np.sqrt(-disc))
        return trig_b(phi, sig2, delta_trig, tau) * sig2

    return (K0 * tau) / (1.0 - phi * tau)


def a_dom(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2)
    lam_bar = params.lam_bar_g + params.lam_bar_h

    disc = delta_base_sq(params, b)
    if disc > 0.0:
        delta = float(np.sqrt(disc))
        term = stable_log_term(phi, delta, tau)
    elif disc < 0.0:
        delta_trig = float(np.sqrt(-disc))
        term = trig_log_term(phi, delta_trig, tau)
    else:
        term = (-phi) * tau

    return -A0 * tau + (params.kappa * lam_bar / sig2) * term


def generic_foreign_affine_coeffs(params, tau, A0, Af, Ag):
    """Closed-form affine coefficients for reduced-model foreign-currency claims.

    This covers the foreign defaultable bond and the G(t,u) object used in CDS pricing.
    """
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa
    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))

    disc_f = delta_i_sq(params, b, Af)
    disc_g = delta_i_sq(params, b, Ag)

    if disc_f > 0.0:
        delta_f = float(np.sqrt(disc_f))
        df = stable_delta(phi, delta_f, tau)
        frac_f = stable_frac_exp(delta_f, tau)
        bf = -2.0 * Af * frac_f / df
        term_f = stable_log_term(phi, delta_f, tau)
    else:
        delta_trig = float(np.sqrt(-disc_f))
        bf = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_f = trig_log_term(phi, delta_trig, tau)

    if disc_g > 0.0:
        delta_g = float(np.sqrt(disc_g))
        dg = stable_delta(phi, delta_g, tau)
        frac_g = stable_frac_exp(delta_g, tau)
        bg = -2.0 * Ag * frac_g / dg
        term_g = stable_log_term(phi, delta_g, tau)
    else:
        delta_trig = float(np.sqrt(-disc_g))
        bg = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_g = trig_log_term(phi, delta_trig, tau)

    a = -A0 * tau
    a += (params.kappa * params.lam_bar_f / sig2) * term_f
    a += (params.kappa * params.lam_bar_g / sig2) * term_g
    return a, bf, bg


def defaultable_foreign_coeffs(params, tau):
    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2) + (1.0 - params.R) * params.h0_star
    Af = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0) + (1.0 - params.R) * params.eta1
    Ag = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0) + (1.0 - params.R) * params.eta2
    return generic_foreign_affine_coeffs(params, tau, A0=A0, Af=Af, Ag=Ag)


# NOTE: kept only to preserve the old quanto functionality.
# It is not used anywhere in the new plotting workflow because the thesis section is now focused on non-quanto CDS and bond results.

def quanto_coeffs(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2) + (1.0 - params.R) * params.h0_star
    Ah = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0)
    Ag = Ah + (1.0 - params.R) * params.eta2
    Af = (1.0 - params.R) * params.eta1

    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))

    disc_h = delta_i_sq(params, b, Ah)
    disc_g = delta_i_sq(params, b, Ag)

    if disc_h > 0.0:
        delta_h = float(np.sqrt(disc_h))
        dh = stable_delta(phi, delta_h, tau)
        frac_h = stable_frac_exp(delta_h, tau)
        bh = -2.0 * Ah * frac_h / dh
        term_h = stable_log_term(phi, delta_h, tau)
    else:
        delta_trig = float(np.sqrt(-disc_h))
        bh = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_h = trig_log_term(phi, delta_trig, tau)

    if disc_g > 0.0:
        delta_g = float(np.sqrt(disc_g))
        dg = stable_delta(phi, delta_g, tau)
        frac_g = stable_frac_exp(delta_g, tau)
        bg = -2.0 * Ag * frac_g / dg
        term_g = stable_log_term(phi, delta_g, tau)
    else:
        delta_trig = float(np.sqrt(-disc_g))
        bg = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_g = trig_log_term(phi, delta_trig, tau)

    delta_f = float(np.sqrt(params.kappa ** 2 + 2.0 * sig2 * Af))
    df = stable_delta_plusminus(delta_f, params.kappa, tau)
    frac_f = stable_frac_exp_plus(delta_f, tau)
    bf = -2.0 * (1.0 - params.R) * params.eta1 * frac_f / df
    term_f = stable_log_term_plus(delta_f, params.kappa, tau)

    a = -A0 * tau
    a += (params.kappa * params.lam_bar_h / sig2) * term_h
    a += (params.kappa * params.lam_bar_g / sig2) * term_g
    a += (params.kappa * params.lam_bar_f / sig2) * term_f

    return a, bh, bg, bf


# =========================
# Basic bond/yield/spread objects
# =========================

def state_from_params(params):
    return params.lam_bar_h, params.lam_bar_g, params.lam_bar_f


def riskfree_foreign_log_price(params, tau, lam_f, lam_g):
    return a_star(params, tau) + b_star(params, tau) * (lam_f + lam_g)


def riskfree_domestic_log_price(params, tau, lam_h, lam_g):
    return a_dom(params, tau) + b_star(params, tau) * (lam_h + lam_g)


def defaultable_foreign_log_price(params, tau, lam_f, lam_g):
    a, bf, bg = defaultable_foreign_coeffs(params, tau)
    return a + bf * lam_f + bg * lam_g


def foreign_riskfree_yield(params, tau, lam_f, lam_g):
    return -riskfree_foreign_log_price(params, tau, lam_f, lam_g) / tau


def foreign_defaultable_yield(params, tau, lam_f, lam_g):
    return -defaultable_foreign_log_price(params, tau, lam_f, lam_g) / tau


def foreign_credit_spread(params, tau, lam_f, lam_g):
    return foreign_defaultable_yield(params, tau, lam_f, lam_g) - foreign_riskfree_yield(params, tau, lam_f, lam_g)


def bond_prices(params, tau_grid, lam_h=None, lam_g=None, lam_f=None):
    if lam_h is None or lam_g is None or lam_f is None:
        lam_h, lam_g, lam_f = state_from_params(params)

    B_star_vals = []
    B_dom_vals = []
    B_D_vals = []
    B_q_vals = []

    for tau in tau_grid:
        B_star_vals.append(np.exp(riskfree_foreign_log_price(params, tau, lam_f, lam_g)))
        B_dom_vals.append(np.exp(riskfree_domestic_log_price(params, tau, lam_h, lam_g)))
        B_D_vals.append(np.exp(defaultable_foreign_log_price(params, tau, lam_f, lam_g)))

        a4, bh, bgq, bfq = quanto_coeffs(params, tau)
        B_q_vals.append(np.exp(a4 + bh * lam_h + bgq * lam_g + bfq * lam_f))

    return np.array(B_star_vals), np.array(B_dom_vals), np.array(B_D_vals), np.array(B_q_vals)


def yields_and_spreads(params, tau_grid, lam_h=None, lam_g=None, lam_f=None):
    if lam_h is None or lam_g is None or lam_f is None:
        lam_h, lam_g, lam_f = state_from_params(params)

    y_star = np.empty_like(tau_grid, dtype=float)
    y_dom = np.empty_like(tau_grid, dtype=float)
    y_D = np.empty_like(tau_grid, dtype=float)
    y_q = np.empty_like(tau_grid, dtype=float)
    s_star = np.empty_like(tau_grid, dtype=float)
    s_q = np.empty_like(tau_grid, dtype=float)
    q_basis = np.empty_like(tau_grid, dtype=float)

    for i, tau in enumerate(tau_grid):
        y_star[i] = foreign_riskfree_yield(params, tau, lam_f, lam_g)
        y_dom[i] = -riskfree_domestic_log_price(params, tau, lam_h, lam_g) / tau
        y_D[i] = foreign_defaultable_yield(params, tau, lam_f, lam_g)

        a4, bh, bgq, bfq = quanto_coeffs(params, tau)
        logB_q = a4 + bh * lam_h + bgq * lam_g + bfq * lam_f
        y_q[i] = -logB_q / tau

        s_star[i] = y_D[i] - y_star[i]
        s_q[i] = y_q[i] - y_dom[i]
        q_basis[i] = y_q[i] - y_D[i]

    return y_star, y_dom, y_D, y_q, s_star, s_q, q_basis


# =========================
# CDS pricing under the reduced model (v = 0)
# =========================

def cds_hat_constants(params):
    A0_hat = params.beta + params.mu - params.gamma * (params.sigma_c ** 2) + params.h0_star
    Af_hat = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0) + params.eta1
    Ag_hat = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0) + params.eta2
    return A0_hat, Af_hat, Ag_hat


def cds_G_coeffs(params, tau):
    A0_hat, Af_hat, Ag_hat = cds_hat_constants(params)
    return generic_foreign_affine_coeffs(params, tau, A0=A0_hat, Af=Af_hat, Ag=Ag_hat)


def _phi_value(params):
    params.compute_b_sdf()
    return float(params.b_sdf) * (params.sigma_lambda ** 2) - params.kappa


def _cf_or_cg_reduced(params, tau, eta_i, Ahat_i):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa
    disc_i = delta_i_sq(params, b, Ahat_i)

    if disc_i > 0.0:
        delta_i = float(np.sqrt(disc_i))
        integ_b = stable_log_term(phi, delta_i, tau) / sig2
    elif disc_i < 0.0:
        delta_trig = float(np.sqrt(-disc_i))
        integ_b = trig_log_term(phi, delta_trig, tau) / sig2
    else:
        K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
        x = -Ahat_i
        if abs(x) < 1e-14:
            integ_b = 0.0
        else:
            b_tau = (x * tau) / (1.0 - phi * tau)
            grid = np.linspace(0.0, tau, 1001)
            vals = (x * grid) / (1.0 - phi * grid)
            integ_b = np.trapz(vals, grid)

    return eta_i * np.exp((phi) * tau + sig2 * integ_b)


def cds_K_coeffs(params, tau, n_int=400):
    """Reduced-model K(t,u) coefficients from the LaTeX write-up.

    K(t,u) = [c0(u) + cf(u) lambda_f + cg(u) lambda_g] * exp(a_G(u) + b_Gf(u) lambda_f + b_Gg(u) lambda_g)
    """
    _, Af_hat, Ag_hat = cds_hat_constants(params)

    cf_tau = _cf_or_cg_reduced(params, tau, params.eta1, Af_hat)
    cg_tau = _cf_or_cg_reduced(params, tau, params.eta2, Ag_hat)

    if tau == 0.0:
        c0_tau = params.h0_star
    else:
        grid = np.linspace(0.0, tau, n_int)
        cf_grid = np.array([_cf_or_cg_reduced(params, s, params.eta1, Af_hat) for s in grid])
        cg_grid = np.array([_cf_or_cg_reduced(params, s, params.eta2, Ag_hat) for s in grid])
        integrand = params.lam_bar_f * cf_grid + params.lam_bar_g * cg_grid
        c0_tau = params.h0_star + params.kappa * np.trapz(integrand, grid)

    return c0_tau, cf_tau, cg_tau


def cds_G_value(params, tau, lam_f, lam_g):
    aG, bGf, bGg = cds_G_coeffs(params, tau)
    return np.exp(aG + bGf * lam_f + bGg * lam_g)


def cds_K_value(params, tau, lam_f, lam_g):
    aG, bGf, bGg = cds_G_coeffs(params, tau)
    c0, cf, cg = cds_K_coeffs(params, tau)
    poly = c0 + cf * lam_f + cg * lam_g
    return poly * np.exp(aG + bGf * lam_f + bGg * lam_g)


def fair_cds_spread(params, maturity, lam_f, lam_g, payment_interval=0.25, n_int=3000):
    """Fair CDS spread in annualized decimal units.

    Matches the reduced-model formula in the LaTeX write-up:
        s_CDS = (1-R) * int_0^T K / [sum_j alpha_j G(t,t_j) + sum_j int_{t_{j-1}}^{t_j} (u-t_{j-1}) K(t,t+u) du]
    """
    T = float(maturity)
    if T <= 0.0:
        raise ValueError("maturity must be positive")

    grid = np.linspace(0.0, T, n_int)
    K_grid = np.array([cds_K_value(params, u, lam_f, lam_g) for u in grid])
    protection_leg = (1.0 - params.R) * np.trapz(K_grid, grid)

    payment_dates = np.arange(payment_interval, T + 1e-12, payment_interval)
    if len(payment_dates) == 0 or payment_dates[-1] < T - 1e-12:
        payment_dates = np.append(payment_dates, T)

    premium_leg = 0.0
    t_prev = 0.0
    for t_j in payment_dates:
        alpha_j = t_j - t_prev
        premium_leg += alpha_j * cds_G_value(params, t_j, lam_f, lam_g)

        idx = (grid >= t_prev) & (grid <= t_j)
        u_local = grid[idx]
        K_local = K_grid[idx]
        if u_local.size >= 2:
            premium_leg += np.trapz((u_local - t_prev) * K_local, u_local)

        t_prev = t_j

    return protection_leg / premium_leg


def cds_term_structure(params, tau_grid, lam_f, lam_g, payment_interval=0.25):
    return np.array([
        fair_cds_spread(params, tau, lam_f, lam_g, payment_interval=payment_interval)
        for tau in tau_grid
    ])


# =========================
# Plotting helpers
# =========================

def _savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, name), dpi=220)
    plt.close()


def plot_nonquanto_bond_prices(params, tau_grid, lam_h, lam_g, lam_f):
    B_star_vals, B_dom_vals, B_D_vals, _ = bond_prices(params, tau_grid, lam_h, lam_g, lam_f)

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(tau_grid, B_star_vals, linewidth=2.4, label="Risk-free foreign bond")
    plt.plot(tau_grid, B_dom_vals, linewidth=2.4, label="Risk-free domestic bond")
    plt.plot(tau_grid, B_D_vals, linewidth=2.4, linestyle="--", label="Defaultable foreign bond")
    plt.ylim(0.55, 1.02)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Price")
    plt.grid(alpha=0.30)
    plt.legend()
    _savefig("bond_prices_nonquanto.png")


def plot_yield_term_structures(params, tau_grid, lam_h, lam_g, lam_f):
    y_star, _, y_D, _, s_star, _, _ = yields_and_spreads(params, tau_grid, lam_h, lam_g, lam_f)

    plt.figure(figsize=(8.8, 5.2))
    plt.plot(tau_grid, 100.0 * y_star, linewidth=2.4, label=r"$y^*(\tau)$ risk-free")
    plt.plot(tau_grid, 100.0 * y_D, linewidth=2.4, linestyle="--", label=r"$y_D^*(\tau)$ defaultable")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Yield (%)")
    plt.grid(alpha=0.28)
    plt.legend()
    _savefig("yield_term_structures_nonquanto.png")

    plt.figure(figsize=(8.8, 5.2))
    plt.plot(tau_grid, 1e4 * s_star, linewidth=2.4, label=r"Bond spread $s^*(\tau)$")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spread (bp)")
    plt.grid(alpha=0.28)
    plt.legend()
    _savefig("bond_credit_spread_term_structure.png")


def plot_bond_spread_vs_cds_term_structure(params, tau_grid, lam_f, lam_g, payment_interval=0.25):
    bond_spread = np.array([foreign_credit_spread(params, tau, lam_f, lam_g) for tau in tau_grid])
    cds_spread = cds_term_structure(params, tau_grid, lam_f, lam_g, payment_interval=payment_interval)

    plt.figure(figsize=(9.0, 5.4))
    plt.plot(tau_grid, 1e4 * bond_spread, linewidth=2.4, label=r"Bond credit spread $s^*(\tau)$")
    plt.plot(tau_grid, 1e4 * cds_spread, linewidth=2.4, linestyle="--", label=r"Fair CDS spread $s_{CDS}(\tau)$")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spread (bp)")
    plt.grid(alpha=0.28)
    plt.legend()
    _savefig("bond_spread_vs_cds_term_structure.png")


# (everything above unchanged...)

def plot_spreads_vs_lambda_f(params, lam_f_grid, lam_g, maturity=5.0, payment_interval=0.25):
    bond_vals = np.array([foreign_credit_spread(params, maturity, lam_f, lam_g) for lam_f in lam_f_grid])
    cds_vals = np.array([fair_cds_spread(params, maturity, lam_f, lam_g, payment_interval=payment_interval) for lam_f in lam_f_grid])

    # =========================
    # DEBUG BLOCK
    # =========================
    print("\n==============================")
    print("DEBUG: lambda_f plot")
    print("==============================")

    print(f"lambda_f range: {lam_f_grid[0]:.8f} → {lam_f_grid[-1]:.8f}")

    print("\n--- Spread Magnitudes ---")
    print(f"Bond spread min/max: {1e4 * bond_vals.min():.6f}, {1e4 * bond_vals.max():.6f} bp")
    print(f"Bond spread range: {(1e4 * (bond_vals.max() - bond_vals.min())):.6f} bp")

    print(f"CDS spread min/max: {1e4 * cds_vals.min():.6f}, {1e4 * cds_vals.max():.6f} bp")
    print(f"CDS spread range: {(1e4 * (cds_vals.max() - cds_vals.min())):.6f} bp")

    bond_slope = np.gradient(bond_vals, lam_f_grid)
    print("\n--- Bond Spread Slope ---")
    print(f"Slope min/max: {bond_slope.min():.6f}, {bond_slope.max():.6f}")
    print(f"Average slope: {bond_slope.mean():.6f}")

    print("\n--- Sample Points ---")
    for i in [0, len(lam_f_grid)//2, -1]:
        print(f"lambda_f={lam_f_grid[i]:.8f}, bond={1e4 * bond_vals[i]:.6f} bp")

    # =========================

    plt.figure(figsize=(9.0, 5.4))
    plt.plot(lam_f_grid, 1e4 * bond_vals, linewidth=2.4, label=rf"Bond spread, $\tau={maturity:g}$")
    plt.plot(lam_f_grid, 1e4 * cds_vals, linewidth=2.4, linestyle="--", label=rf"CDS spread, $\tau={maturity:g}$")
    plt.xlabel(r"Foreign disaster intensity $\lambda_t^f$")
    plt.ylabel("Spread (bp)")
    plt.grid(alpha=0.28)
    plt.legend()
    _savefig(f"spreads_vs_lambda_f_{str(maturity).replace('.', 'p')}y.png")


def plot_spreads_vs_lambda_g(params, lam_g_grid, lam_f, maturity=5.0, payment_interval=0.25):
    bond_vals = np.array([foreign_credit_spread(params, maturity, lam_f, lam_g) for lam_g in lam_g_grid])
    cds_vals = np.array([fair_cds_spread(params, maturity, lam_f, lam_g, payment_interval=payment_interval) for lam_g in lam_g_grid])

    # =========================
    # DEBUG BLOCK
    # =========================
    print("\n==============================")
    print("DEBUG: lambda_g plot")
    print("==============================")

    print(f"lambda_g range: {lam_g_grid[0]:.8f} → {lam_g_grid[-1]:.8f}")

    print("\n--- Spread Magnitudes ---")
    print(f"Bond spread min/max: {1e4 * bond_vals.min():.6f}, {1e4 * bond_vals.max():.6f} bp")
    print(f"Bond spread range: {(1e4 * (bond_vals.max() - bond_vals.min())):.6f} bp")

    print(f"CDS spread min/max: {1e4 * cds_vals.min():.6f}, {1e4 * cds_vals.max():.6f} bp")
    print(f"CDS spread range: {(1e4 * (cds_vals.max() - cds_vals.min())):.6f} bp")

    bond_slope = np.gradient(bond_vals, lam_g_grid)
    print("\n--- Bond Spread Slope ---")
    print(f"Slope min/max: {bond_slope.min():.6f}, {bond_slope.max():.6f}")
    print(f"Average slope: {bond_slope.mean():.6f}")

    print("\n--- Sample Points ---")
    for i in [0, len(lam_g_grid)//2, -1]:
        print(f"lambda_g={lam_g_grid[i]:.8f}, bond={1e4 * bond_vals[i]:.6f} bp")

    # =========================

    plt.figure(figsize=(9.0, 5.4))
    plt.plot(lam_g_grid, 1e4 * bond_vals, linewidth=2.4, label=rf"Bond spread, $\tau={maturity:g}$")
    plt.plot(lam_g_grid, 1e4 * cds_vals, linewidth=2.4, linestyle="--", label=rf"CDS spread, $\tau={maturity:g}$")
    plt.xlabel(r"Global disaster intensity $\lambda_t^g$")
    plt.ylabel("Spread (bp)")
    plt.grid(alpha=0.28)
    plt.legend()
    _savefig(f"spreads_vs_lambda_g_{str(maturity).replace('.', 'p')}y.png")


def plot_spreads_vs_eta_f(params, eta_f_grid, maturity=5.0, payment_interval=0.25):
    lam_h, lam_g, lam_f = state_from_params(params)
    bond_vals = []
    cds_vals = []
    old_eta_f = params.eta1

    for eta_f in eta_f_grid:
        params.eta1 = float(eta_f)
        bond_vals.append(foreign_credit_spread(params, maturity, lam_f, lam_g))
        cds_vals.append(fair_cds_spread(params, maturity, lam_f, lam_g, payment_interval=payment_interval))

    params.eta1 = old_eta_f

    plt.figure(figsize=(9.0, 5.4))
    plt.plot(eta_f_grid, 1e4 * np.array(bond_vals), linewidth=2.4, label=rf"Bond spread, $\tau={maturity:g}$")
    plt.plot(eta_f_grid, 1e4 * np.array(cds_vals), linewidth=2.4, linestyle="--", label=rf"CDS spread, $\tau={maturity:g}$")
    plt.xlabel(r"Hazard loading $\eta_f$")
    plt.ylabel("Spread (bp)")
    plt.grid(alpha=0.28)
    plt.legend()
    _savefig(f"spreads_vs_eta_f_{str(maturity).replace('.', 'p')}y.png")


# =========================
# Main driver
# =========================

def main():
    os.makedirs(FIGDIR, exist_ok=True)

    params = DisasterModelParams()
    if hasattr(params, "v"):
        params.v = 0.0

    params.compute_b_sdf()
    check_parameter_admissibility(params)

    # =========================
    # DEBUG: affine coefficients
    # =========================
    print("\n==============================")
    print("DEBUG: Affine Coefficients @ tau=5")
    print("==============================")

    tau_test = 5.0
    a_rf = a_star(params, tau_test)
    b_rf = b_star(params, tau_test)

    a_d, b_df, b_dg = defaultable_foreign_coeffs(params, tau_test)

    print(f"b_rf: {b_rf:.6f}")
    print(f"b_df: {b_df:.6f}")
    print(f"b_dg: {b_dg:.6f}")
    print(f"b_df - b_rf: {(b_df - b_rf):.6f}")
    print(f"b_dg - b_rf: {(b_dg - b_rf):.6f}")

    # A coefficients
    C = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1)
    Af = C + (1 - params.R) * params.eta1
    Ag = C + (1 - params.R) * params.eta2

    print("\nA coefficients:")
    print(f"C: {C:.6f}")
    print(f"A_f: {Af:.6f}")
    print(f"A_g: {Ag:.6f}")

    lam_h, lam_g, lam_f = state_from_params(params)
    tau_grid = np.linspace(0.25, 60.0, 220)

    plot_nonquanto_bond_prices(params, tau_grid, lam_h, lam_g, lam_f)
    plot_yield_term_structures(params, tau_grid, lam_h, lam_g, lam_f)
    plot_bond_spread_vs_cds_term_structure(params, tau_grid, lam_f, lam_g)

    lam_f_grid = np.linspace(max(1e-5, 0.25 * lam_f), 3.0 * lam_f if lam_f > 0 else 0.06, 120)
    lam_g_grid = np.linspace(max(1e-5, 0.25 * lam_g), 3.0 * lam_g if lam_g > 0 else 0.06, 120)
    eta_f_grid = np.linspace(max(1e-5, 0.4 * params.eta1), 2.0 * params.eta1 if params.eta1 > 0 else 2.0, 120)

    plot_spreads_vs_lambda_f(params, lam_f_grid, lam_g, maturity=5.0)
    plot_spreads_vs_lambda_g(params, lam_g_grid, lam_f, maturity=5.0)
    plot_spreads_vs_eta_f(params, eta_f_grid, maturity=5.0)

    # Small console summary for quick sanity checks.
    cds_5y = fair_cds_spread(params, 5.0, lam_f, lam_g)
    bond_5y = foreign_credit_spread(params, 5.0, lam_f, lam_g)
    print("\n5Y bond credit spread (bp):", 1e4 * bond_5y)
    print("5Y fair CDS spread (bp):   ", 1e4 * cds_5y)


if __name__ == "__main__":
    main()
