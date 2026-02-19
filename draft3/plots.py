import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------
# Package path and parameters
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from params.modelparams import DisasterModelParams

FIGDIR = os.path.join(BASE_DIR, "figures")

# ---------------------------------------------------------------------
# Core coefficient ODEs
# ---------------------------------------------------------------------

def get_riskfree_coeffs(params: DisasterModelParams, tau: float):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)

    lam_bar_total = params.lam_bar_f + params.lam_bar_g

    def ode(tau_local, y):
        a, b = y
        da = (
            params.kappa * lam_bar_total * b
            - params.beta
            - params.mu
            + params.gamma * params.sigma_c ** 2
        )
        db = (
            -(params.kappa + params.v) * b
            + bbar * params.sigma_lambda ** 2 * b
            + 0.5 * params.sigma_lambda ** 2 * b ** 2
            + C * (np.exp(b * params.v) - np.exp(params.Z))
        )
        return [da, db]

    if tau == 0.0:
        return 0.0, 0.0

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0])
    a_tau, b_tau = sol.y[:, -1]
    return a_tau, b_tau


def get_domestic_riskfree_coeffs(params: DisasterModelParams, tau: float):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)

    lam_bar_total = params.lam_bar_h + params.lam_bar_g

    def ode(tau_local, y):
        a, b = y
        da = (
            params.kappa * lam_bar_total * b
            - params.beta
            - params.mu
            + params.gamma * params.sigma_c ** 2
        )
        db = (
            -(params.kappa + params.v) * b
            + bbar * params.sigma_lambda ** 2 * b
            + 0.5 * params.sigma_lambda ** 2 * b ** 2
            + C * (np.exp(b * params.v) - np.exp(params.Z))
        )
        return [da, db]

    if tau == 0.0:
        return 0.0, 0.0

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0])
    a_tau, b_tau = sol.y[:, -1]
    return a_tau, b_tau


def hazard_affine_coeffs(params: DisasterModelParams):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z) * (np.exp(params.Z) - 1.0)

    A0 = (
        params.beta
        + params.mu
        - params.gamma * params.sigma_c ** 2
        + (1.0 - params.R) * params.h0_star
    )
    Af = C + (1.0 - params.R) * params.eta1
    Ag = C + (1.0 - params.R) * params.eta2
    return A0, Af, Ag


def get_defaultable_coeffs(params: DisasterModelParams, tau: float):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)
    A0, Af, Ag = hazard_affine_coeffs(params)

    def ode(tau_local, y):
        a, b_f, b_g = y

        db_f = (
            -(params.kappa + params.v) * b_f
            + bbar * params.sigma_lambda ** 2 * b_f
            + 0.5 * params.sigma_lambda ** 2 * b_f ** 2
            + C * (np.exp(b_f * params.v) - np.exp(params.Z))
            - Af
        )

        db_g = (
            -(params.kappa + params.v) * b_g
            + bbar * params.sigma_lambda ** 2 * b_g
            + 0.5 * params.sigma_lambda ** 2 * b_g ** 2
            + C * (np.exp(b_g * params.v) - np.exp(params.Z))
            - Ag
        )

        da = params.kappa * (
            params.lam_bar_f * b_f + params.lam_bar_g * b_g
        ) - A0

        return [da, db_f, db_g]

    if tau == 0.0:
        return 0.0, 0.0, 0.0

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0, 0.0])
    a_tau, b_f_tau, b_g_tau = sol.y[:, -1]
    return a_tau, b_f_tau, b_g_tau


def quanto_affine_coeffs(params: DisasterModelParams):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z) * (np.exp(params.Z) - 1.0)

    A0_q = (
        params.beta
        + params.mu
        - params.gamma * params.sigma_c ** 2
        + (1.0 - params.R) * params.h0_star
    )
    Ah_q = C
    Ag_q = C + (1.0 - params.R) * params.eta2
    Af_q = (1.0 - params.R) * params.eta1
    return A0_q, Ah_q, Ag_q, Af_q, C


def get_quanto_defaultable_coeffs(params: DisasterModelParams, tau: float):
    params.compute_b_sdf()
    bbar = params.b_sdf
    sigma_l = params.sigma_lambda
    kappa = params.kappa
    v = params.v

    A0_q, Ah_q, Ag_q, Af_q, C = quanto_affine_coeffs(params)

    def ode(tau_local, y):
        a, b_h, b_g, b_f = y

        db_h = (
            (bbar * sigma_l ** 2 - kappa - v) * b_h
            + 0.5 * sigma_l ** 2 * b_h ** 2
            + np.exp(bbar * v - params.gamma * params.Z)
            * (np.exp(b_h * v) - np.exp(params.Z))
            - Ah_q
        )

        db_g = (
            (bbar * sigma_l ** 2 - kappa - v) * b_g
            + 0.5 * sigma_l ** 2 * b_g ** 2
            + np.exp(bbar * v - params.gamma * params.Z)
            * (np.exp(b_g * v) - np.exp(params.Z))
            - Ag_q
        )

        db_f = (
            (-kappa - v) * b_f
            + 0.5 * sigma_l ** 2 * b_f ** 2
            + (np.exp(b_f * v) - 1.0)
            - Af_q
        )

        da = kappa * (
            params.lam_bar_h * b_h
            + params.lam_bar_g * b_g
            + params.lam_bar_f * b_f
        ) - A0_q

        return [da, db_h, db_g, db_f]

    if tau == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0, 0.0, 0.0])
    a_tau, b_h_tau, b_g_tau, b_f_tau = sol.y[:, -1]
    return a_tau, b_h_tau, b_g_tau, b_f_tau


# ---------------------------------------------------------------------
# Helper: bond prices, yields, loadings
# ---------------------------------------------------------------------

def compute_bond_term_structures(params, tau_grid, lam_h, lam_g, lam_f):
    B_star = np.zeros_like(tau_grid)
    B_dom = np.zeros_like(tau_grid)
    B_D_star = np.zeros_like(tau_grid)
    B_quanto = np.zeros_like(tau_grid)

    for i, tau in enumerate(tau_grid):
        a_rf, b_rf = get_riskfree_coeffs(params, float(tau))
        a_dom, b_dom = get_domestic_riskfree_coeffs(params, float(tau))
        a_d, b_df, b_dg = get_defaultable_coeffs(params, float(tau))
        a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, float(tau))

        lam_star = lam_f + lam_g
        lam_home = lam_h + lam_g

        B_star[i] = np.exp(a_rf + b_rf * lam_star)
        B_dom[i] = np.exp(a_dom + b_dom * lam_home)
        B_D_star[i] = np.exp(a_d + b_df * lam_f + b_dg * lam_g)
        B_quanto[i] = np.exp(a_q + b_qh * lam_h + b_qg * lam_g + b_qf * lam_f)

    return B_star, B_dom, B_D_star, B_quanto


def compute_loading_term_structures(params, tau_grid):
    b_star = np.zeros_like(tau_grid)
    b_dom = np.zeros_like(tau_grid)
    b_D_f = np.zeros_like(tau_grid)
    b_D_g = np.zeros_like(tau_grid)
    b_q_h = np.zeros_like(tau_grid)
    b_q_g = np.zeros_like(tau_grid)
    b_q_f = np.zeros_like(tau_grid)

    for i, tau in enumerate(tau_grid):
        _, b_star[i] = get_riskfree_coeffs(params, float(tau))
        _, b_dom[i] = get_domestic_riskfree_coeffs(params, float(tau))
        _, b_D_f[i], b_D_g[i] = get_defaultable_coeffs(params, float(tau))
        _, b_q_h[i], b_q_g[i], b_q_f[i] = get_quanto_defaultable_coeffs(params, float(tau))

    return {
        "b_star": b_star,
        "b_dom": b_dom,
        "b_D_f": b_D_f,
        "b_D_g": b_D_g,
        "b_q_h": b_q_h,
        "b_q_g": b_q_g,
        "b_q_f": b_q_f,
    }


def compute_yield_term_structures(params, tau_grid, lam_h, lam_g, lam_f):
    B_star, B_dom, B_D_star, B_quanto = compute_bond_term_structures(
        params, tau_grid, lam_h, lam_g, lam_f
    )

    y_star = -np.log(B_star) / tau_grid * 100.0
    y_dom = -np.log(B_dom) / tau_grid * 100.0
    y_D_star = -np.log(B_D_star) / tau_grid * 100.0
    y_quanto = -np.log(B_quanto) / tau_grid * 100.0

    s_star = y_D_star - y_star
    s_tilde = y_quanto - y_dom
    q_tilde = y_quanto - y_D_star

    return {
        "y_star": y_star,
        "y_dom": y_dom,
        "y_D_star": y_D_star,
        "y_quanto": y_quanto,
        "s_star": s_star,
        "s_tilde": s_tilde,
        "q_tilde": q_tilde,
    }


# ---------------------------------------------------------------------
# 3.1 Bond price term structures
# ---------------------------------------------------------------------

def plot_bond_price_term_structures(params):
    tau_grid = np.linspace(0.25, 10.0, 120)
    lam_h, lam_g, lam_f = params.lam_bar_h, params.lam_bar_g, params.lam_bar_f

    B_star, B_dom, B_D_star, B_quanto = compute_bond_term_structures(
        params, tau_grid, lam_h, lam_g, lam_f
    )

    plt.figure()
    plt.plot(tau_grid, B_star, label=r"$B^*(\tau)$ (foreign risk-free)")
    plt.plot(tau_grid, B_dom, label=r"$B(\tau)$ (domestic risk-free)")
    plt.plot(tau_grid, B_D_star, linestyle="--", label=r"$B_D^*(\tau)$ (foreign def.)")
    plt.plot(tau_grid, B_quanto, linestyle="--", label=r"$\tilde B_D^*(\tau)$ (quanto)")
    plt.xlabel(r"Maturity $\tau$ (years)")
    plt.ylabel(r"Price $B(\tau)$")
    plt.title(r"Bond price term structures")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_1_bond_price_term_structures.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 3.2 Loading functions on disaster intensities
# ---------------------------------------------------------------------

def plot_loading_functions(params):
    tau_grid = np.linspace(0.25, 10.0, 120)
    coeffs = compute_loading_term_structures(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["b_star"], linestyle="-", label=r"$b^*(\tau)$")
    plt.plot(tau_grid, coeffs["b_dom"], linestyle="--", label=r"$b(\tau)$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Loading on $\lambda$")
    plt.title(r"Risk-free loadings $b^*(\tau)$ and $b(\tau)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_2a_riskfree_loadings.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(tau_grid, coeffs["b_D_f"], label=r"$b_{D,f}^*(\tau)$")
    plt.plot(tau_grid, coeffs["b_D_g"], label=r"$b_{D,g}^*(\tau)$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Loading on $\lambda_t^f,\lambda_t^g$")
    plt.title(r"Defaultable bond loadings $b_{D,f}^*(\tau)$ and $b_{D,g}^*(\tau)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_2b_defaultable_loadings.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(tau_grid, coeffs["b_q_h"], label=r"$\tilde b_h(\tau)$")
    plt.plot(tau_grid, coeffs["b_q_g"], label=r"$\tilde b_g(\tau)$")
    plt.plot(tau_grid, coeffs["b_q_f"], label=r"$\tilde b_f(\tau)$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Loading on $\lambda_t^h,\lambda_t^g,\lambda_t^f$")
    plt.title(r"Quanto loadings $\tilde b_i(\tau)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_2c_quanto_loadings.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(tau_grid, coeffs["b_q_f"], label=r"$\tilde b_f(\tau)$")
    plt.plot(tau_grid, coeffs["b_D_f"], label=r"$b_{D,f}^*(\tau)$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Loading on $\lambda_t^f$")
    plt.title(r"Loadings on $\lambda_t^f$: quanto $\tilde b_f(\tau)$ vs. defaultable $b_{D,f}^*(\tau)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_2d_defaultable_vs_quanto_loadings.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 3.3 Yield and spread term structures
# ---------------------------------------------------------------------

def plot_yield_and_spread_term_structures(params):
    tau_grid = np.linspace(0.5, 50.0, 120)
    lam_h, lam_g, lam_f = params.lam_bar_h, params.lam_bar_g, params.lam_bar_f

    curves = compute_yield_term_structures(params, tau_grid, lam_h, lam_g, lam_f)

    plt.figure()
    plt.plot(tau_grid, curves["y_star"], label=r"$y^*(\tau)$ (foreign RF)")
    plt.plot(tau_grid, curves["y_dom"], label=r"$y(\tau)$ (domestic RF)")
    plt.plot(tau_grid, curves["y_D_star"], label=r"$y_D^*(\tau)$ (foreign def.)")
    plt.plot(tau_grid, curves["y_quanto"], label=r"$\tilde y(\tau)$ (quanto)")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Yield (annual \%)")
    plt.title(r"Yield term structures")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_3_yield_term_structures.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(tau_grid, curves["s_star"], label=r"$s^*(\tau)$ (foreign spread)")
    plt.plot(tau_grid, curves["s_tilde"], label=r"$\tilde s(\tau)$ (domestic spread)")
    plt.plot(tau_grid, curves["q_tilde"], label=r"$\tilde q(\tau)$ (quanto basis)")
    plt.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Spread (percentage points)")
    plt.title(r"Credit spreads and quanto basis")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_3_spread_term_structures.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 3.4 Sensitivities of bond prices
# ---------------------------------------------------------------------

def compute_sensitivities_vs_tau(params, tau_grid, lam_h, lam_g, lam_f):
    dBD_dlamf = np.zeros_like(tau_grid)
    dBq_dlamf = np.zeros_like(tau_grid)
    dBq_dlamg = np.zeros_like(tau_grid)
    dBq_dlamh = np.zeros_like(tau_grid)

    for i, tau in enumerate(tau_grid):
        a_d, b_df, b_dg = get_defaultable_coeffs(params, float(tau))
        a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, float(tau))

        B_D = np.exp(a_d + b_df * lam_f + b_dg * lam_g)
        B_q = np.exp(a_q + b_qh * lam_h + b_qg * lam_g + b_qf * lam_f)

        dBD_dlamf[i] = b_df * B_D
        dBq_dlamf[i] = b_qf * B_q
        dBq_dlamg[i] = b_qg * B_q
        dBq_dlamh[i] = b_qh * B_q

    return dBD_dlamf, dBq_dlamf, dBq_dlamg, dBq_dlamh


def plot_sensitivities(params):
    tau_grid = np.linspace(0.5, 10.0, 120)
    lam_h, lam_g, lam_f = params.lam_bar_h, params.lam_bar_g, params.lam_bar_f

    dBD_dlamf, dBq_dlamf, dBq_dlamg, dBq_dlamh = compute_sensitivities_vs_tau(
        params, tau_grid, lam_h, lam_g, lam_f
    )

    plt.figure()
    plt.plot(tau_grid, dBD_dlamf, label=r"$\partial B_D^*(\tau)/\partial \lambda_t^f$")
    plt.plot(
        tau_grid,
        dBq_dlamf,
        label=r"$\partial \tilde B_D^*(\tau)/\partial \lambda_t^f$",
    )
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Sensitivity to $\lambda_t^f$")
    plt.title(r"Sensitivity of foreign and quanto bonds to $\lambda_t^f$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_4_sensitivity_lambda_f.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(tau_grid, dBq_dlamf, label=r"$\partial \tilde B_D^*/\partial \lambda_t^f$")
    plt.plot(tau_grid, dBq_dlamg, label=r"$\partial \tilde B_D^*/\partial \lambda_t^g$")
    plt.plot(tau_grid, dBq_dlamh, label=r"$\partial \tilde B_D^*/\partial \lambda_t^h$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Sensitivity of $\tilde B_D^*$")
    plt.title(r"Quanto bond sensitivities to $(\lambda_t^h,\lambda_t^g,\lambda_t^f)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGDIR, "sec6_4_sensitivity_quanto_all_intensities.png"),
        dpi=300,
    )
    plt.close()


# ---------------------------------------------------------------------
# 3.5 Quanto basis as function of maturity and default risk
# ---------------------------------------------------------------------

def plot_quanto_basis_profiles(params):
    lam_h, lam_g, lam_f = params.lam_bar_h, params.lam_bar_g, params.lam_bar_f

    tau_grid = np.linspace(0.5, 10.0, 120)
    curves = compute_yield_term_structures(params, tau_grid, lam_h, lam_g, lam_f)
    q_tilde = curves["q_tilde"]

    plt.figure()
    plt.plot(tau_grid, q_tilde * 100, label=r"$\tilde q(\tau)$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Quanto basis (bps)")
    plt.title(r"Quanto basis term structure $\tilde q(\tau)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_5_quanto_basis_vs_tau.png"), dpi=300)
    plt.close()

    tau0 = 5.0
    lam_f_grid = np.linspace(0.0, 0.12, 150)
    q_vals = np.zeros_like(lam_f_grid)

    for i, lf in enumerate(lam_f_grid):
        a_d, b_df, b_dg = get_defaultable_coeffs(params, tau0)
        a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, tau0)

        B_D = np.exp(a_d + b_df * lf + b_dg * lam_g)
        B_q = np.exp(a_q + b_qh * lam_h + b_qg * lam_g + b_qf * lf)

        y_D = -np.log(B_D) / tau0 * 100.0
        y_q = -np.log(B_q) / tau0 * 100.0
        q_vals[i] = y_q - y_D

    plt.figure()
    plt.plot(lam_f_grid, q_vals * 100)
    plt.xlabel(r"Foreign disaster intensity $\lambda_t^f$")
    plt.ylabel(r"Quanto basis at $\tau_0$ (bps)")
    plt.title(
        r"Quanto basis $\tilde q(\tau_0)$ vs $\lambda_t^f$,  "
        r"$\tau_0=5$ years"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec6_5_quanto_basis_vs_lambda_f.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Correlation between log FX and default intensity (Section 3)
# ---------------------------------------------------------------------

def _rho_P(params: DisasterModelParams, lam_h, lam_f, lam_g):
    params.compute_b_sdf()
    b = params.b_sdf
    sigma_lambda = params.sigma_lambda
    v = params.v
    gamma = params.gamma
    Z = params.Z
    sigma = params.sigma_c
    eta1 = params.eta1
    eta2 = params.eta2
    rho_C = params.rho_C

    num = eta1 * lam_f * (b * sigma_lambda ** 2 + v * (b * v - gamma * Z))

    var_intensity = (sigma_lambda ** 2 + v ** 2) * (eta1 ** 2 * lam_f + eta2 ** 2 * lam_g)
    var_fx = (
        2 * (gamma * sigma) ** 2 * (1 - rho_C)
        + (b * sigma_lambda) ** 2 * (lam_h + lam_f)
        + (b * v - gamma * Z) ** 2 * (lam_h + lam_f)
    )

    denom = np.sqrt(var_intensity * var_fx)
    if denom <= 0:
        return np.nan
    return num / denom


def _rho_Q(params: DisasterModelParams, lam_h, lam_f, lam_g):
    params.compute_b_sdf()
    b = params.b_sdf
    sigma_lambda = params.sigma_lambda
    v = params.v
    gamma = params.gamma
    Z = params.Z
    sigma = params.sigma_c
    eta1 = params.eta1
    eta2 = params.eta2
    rho_C = params.rho_C

    tilt = np.exp(b * v - gamma * Z)

    num = eta1 * lam_f * (b * sigma_lambda ** 2 + v * (b * v - gamma * Z))

    var_intensity = (
        sigma_lambda ** 2 * (eta1 ** 2 * lam_f + eta2 ** 2 * lam_g)
        + v ** 2 * (eta1 ** 2 * lam_f + eta2 ** 2 * tilt * lam_g)
    )

    var_fx = (
        2 * (gamma * sigma) ** 2 * (1 - rho_C)
        + (b * sigma_lambda) ** 2 * (lam_h + lam_f)
        + (b * v - gamma * Z) ** 2 * (tilt * lam_h + lam_f)
    )

    denom = np.sqrt(var_intensity * var_fx)
    if denom <= 0:
        return np.nan
    return num / denom


def _rho_Q_star(params: DisasterModelParams, lam_h, lam_f, lam_g):
    params.compute_b_sdf()
    b = params.b_sdf
    sigma_lambda = params.sigma_lambda
    v = params.v
    gamma = params.gamma
    Z = params.Z
    sigma = params.sigma_c
    eta1 = params.eta1
    eta2 = params.eta2
    rho_C = params.rho_C

    tilt = np.exp(b * v - gamma * Z)

    num = eta1 * lam_f * (b * sigma_lambda ** 2 + v * (b * v - gamma * Z) * tilt)

    var_intensity = (sigma_lambda ** 2 + tilt * v ** 2) * (
        eta1 ** 2 * lam_f + eta2 ** 2 * lam_g
    )

    var_fx = (
        2 * (gamma * sigma) ** 2 * (1 - rho_C)
        + (b * sigma_lambda) ** 2 * (lam_h + lam_f)
        + (b * v - gamma * Z) ** 2 * (lam_h + tilt * lam_f)
    )

    denom = np.sqrt(var_intensity * var_fx)
    if denom <= 0:
        return np.nan
    return num / denom


def plot_corr_vs_lambda_f(params):
    lam_h, lam_g = params.lam_bar_h, params.lam_bar_g
    lam_f_vals = np.linspace(1e-6, 0.20, 200)

    rho_P_vals = np.array([_rho_P(params, lam_h, lf, lam_g) for lf in lam_f_vals])
    rho_Q_vals = np.array([_rho_Q(params, lam_h, lf, lam_g) for lf in lam_f_vals])
    rho_Qs_vals = np.array([_rho_Q_star(params, lam_h, lf, lam_g) for lf in lam_f_vals])

    plt.figure()
    plt.plot(lam_f_vals, rho_P_vals, label=r"$\rho^{\mathbb{P}}$")
    plt.plot(lam_f_vals, rho_Q_vals, label=r"$\rho^{\mathbb{Q}}$")
    plt.plot(lam_f_vals, rho_Qs_vals, label=r"$\rho^{\mathbb{Q}^*}$")
    plt.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    plt.xlabel(r"Foreign disaster intensity $\lambda_t^f$")
    plt.ylabel(r"Instantaneous corr. $\rho^{\mathbb{M}}(d\ln e_t, dh_t^*)$")
    plt.ylim(0.0, 1.0)
    plt.title(
        r"Default--devaluation correlation vs $\lambda_t^f$ "
        r"(saturation level $\bar\rho^{\mathbb{M}}$)"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec3_corr_vs_lambdaf.png"), dpi=300)
    plt.close()


def plot_corr_vs_lambda_g(params):
    lam_h, lam_f = params.lam_bar_h, params.lam_bar_f
    lam_g_vals = np.linspace(1e-6, 0.20, 200)

    rho_P_vals = np.array([_rho_P(params, lam_h, lam_f, lg) for lg in lam_g_vals])
    rho_Q_vals = np.array([_rho_Q(params, lam_h, lam_f, lg) for lg in lam_g_vals])
    rho_Qs_vals = np.array([_rho_Q_star(params, lam_h, lam_f, lg) for lg in lam_g_vals])

    plt.figure()
    plt.plot(lam_g_vals, rho_P_vals, label=r"$\rho^{\mathbb{P}}$")
    plt.plot(lam_g_vals, rho_Q_vals, label=r"$\rho^{\mathbb{Q}}$")
    plt.plot(lam_g_vals, rho_Qs_vals, label=r"$\rho^{\mathbb{Q}^*}$")
    plt.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    plt.xlabel(r"Global disaster intensity $\lambda_t^g$")
    plt.ylabel(r"Instantaneous corr. $\rho^{\mathbb{M}}(d\ln e_t, dh_t^*)$")
    plt.ylim(0.0, 1.0)
    plt.title(
        r"Default--devaluation correlation vs $\lambda_t^g$ "
        r"(saturation level $\bar\rho^{\mathbb{M}}$)"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "sec3_corr_vs_lambdag.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main: clear figure directory and generate all figures
# ---------------------------------------------------------------------

def main():
    if os.path.exists(FIGDIR):
        for fname in os.listdir(FIGDIR):
            fpath = os.path.join(FIGDIR, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(".png"):
                os.remove(fpath)
    else:
        os.makedirs(FIGDIR, exist_ok=True)

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    params = DisasterModelParams()

    plot_bond_price_term_structures(params)
    plot_loading_functions(params)
    plot_yield_and_spread_term_structures(params)
    plot_sensitivities(params)
    plot_quanto_basis_profiles(params)
    plot_corr_vs_lambda_f(params)
    plot_corr_vs_lambda_g(params)


if __name__ == "__main__":
    main()
