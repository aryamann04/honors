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


def safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))


def safe_exp(x):
    return np.exp(np.clip(x, -700.0, 700.0))


def check_parameter_admissibility(params):
    beta = params.beta
    kappa = params.kappa
    sigl = params.sigma_lambda
    gamma = params.gamma
    Z = params.Z
    R = params.R
    eta_f = params.eta1
    eta_g = params.eta2

    failures = []

    if kappa <= 0:
        failures.append(f"kappa must be > 0. Got {kappa}")
    if sigl <= 0:
        failures.append(f"sigma_lambda must be > 0. Got {sigl}")
    if gamma <= 0:
        failures.append(f"gamma must be > 0. Got {gamma}")
    if Z >= 0:
        failures.append(f"Z must be < 0. Got {Z}")
    if not (0 <= R <= 1):
        failures.append(f"R must be in [0,1]. Got {R}")
    if eta_f < 0:
        failures.append(f"eta_f must be >= 0. Got {eta_f}")
    if eta_g < 0:
        failures.append(f"eta_g must be >= 0. Got {eta_g}")

    D0 = np.exp((1 - gamma) * Z) - 1
    if (beta + kappa) ** 2 <= 2 * sigl**2 * D0:
        failures.append("b discriminant condition fails")

    params.compute_b_sdf()
    b = params.b_sdf

    K0 = np.exp(-gamma * Z) * (1 - np.exp(Z))
    eta_max = max(eta_f, eta_g)

    if (b * sigl**2 - kappa) ** 2 <= 2 * sigl**2 * (2 * K0 - (1 - R) * eta_max):
        failures.append("delta condition fails")

    if failures:
        for f in failures:
            print(f)
        raise ValueError("Model parameter restrictions violated.")


def delta_base(params, b):
    return safe_sqrt(
        (b * params.sigma_lambda**2 - params.kappa) ** 2
        - 2 * params.sigma_lambda**2
        * np.exp(-params.gamma * params.Z)
        * (1 - np.exp(params.Z))
    )


def delta_i(params, b, A_i):
    return safe_sqrt(
        (b * params.sigma_lambda**2 - params.kappa) ** 2
        - 2 * params.sigma_lambda**2
        * (
            np.exp(-params.gamma * params.Z)
            * (1 - np.exp(params.Z))
            - A_i
        )
    )


def a_star(params, tau):
    params.compute_b_sdf()
    b = params.b_sdf
    delta = delta_base(params, b)
    phi = b * params.sigma_lambda**2 - params.kappa
    denom = (phi + delta) + (delta - phi) * np.exp(delta * tau)
    term = (delta - phi) * tau + 2 * np.log(2 * delta / denom)
    A0 = params.beta + params.mu - params.gamma * params.sigma_c**2
    lam_bar = params.lam_bar_g + params.lam_bar_f
    return -A0 * tau + (params.kappa * lam_bar / params.sigma_lambda**2) * term


def b_star(params, tau):
    params.compute_b_sdf()
    b = params.b_sdf
    delta = delta_base(params, b)
    phi = b * params.sigma_lambda**2 - params.kappa
    denom = (phi + delta) + (delta - phi) * np.exp(delta * tau)
    num = (
        2
        * np.exp(-params.gamma * params.Z)
        * (1 - np.exp(params.Z))
        * (np.exp(delta * tau) - 1)
    )
    return num / denom


def a_dom(params, tau):
    params.compute_b_sdf()
    b = params.b_sdf
    delta = delta_base(params, b)
    phi = b * params.sigma_lambda**2 - params.kappa
    denom = (phi + delta) + (delta - phi) * np.exp(delta * tau)
    term = (delta - phi) * tau + 2 * np.log(2 * delta / denom)
    A0 = params.beta + params.mu - params.gamma * params.sigma_c**2
    lam_bar = params.lam_bar_g + params.lam_bar_h
    return -A0 * tau + (params.kappa * lam_bar / params.sigma_lambda**2) * term


def defaultable_foreign_coeffs(params, tau):
    params.compute_b_sdf()
    b = params.b_sdf

    A0 = (
        params.beta
        + params.mu
        - params.gamma * params.sigma_c**2
        + (1 - params.R) * params.h0_star
    )

    Af = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1) + (
        1 - params.R
    ) * params.eta1
    Ag = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1) + (
        1 - params.R
    ) * params.eta2

    delta_f = delta_i(params, b, Af)
    delta_g = delta_i(params, b, Ag)

    phi = b * params.sigma_lambda**2 - params.kappa

    denom_f = (phi + delta_f) + (delta_f - phi) * np.exp(delta_f * tau)
    denom_g = (phi + delta_g) + (delta_g - phi) * np.exp(delta_g * tau)

    bf = (
        2
        * (
            np.exp(-params.gamma * params.Z) * (1 - np.exp(params.Z))
            - Af
        )
        * (np.exp(delta_f * tau) - 1)
        / denom_f
    )

    bg = (
        2
        * (
            np.exp(-params.gamma * params.Z) * (1 - np.exp(params.Z))
            - Ag
        )
        * (np.exp(delta_g * tau) - 1)
        / denom_g
    )

    term_f = (delta_f - phi) * tau + 2 * np.log(2 * delta_f / denom_f)
    term_g = (delta_g - phi) * tau + 2 * np.log(2 * delta_g / denom_g)

    a = -A0 * tau
    a += (params.kappa * params.lam_bar_f / params.sigma_lambda**2) * term_f
    a += (params.kappa * params.lam_bar_g / params.sigma_lambda**2) * term_g

    return a, bf, bg


def quanto_coeffs(params, tau):
    params.compute_b_sdf()
    b = params.b_sdf

    A0 = (
        params.beta
        + params.mu
        - params.gamma * params.sigma_c**2
        + (1 - params.R) * params.h0_star
    )

    Ah = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1)
    Ag = Ah + (1 - params.R) * params.eta2
    Af = (1 - params.R) * params.eta1

    delta_h = delta_i(params, b, Ah)
    delta_g = delta_i(params, b, Ag)
    delta_f = safe_sqrt(params.kappa**2 + 2 * params.sigma_lambda**2 * Af)

    phi = b * params.sigma_lambda**2 - params.kappa

    denom_h = (phi + delta_h) + (delta_h - phi) * np.exp(delta_h * tau)
    denom_g = (phi + delta_g) + (delta_g - phi) * np.exp(delta_g * tau)
    denom_f = (delta_f - params.kappa) + (
        delta_f + params.kappa
    ) * np.exp(delta_f * tau)

    bh = (
        2
        * (
            np.exp(-params.gamma * params.Z) * (1 - np.exp(params.Z))
            - Ah
        )
        * (np.exp(delta_h * tau) - 1)
        / denom_h
    )

    bg = (
        2
        * (
            np.exp(-params.gamma * params.Z) * (1 - np.exp(params.Z))
            - Ag
        )
        * (np.exp(delta_g * tau) - 1)
        / denom_g
    )

    bf = (
        -2
        * (1 - params.R)
        * params.eta1
        * (np.exp(delta_f * tau) - 1)
        / denom_f
    )

    term_h = (delta_h - phi) * tau + 2 * np.log(2 * delta_h / denom_h)
    term_g = (delta_g - phi) * tau + 2 * np.log(2 * delta_g / denom_g)
    term_f = (delta_f + params.kappa) * tau + 2 * np.log(2 * delta_f / denom_f)

    a = -A0 * tau
    a += (params.kappa * params.lam_bar_h / params.sigma_lambda**2) * term_h
    a += (params.kappa * params.lam_bar_g / params.sigma_lambda**2) * term_g
    a += (params.kappa * params.lam_bar_f / params.sigma_lambda**2) * term_f

    return a, bh, bg, bf


def bond_prices(params, tau_grid):
    lam_h = params.lam_bar_h
    lam_g = params.lam_bar_g
    lam_f = params.lam_bar_f

    B_star_vals = []
    B_dom_vals = []
    B_D_vals = []
    B_q_vals = []

    for tau in tau_grid:
        a1 = a_star(params, tau)
        b1 = b_star(params, tau)
        B_star_vals.append(safe_exp(a1 + b1 * (lam_f + lam_g)))

        a2 = a_dom(params, tau)
        b2 = b_star(params, tau)
        B_dom_vals.append(safe_exp(a2 + b2 * (lam_h + lam_g)))

        a3, bf, bg = defaultable_foreign_coeffs(params, tau)
        B_D_vals.append(safe_exp(a3 + bf * lam_f + bg * lam_g))

        a4, bh, bgq, bfq = quanto_coeffs(params, tau)
        B_q_vals.append(safe_exp(a4 + bh * lam_h + bgq * lam_g + bfq * lam_f))

    return (
        np.array(B_star_vals),
        np.array(B_dom_vals),
        np.array(B_D_vals),
        np.array(B_q_vals),
    )

def yields(params, tau_grid):
    lam_h = params.lam_bar_h
    lam_g = params.lam_bar_g
    lam_f = params.lam_bar_f

    y_star_vals = []
    y_dom_vals = []
    y_D_vals = []
    y_q_vals = []

    for tau in tau_grid:

        a1 = a_star(params, tau)
        b1 = b_star(params, tau)
        logB_star = a1 + b1 * (lam_f + lam_g)
        y_star_vals.append(-logB_star / tau)

        a2 = a_dom(params, tau)
        b2 = b_star(params, tau)
        logB_dom = a2 + b2 * (lam_h + lam_g)
        y_dom_vals.append(-logB_dom / tau)

        a3, bf, bg = defaultable_foreign_coeffs(params, tau)
        logB_D = a3 + bf * lam_f + bg * lam_g
        y_D_vals.append(-logB_D / tau)

        a4, bh, bgq, bfq = quanto_coeffs(params, tau)
        logB_q = a4 + bh * lam_h + bgq * lam_g + bfq * lam_f
        y_q_vals.append(-logB_q / tau)

    return (
        np.array(y_star_vals),
        np.array(y_dom_vals),
        np.array(y_D_vals),
        np.array(y_q_vals),
    )

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    tau_grid = np.linspace(0.25, 10, 200)
    params = DisasterModelParams()

    params.compute_b_sdf()
    check_parameter_admissibility(params)

    B_star_vals, B_dom_vals, B_D_vals, B_q_vals = bond_prices(params, tau_grid)

    plt.figure(figsize=(8, 5))
    plt.plot(tau_grid, B_star_vals, linewidth=2.5, label="Risk-Free Foreign")
    plt.plot(tau_grid, B_dom_vals, linewidth=2.5, label="Risk-Free Domestic")
    plt.plot(tau_grid, B_D_vals, linewidth=2.5, linestyle="--", label="Defaultable")
    plt.plot(tau_grid, B_q_vals, linewidth=2.5, linestyle="--", label="Quanto")
    plt.ylim(0.55, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "bond_prices.png"), dpi=200)
    plt.close()

    y_star_vals, y_dom_vals, y_D_vals, y_q_vals = yields(params, tau_grid)

    plt.figure(figsize=(8,5))

    plt.plot(tau_grid, y_star_vals * 100, linewidth=2.5, label="y* (Foreign RF)")
    plt.plot(tau_grid, y_dom_vals * 100, linewidth=2.5, label="y (Domestic RF)")
    plt.plot(tau_grid, y_D_vals * 100, linewidth=2.5, linestyle="--", label="y_D* (Foreign Risky)")
    plt.plot(tau_grid, y_q_vals * 100, linewidth=2.5, linestyle="--", label="y~ (Quanto)")

    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(FIGDIR, "yields.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()