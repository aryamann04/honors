import numpy as np
import matplotlib.pyplot as plt

from params.modelparams import DisasterModelParams
from pricing import (
    get_riskfree_coeffs,
    get_defaultable_coeffs,
    get_quanto_defaultable_coeffs,
)


def compute_yield_term_structures(
    params,
    tau_grid,
    lam_h=None,
    lam_g=None,
    lam_f=None,
):
    if lam_h is None:
        lam_h = params.lam_bar_h
    if lam_g is None:
        lam_g = params.lam_bar_g
    if lam_f is None:
        lam_f = params.lam_bar_f

    lambda_star = lam_f + lam_g

    y_rf = np.full_like(tau_grid, np.nan, float)
    y_def = np.full_like(tau_grid, np.nan, float)
    y_q = np.full_like(tau_grid, np.nan, float)
    s_credit = np.full_like(tau_grid, np.nan, float)
    q_basis = np.full_like(tau_grid, np.nan, float)

    for i, tau in enumerate(tau_grid):
        if tau <= 0:
            continue
        try:
            a_rf, b_rf = get_riskfree_coeffs(params, tau)
            log_B_rf = a_rf + b_rf * lambda_star

            a_d, b_f_d, b_g_d = get_defaultable_coeffs(params, tau)
            log_B_def = a_d + b_f_d * lam_f + b_g_d * lam_g

            a_q, b_h_q, b_g_q, b_f_q = get_quanto_defaultable_coeffs(params, tau)
            log_B_q = a_q + b_h_q * lam_h + b_g_q * lam_g + b_f_q * lam_f

            if not np.isfinite(log_B_rf) or not np.isfinite(log_B_def) or not np.isfinite(log_B_q):
                continue

            y_rf[i] = -log_B_rf / tau * 100
            y_def[i] = -log_B_def / tau * 100
            y_q[i] = -log_B_q / tau * 100

            s_credit[i] = y_def[i] - y_rf[i]
            q_basis[i] = y_q[i] - y_def[i]

        except:
            continue

    return {
        "y_rf": y_rf,
        "y_def": y_def,
        "y_q": y_q,
        "s_credit": s_credit,
        "q_basis": q_basis,
    }


def plot_yield_term_structures(
    params,
    tau_max=15.0,
    n_tau=80,
    lam_h=None,
    lam_g=None,
    lam_f=None,
    save_prefix="/Users/aryaman/honors/draft2/figures/",
):
    tau_grid = np.linspace(0.25, tau_max, n_tau)

    curves = compute_yield_term_structures(
        params, tau_grid, lam_h=lam_h, lam_g=lam_g, lam_f=lam_f
    )

    y_rf = curves["y_rf"]
    y_def = curves["y_def"]
    y_q = curves["y_q"]
    s_credit = curves["s_credit"]
    q_basis = curves["q_basis"]

    plt.figure()
    plt.plot(tau_grid, y_rf, label="Risk-free yield")
    plt.plot(tau_grid, y_def, label="Defaultable yield")
    plt.plot(tau_grid, y_q, label="Quanto yield", linestyle="--")
    plt.xlabel("Time to maturity τ")
    plt.ylabel("Yield (%)")
    plt.title("Yield term structures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(save_prefix + "term_structures_yields.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(tau_grid, s_credit, label="Credit spread")
    plt.plot(tau_grid, q_basis, label="Quanto basis", linestyle="--")
    plt.axhline(0, color="black", linewidth=0.7, alpha=0.7)
    plt.xlabel("Time to maturity τ")
    plt.ylabel("Spread (pp)")
    plt.title("Credit spread and quanto basis term structures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(save_prefix + "term_structures_spreads.png", dpi=200)
    plt.show()


def plot_short_rate_and_hazard_levels(
    params,
    lam_h=0.000001,
    lam_g=0.000001,
    lam_f=0.000001,
    n_points=200,
    lam_f_max=0.1,
    save_prefix="/Users/aryaman/honors/draft2/figures/",
):
    if lam_g is None:
        lam_g = params.lam_bar_g
    if lam_h is None:
        lam_h = params.lam_bar_h
    if lam_f is None:
        lam_f = params.lam_bar_f

    lam_f_vals = np.linspace(0.0, lam_f_max, n_points)

    params.compute_b_sdf()
    C_r = np.exp(params.b_sdf * params.v - params.gamma * params.Z) * (np.exp(params.Z) - 1)
    A_r = params.beta + params.mu - params.gamma * params.sigma_c ** 2

    r_vals = A_r + C_r * (lam_f_vals + lam_g)
    h_vals = params.h0_star + params.eta1 * lam_f_vals + params.eta2 * lam_g
    adj_vals = r_vals + (1 - params.R) * h_vals

    plt.figure()
    plt.plot(lam_f_vals, r_vals, label="r_t*")
    plt.plot(lam_f_vals, (1 - params.R) * h_vals, label="(1-R)h_t*")
    plt.plot(lam_f_vals, adj_vals, label="r_t* + (1-R)h_t*")
    plt.xlabel("λ^f")
    plt.ylabel("Rates")
    plt.title("Short-rate and hazard components vs λ^f")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(save_prefix + "short_rate_hazard_levels.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    params = DisasterModelParams()
    lam_h = params.lam_bar_h
    lam_g = params.lam_bar_g
    lam_f = params.lam_bar_f

    plot_yield_term_structures(
        params, tau_max=15, n_tau=80, lam_h=lam_h, lam_g=lam_g, lam_f=lam_f
    )

    plot_short_rate_and_hazard_levels(params)
