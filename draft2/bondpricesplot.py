import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from scipy.integrate import solve_ivp

from modelparams import DisasterModelParams

def get_riskfree_coeffs(params: DisasterModelParams, tau: float):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)
    lam_bar_total = params.lam_bar_f + params.lam_bar_g

    def ode(tau, y):
        a, b = y
        da = (params.kappa * lam_bar_total * b
            - params.beta - params.mu + params.gamma * params.sigma_c ** 2)
        db = (-(params.kappa + params.v) * b
            + bbar * params.sigma_lambda ** 2 * b
            + 0.5 * params.sigma_lambda ** 2 * b ** 2
            + C * (np.exp(b * params.v) - np.exp(params.Z)))
        return [da, db]

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0])
    a_tau, b_tau = sol.y[:, -1]
    return a_tau, b_tau

def hazard_affine_coeffs(params: DisasterModelParams):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z) * (np.exp(params.Z) - 1.0)

    A0 = (params.beta
        + params.mu
        - params.gamma * params.sigma_c ** 2
        + (1.0 - params.R) * params.h0_star)
    Af = C + (1.0 - params.R) * params.eta1
    Ag = C + (1.0 - params.R) * params.eta2

    return A0, Af, Ag

def get_defaultable_coeffs(params: DisasterModelParams, tau: float):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)
    A0, Af, Ag = hazard_affine_coeffs(params)

    def ode(tau, y):
        a, b_f, b_g = y

        db_f = (-(params.kappa + params.v) * b_f
                + bbar * params.sigma_lambda ** 2 * b_f
                + 0.5 * params.sigma_lambda ** 2 * b_f ** 2
                + C * (np.exp(b_f * params.v) - np.exp(params.Z))
                - Af)

        db_g = (-(params.kappa + params.v) * b_g
                + bbar * params.sigma_lambda ** 2 * b_g
                + 0.5 * params.sigma_lambda ** 2 * b_g ** 2
                + C * (np.exp(b_g * params.v) - np.exp(params.Z))
                - Ag)

        da = (params.kappa
            * (params.lam_bar_f * b_f + params.lam_bar_g * b_g)
            - A0)

        return [da, db_f, db_g]

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0, 0.0])
    a_tau, b_f_tau, b_g_tau = sol.y[:, -1]
    return a_tau, b_f_tau, b_g_tau

def compute_bond_prices_vs_lambda_f(params: DisasterModelParams,
                                    tau: float,
                                    lam_f_vals: np.ndarray,
                                    lam_g_fixed: float = None):
    if lam_g_fixed is None:
        lam_g_fixed = params.lam_bar_g

    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_d, b_f, b_g = get_defaultable_coeffs(params, tau)

    lambda_star_vals = lam_f_vals + lam_g_fixed

    B_rf_vals = np.exp(a_rf + b_rf * lambda_star_vals)
    B_d_vals = np.exp(a_d + b_f * lam_f_vals + b_g * lam_g_fixed)

    return B_rf_vals, B_d_vals

def compute_yields_from_prices(B_rf_vals: np.ndarray,
                            B_d_vals: np.ndarray,
                            tau: float):
    y_rf = -np.log(B_rf_vals) / tau * 100.0
    y_d = -np.log(B_d_vals) / tau * 100.0
    spread = y_d - y_rf
    return y_rf, y_d, spread

def plot_rf_vs_defaultable(params: DisasterModelParams, tau: float = 5.0):
    lam_f_vals = np.linspace(0.0, 0.2, 200)
    B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(params, tau, lam_f_vals)

    plt.figure()
    plt.plot(lam_f_vals, B_rf_vals, label="Risk-free ZCB")
    plt.plot(lam_f_vals, B_d_vals, label="Defaultable ZCB")
    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Bond price")
    plt.title(rf"Risk-free vs defaultable bond ($\tau = {tau}$ years, $\lambda^g$ fixed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/rf_vs_defaultable_prices.png")
    plt.show()

def plot_defaultable_heatmap(params: DisasterModelParams, tau: float = 5.0):
    a_d, b_f, b_g = get_defaultable_coeffs(params, tau)
    lam_f_vals = np.linspace(0.0, 0.2, 60)
    lam_g_vals = np.linspace(0.0, 0.2, 60)
    Lf, Lg = np.meshgrid(lam_f_vals, lam_g_vals)
    B_d = np.exp(a_d + b_f * Lf + b_g * Lg)

    plt.figure()
    cp = plt.contourf(Lf, Lg, B_d, levels=20)
    plt.colorbar(cp, label="Defaultable ZCB price")
    plt.xlabel(r"$\lambda^f$")
    plt.ylabel(r"$\lambda^g$")
    plt.title(rf"Defaultable bond price surface ($\tau={tau}$ years)")
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/defaultable_heatmap.png")
    plt.show()

def plot_gamma_sensitivity(params_base: DisasterModelParams, tau: float = 5.0):
    lam_f_vals = np.linspace(0.0, 0.2, 200)
    lam_g_fixed = params_base.lam_bar_g
    gammas = [2.0, 4.0, 8.0]

    plt.figure()
    for g in gammas:
        p = replace(params_base, gamma=g, b_sdf=None)
        B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(p, tau, lam_f_vals, lam_g_fixed)
        plt.plot(lam_f_vals, B_d_vals, label=rf"$\gamma={g}$")

    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Defaultable bond price")
    plt.title(rf"Sensitivity to risk aversion $\gamma$ ($\tau={tau}$ years)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/defaultable_gamma_sensitivity.png")
    plt.show()

def plot_yields_and_spread_vs_lambda_f(params: DisasterModelParams,
                                       tau: float = 5.0,
                                       lam_f_max: float = 0.2):
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)
    B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(params, tau, lam_f_vals)
    y_rf, y_d, spread = compute_yields_from_prices(B_rf_vals, B_d_vals, tau)

    fig, ax1 = plt.subplots()
    ax1.plot(lam_f_vals, y_rf, label=r"Risk-free yield")
    ax1.plot(lam_f_vals, y_d, label=r"Defaultable yield")
    ax1.set_xlabel(r"Foreign disaster intensity $\lambda^f$")
    ax1.set_ylabel(r"Yield (annual %)")
    ax1.set_title(rf"Implied yields and credit spread ($\tau={tau}$ years)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lam_f_vals, spread, label=r"Yield spread (%)", color="black", alpha=0.7, linestyle="--")
    ax2.set_ylabel(r"Yield spread (%)")

    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/yields_and_spread_vs_lambda_f.png")
    plt.show()


def plot_param_sensitivity_defaultable(params_base: DisasterModelParams,
                                    tau: float,
                                    param_name: str,
                                    param_values,
                                    lam_f_max: float = 0.2):
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)
    lam_g_fixed = params_base.lam_bar_g

    plt.figure()
    for val in param_values:
        p = replace(params_base, **{param_name: val}, b_sdf=None)
        B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(p, tau, lam_f_vals, lam_g_fixed)
        if param_name not in ["R", "Z", "v"]: 
            plt.plot(lam_f_vals, B_d_vals, label=rf"$\{param_name}={val}$")
        else: 
            plt.plot(lam_f_vals, B_d_vals, label=rf"${param_name}={val}$")

    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Defaultable bond price")

    if param_name not in ["R", "Z", "v"]:
        plt.title(rf"Sensitivity of defaultable bond to $\{param_name}$ ($\tau={tau}$ years)")
    else:
        plt.title(rf"Sensitivity of defaultable bond to ${param_name}$ ($\tau={tau}$ years)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/Users/aryaman/honors/draft2/figures/defaultable_sensitivity_{param_name}.png")
    plt.show()

def plot_short_rate_vs_lambda_f(params: DisasterModelParams,
                                lam_f_max: float = 0.2):
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)

    params.compute_b_sdf()
    C_r = np.exp(params.b_sdf * params.v - params.gamma * params.Z) * (np.exp(params.Z) - 1)
    A_r = params.beta + params.mu - params.gamma * params.sigma_c**2
    lam_g = params.lam_bar_g
    r_vals = A_r + C_r * (lam_f_vals + lam_g)

    plt.figure()
    plt.plot(lam_f_vals, r_vals, color="blue", label=r"Foreign short rate $r_t^*$")
    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel(r"Short rate $r_t^*$")
    plt.title(r"Foreign short rate $r_t^*$ as function of $\lambda^f$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/short_rate_vs_lambda_f.png")
    plt.show()


if __name__ == "__main__":
    params = DisasterModelParams()
    tau = 5.0  

    plot_rf_vs_defaultable(params, tau=tau)
    # plot_defaultable_heatmap(params, tau=tau)
    plot_gamma_sensitivity(params, tau=tau)
    plot_yields_and_spread_vs_lambda_f(params, tau=tau)
    plot_short_rate_vs_lambda_f(params)

    plot_param_sensitivity_defaultable(params, tau, "beta", [0.01, 0.02, 0.03])
    plot_param_sensitivity_defaultable(params, tau, "mu", [0.005, 0.015, 0.025])
    plot_param_sensitivity_defaultable(params, tau, "sigma_c", [0.01, 0.02, 0.04])
    plot_param_sensitivity_defaultable(params, tau, "Z", [-0.10, -0.172, -0.25])
    plot_param_sensitivity_defaultable(params, tau, "kappa", [0.05, 0.15, 0.3])
    # plot_param_sensitivity_defaultable(params, tau, "sigma_lambda", [0.05, 0.1, 0.2])
    plot_param_sensitivity_defaultable(params, tau, "v", [0.005, 0.01, 0.05])
    plot_param_sensitivity_defaultable(params, tau, "R", [0.2, 0.4, 0.6])
    plot_param_sensitivity_defaultable(params, tau, "eta1", [1, 1.5, 2, 2.5])
    plot_param_sensitivity_defaultable(params, tau, "eta2", [1, 1.5, 2, 2.5])
    