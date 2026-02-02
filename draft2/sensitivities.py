import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from params.modelparams import DisasterModelParams
from pricing import (
    get_riskfree_coeffs,
    get_defaultable_coeffs,
    get_quanto_defaultable_coeffs,
)

FIGDIR = "/Users/aryaman/honors/draft2/figures"

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


def compute_coeff_term_structures(params: DisasterModelParams, tau_grid: np.ndarray):
    a_rf = np.zeros_like(tau_grid)
    b_rf = np.zeros_like(tau_grid)

    a_dom = np.zeros_like(tau_grid)
    b_dom = np.zeros_like(tau_grid)

    a_d = np.zeros_like(tau_grid)
    b_df = np.zeros_like(tau_grid)
    b_dg = np.zeros_like(tau_grid)

    a_q = np.zeros_like(tau_grid)
    b_qh = np.zeros_like(tau_grid)
    b_qg = np.zeros_like(tau_grid)
    b_qf = np.zeros_like(tau_grid)

    for i, tau in enumerate(tau_grid):
        if tau == 0.0:
            continue

        a_rf[i], b_rf[i] = get_riskfree_coeffs(params, float(tau))
        a_dom[i], b_dom[i] = get_domestic_riskfree_coeffs(params, float(tau))
        a_d[i], b_df[i], b_dg[i] = get_defaultable_coeffs(params, float(tau))
        a_q[i], b_qh[i], b_qg[i], b_qf[i] = get_quanto_defaultable_coeffs(
            params, float(tau)
        )

    return {
        "a_rf": a_rf,
        "b_rf": b_rf,
        "a_dom": a_dom,
        "b_dom": b_dom,
        "a_d": a_d,
        "b_df": b_df,
        "b_dg": b_dg,
        "a_q": a_q,
        "b_qh": b_qh,
        "b_qg": b_qg,
        "b_qf": b_qf,
    }


def plot_b_loadings_term_structures(params: DisasterModelParams, tau_max: float = 15.0):
    tau_grid = np.linspace(0.1, tau_max, 150)
    coeffs = compute_coeff_term_structures(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["b_dom"], label="b_dom (domestic RF)")
    plt.plot(tau_grid, coeffs["b_rf"], label="b_rf (foreign RF)")
    plt.xlabel("Maturity τ (years)")
    plt.ylabel("Loading on disaster intensity")
    plt.title("Risk-free bond λ-loadings vs maturity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/b_loadings_riskfree_term_structures.png")
    plt.close()

    plt.figure()
    plt.plot(tau_grid, coeffs["b_df"], label="b_df (defaultable, λ^f)")
    plt.plot(tau_grid, coeffs["b_dg"], label="b_dg (defaultable, λ^g)")
    plt.xlabel("Maturity τ (years)")
    plt.ylabel("Loading")
    plt.title("Defaultable bond λ-loadings vs maturity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/b_loadings_defaultable_term_structures.png")
    plt.close()

    plt.figure()
    plt.plot(tau_grid, coeffs["b_qh"], label="b_qh (quanto, λ^h)")
    plt.plot(tau_grid, coeffs["b_qg"], label="b_qg (quanto, λ^g)")
    plt.plot(tau_grid, coeffs["b_qf"], label="b_qf (quanto, λ^f)")
    plt.xlabel("Maturity τ (years)")
    plt.ylabel("Loading")
    plt.title("Quanto defaultable bond λ-loadings vs maturity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/b_loadings_quanto_term_structures.png")
    plt.show()
    plt.close()


def compute_bond_prices_for_tau(
    params: DisasterModelParams,
    tau: float,
    lam_h: float,
    lam_g: float,
    lam_f: float,
):
    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_dom, b_dom = get_domestic_riskfree_coeffs(params, tau)
    a_d, b_df, b_dg = get_defaultable_coeffs(params, tau)
    a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, tau)

    lam_star_foreign = lam_f + lam_g
    lam_home = lam_h + lam_g

    B_rf = np.exp(a_rf + b_rf * lam_star_foreign)
    B_dom = np.exp(a_dom + b_dom * lam_home)
    B_d = np.exp(a_d + b_df * lam_f + b_dg * lam_g)
    B_q = np.exp(a_q + b_qh * lam_h + b_qg * lam_g + b_qf * lam_f)

    return {
        "B_rf": B_rf,
        "B_dom": B_dom,
        "B_d": B_d,
        "B_q": B_q,
        "a_rf": a_rf,
        "b_rf": b_rf,
        "a_dom": a_dom,
        "b_dom": b_dom,
        "a_d": a_d,
        "b_df": b_df,
        "b_dg": b_dg,
        "a_q": a_q,
        "b_qh": b_qh,
        "b_qg": b_qg,
        "b_qf": b_qf,
    }


def plot_dB_dlambda_f(params: DisasterModelParams, tau: float = 5.0, lam_f_max: float = 0.1):
    lam_h0 = params.lam_bar_h
    lam_g0 = params.lam_bar_g
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)

    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_dom, b_dom = get_domestic_riskfree_coeffs(params, tau)
    a_d, b_df, b_dg = get_defaultable_coeffs(params, tau)
    a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, tau)

    B_dom = np.exp(a_dom + b_dom * (lam_h0 + lam_g0))

    d_rf = []
    d_dom = []
    d_d = []
    d_q = []

    for lam_f in lam_f_vals:
        lam_star = lam_f + lam_g0
        B_rf = np.exp(a_rf + b_rf * lam_star)
        B_d = np.exp(a_d + b_df * lam_f + b_dg * lam_g0)
        B_q = np.exp(a_q + b_qh * lam_h0 + b_qg * lam_g0 + b_qf * lam_f)

        d_rf.append(b_rf * B_rf)
        d_dom.append(0.0)
        d_d.append(b_df * B_d)
        d_q.append(b_qf * B_q)

    d_rf = np.array(d_rf)
    d_dom = np.array(d_dom)
    d_d = np.array(d_d)
    d_q = np.array(d_q)

    plt.figure()
    plt.plot(lam_f_vals, d_dom, label="dB_dom/dλ^f (domestic RF)")
    plt.plot(lam_f_vals, d_rf, label="dB_rf/dλ^f (foreign RF)")
    plt.plot(lam_f_vals, d_d, label="dB_D/dλ^f (defaultable, foreign)")
    plt.plot(lam_f_vals, d_q, label="dB_q/dλ^f (quanto, domestic)")
    plt.xlabel(r"Foreign intensity $\lambda^f$")
    plt.ylabel(r"$\partial B / \partial \lambda^f$")
    plt.title(rf"Price sensitivity to $\lambda^f$ at $\tau={tau}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/dB_dlambda_f_tau_{tau}.png")
    plt.show()
    plt.close()


def plot_dB_dlambda_g(params: DisasterModelParams, tau: float = 5.0, lam_g_max: float = 0.1):
    lam_h0 = params.lam_bar_h
    lam_f0 = params.lam_bar_f
    lam_g_vals = np.linspace(0.0, lam_g_max, 200)

    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_dom, b_dom = get_domestic_riskfree_coeffs(params, tau)
    a_d, b_df, b_dg = get_defaultable_coeffs(params, tau)
    a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, tau)

    d_rf = []
    d_dom = []
    d_d = []
    d_q = []

    for lam_g in lam_g_vals:
        lam_star = lam_f0 + lam_g
        B_rf = np.exp(a_rf + b_rf * lam_star)
        B_dom = np.exp(a_dom + b_dom * (lam_h0 + lam_g))
        B_d = np.exp(a_d + b_df * lam_f0 + b_dg * lam_g)
        B_q = np.exp(a_q + b_qh * lam_h0 + b_qg * lam_g + b_qf * lam_f0)

        d_rf.append(b_rf * B_rf)
        d_dom.append(b_dom * B_dom)
        d_d.append(b_dg * B_d)
        d_q.append(b_qg * B_q)

    d_rf = np.array(d_rf)
    d_dom = np.array(d_dom)
    d_d = np.array(d_d)
    d_q = np.array(d_q)

    plt.figure()
    plt.plot(lam_g_vals, d_dom, label="dB_dom/dλ^g (domestic RF)")
    plt.plot(lam_g_vals, d_rf, label="dB_rf/dλ^g (foreign RF)")
    plt.plot(lam_g_vals, d_d, label="dB_D/dλ^g (defaultable, foreign)")
    plt.plot(lam_g_vals, d_q, label="dB_q/dλ^g (quanto, domestic)")
    plt.xlabel(r"Global intensity $\lambda^g$")
    plt.ylabel(r"$\partial B / \partial \lambda^g$")
    plt.title(rf"Price sensitivity to $\lambda^g$ at $\tau={tau}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/dB_dlambda_g_tau_{tau}.png")
    plt.show()
    plt.close()


def plot_dB_dlambda_h(params: DisasterModelParams, tau: float = 5.0, lam_h_max: float = 0.1):
    lam_g0 = params.lam_bar_g
    lam_f0 = params.lam_bar_f
    lam_h_vals = np.linspace(0.0, lam_h_max, 200)

    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_dom, b_dom = get_domestic_riskfree_coeffs(params, tau)
    a_d, b_df, b_dg = get_defaultable_coeffs(params, tau)
    a_q, b_qh, b_qg, b_qf = get_quanto_defaultable_coeffs(params, tau)

    d_rf = np.zeros_like(lam_h_vals)
    d_d = np.zeros_like(lam_h_vals)
    d_dom = []
    d_q = []

    for lam_h in lam_h_vals:
        B_dom = np.exp(a_dom + b_dom * (lam_h + lam_g0))
        B_q = np.exp(a_q + b_qh * lam_h + b_qg * lam_g0 + b_qf * lam_f0)
        d_dom.append(b_dom * B_dom)
        d_q.append(b_qh * B_q)

    d_dom = np.array(d_dom)
    d_q = np.array(d_q)

    plt.figure()
    plt.plot(lam_h_vals, d_dom, label="dB_dom/dλ^h (domestic RF)")
    plt.plot(lam_h_vals, d_q, label="dB_q/dλ^h (quanto, domestic)")
    plt.plot(lam_h_vals, d_rf, label="dB_rf/dλ^h (foreign RF)")
    plt.plot(lam_h_vals, d_d, label="dB_D/dλ^h (defaultable, foreign)")
    plt.xlabel(r"Home intensity $\lambda^h$")
    plt.ylabel(r"$\partial B / \partial \lambda^h$")
    plt.title(rf"Price sensitivity to $\lambda^h$ at $\tau={tau}$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/dB_dlambda_h_tau_{tau}.png")
    plt.show()
    plt.close()


def compute_term_structures(
    params: DisasterModelParams,
    tau_grid: np.ndarray,
    lam_h0: float,
    lam_g0: float,
    lam_f0: float,
):
    y_dom = np.zeros_like(tau_grid)
    y_rf = np.zeros_like(tau_grid)
    y_d = np.zeros_like(tau_grid)
    y_q = np.zeros_like(tau_grid)

    for i, tau in enumerate(tau_grid):
        if tau == 0.0:
            continue

        vals = compute_bond_prices_for_tau(params, float(tau), lam_h0, lam_g0, lam_f0)
        B_dom = vals["B_dom"]
        B_rf = vals["B_rf"]
        B_d = vals["B_d"]
        B_q = vals["B_q"]

        y_dom[i] = -np.log(B_dom) / tau
        y_rf[i] = -np.log(B_rf) / tau
        y_d[i] = -np.log(B_d) / tau
        y_q[i] = -np.log(B_q) / tau

    return y_dom, y_rf, y_d, y_q


def plot_yield_and_spread_term_structures(params: DisasterModelParams, tau_max: float = 15.0):
    tau_grid = np.linspace(0.5, tau_max, 150)
    lam_h0 = params.lam_bar_h
    lam_g0 = params.lam_bar_g
    lam_f0 = params.lam_bar_f

    y_dom, y_rf, y_d, y_q = compute_term_structures(
        params, tau_grid, lam_h0, lam_g0, lam_f0
    )

    plt.figure()
    plt.plot(tau_grid, y_dom * 100.0, label="Domestic RF yield")
    plt.plot(tau_grid, y_rf * 100.0, label="Foreign RF yield")
    plt.plot(tau_grid, y_d * 100.0, label="Foreign defaultable yield")
    plt.plot(tau_grid, y_q * 100.0, label="Domestic quanto yield")
    plt.xlabel("Maturity τ (years)")
    plt.ylabel("Yield (annual %)")
    plt.title("Yield term structures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/yield_term_structures.png")
    plt.close()

    credit_spread_foreign = (y_d - y_rf) * 100.0
    quanto_basis = (y_q - y_d) * 100.0
    local_vs_foreign_spread = (y_q - y_dom) * 100.0

    plt.figure()
    plt.plot(tau_grid, credit_spread_foreign, label="Foreign credit spread y_D^* - y^*")
    plt.plot(tau_grid, quanto_basis, label="Quanto basis ŷ_D^* - y_D^*")
    plt.plot(
        tau_grid,
        local_vs_foreign_spread,
        label="Local vs foreign risky ỹ_D^* - y_dom",
    )
    plt.xlabel("Maturity τ (years)")
    plt.ylabel("Spread (percentage points)")
    plt.title("Spread term structures")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/spread_term_structures.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    params = DisasterModelParams()
    plot_b_loadings_term_structures(params, tau_max=15.0)
    plot_dB_dlambda_f(params, tau=5.0, lam_f_max=0.1)
    plot_dB_dlambda_g(params, tau=5.0, lam_g_max=0.1)
    plot_dB_dlambda_h(params, tau=5.0, lam_h_max=0.1)
    plot_yield_and_spread_term_structures(params, tau_max=15.0)
