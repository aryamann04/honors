import os
import sys
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from params.modelparams import DisasterModelParams
from draft4.ode_solver import (
    get_riskfree_coeffs,
    get_domestic_riskfree_coeffs,
    get_defaultable_coeffs,
    get_quanto_defaultable_coeffs,
)

FIGDIR = os.path.join(BASE_DIR, "figures")

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

def compute_yield_term_structures(params, tau_grid, lam_h, lam_g, lam_f):
    B_star, B_dom, B_D_star, B_quanto = compute_bond_term_structures(
        params, tau_grid, lam_h, lam_g, lam_f
    )

    y_star = -np.log(B_star) / tau_grid * 100.0
    y_dom = -np.log(B_dom) / tau_grid * 100.0
    y_D_star = -np.log(B_D_star) / tau_grid * 100.0
    y_quanto = -np.log(B_quanto) / tau_grid * 100.0

    return y_star, y_dom, y_D_star, y_quanto

def plot_yield_term_structures(params, max_tau):
    tau_grid = np.linspace(0.5, max_tau, 120)
    lam_h, lam_g, lam_f = params.lam_bar_h, params.lam_bar_g, params.lam_bar_f

    y_star, y_dom, y_D_star, y_quanto = compute_yield_term_structures(
        params, tau_grid, lam_h, lam_g, lam_f
    )

    plt.figure()
    plt.plot(tau_grid, y_star, label=r"$y^*(\tau)$")
    plt.plot(tau_grid, y_dom, label=r"$y(\tau)$")
    plt.plot(tau_grid, y_D_star, label=r"$y_D^*(\tau)$")
    plt.plot(tau_grid, y_quanto, label=r"$\tilde y(\tau)$")
    plt.xlabel(r"Maturity $\tau$")
    plt.ylabel(r"Yield (annual \%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "yield_term_structures.png"), dpi=300)
    plt.close()

def main(max_tau=50.0):
    if os.path.exists(FIGDIR):
        for fname in os.listdir(FIGDIR):
            fpath = os.path.join(FIGDIR, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(".png"):
                os.remove(fpath)
    else:
        os.makedirs(FIGDIR, exist_ok=True)

    params = DisasterModelParams()
    plot_yield_term_structures(params, max_tau)

if __name__ == "__main__":
    main(max_tau=30.0)