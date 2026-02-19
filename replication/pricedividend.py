import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from params.modelparams import DisasterModelParams

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

FIGDIR = os.path.join(BASE_DIR, "figures")

def solve_a_b(params: DisasterModelParams, tau_max: float, n_tau: int):
    tau_grid = np.linspace(0.0, tau_max, n_tau)
    params.compute_b_sdf()

    def rhs(tau, y):
        bphi, aphi = y
        db = (
            -params.kappa * bphi
            + (0.5 * bphi * bphi + params.b_sdf * bphi) * (params.sigma_lambda ** 2)
            - np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - np.exp(params.phi * params.Z))
        )
        da = (
            -(params.beta + params.mu - params.mu_D + params.gamma * (params.phi - 1.0) * (params.sigma_c ** 2))
            + params.kappa * (params.lam_bar_f + params.lam_bar_g) * bphi
        )
        return [db, da]

    sol = solve_ivp(
        rhs,
        (0.0, tau_max),
        y0=[0.0, 0.0],
        t_eval=tau_grid,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    bphi = sol.y[0]
    aphi = sol.y[1]
    return tau_grid, aphi, bphi

def pd_ratio_curve(params: DisasterModelParams, lambda_grid: np.ndarray, tau_max: float = 200.0, n_tau: int = 6000):
    tau, aphi, bphi = solve_a_b(params, tau_max=tau_max, n_tau=n_tau)
    pd_vals = np.empty_like(lambda_grid, dtype=float)
    for i, lam in enumerate(lambda_grid):
        integrand = np.exp(aphi + bphi * lam)
        pd_vals[i] = np.trapz(integrand, tau)
    return pd_vals

def main():
    params = DisasterModelParams()

    lambda_grid = np.linspace(0.0, 0.035, 200)
    pd_vals = pd_ratio_curve(params, lambda_grid, tau_max=200.0, n_tau=6000)

    plt.figure()
    plt.plot(lambda_grid, pd_vals)
    plt.xlabel(r"$\lambda_t$")
    plt.ylabel(r"$P_t/D_t$")
    plt.title(r"$P_t/D_t$ vs $\lambda_t$")
    plt.grid(True)
    plt.savefig(os.path.join(FIGDIR, "pd_ratio.png"), dpi=300)

if __name__ == "__main__":
    main()