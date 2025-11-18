import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from scipy.integrate import solve_ivp
from scipy.optimize import newton

@dataclass
class DisasterModelParams:
    # Preferences / consumption
    beta: float = 0.02      # subjective discount rate β
    gamma: float = 4.0      # risk aversion γ
    mu: float = 0.015       # drift of log consumption μ
    sigma_c: float = 0.02   # consumption volatility σ
    Z: float = -0.172       # disaster jump in log consumption (negative)

    # Disaster intensity dynamics (foreign/global)
    kappa: float = 0.15        # mean reversion of λ
    lam_bar_f: float = 0.00255 # long-run mean of foreign intensity λ̄^f
    lam_bar_g: float = 0.01445 # long-run mean of global intensity λ̄^g
    sigma_lambda: float = 0.1  # volatility of intensity σ_λ
    v: float = 0.0057          # jump size in intensity when a disaster hits

    # Default/hazard structure
    R: float = 0.4          # recovery of market value (RMV) R
    h0_star: float = 0.02   # baseline hazard h*_0
    eta1: float = 0.6       # loading on λ_f in hazard
    eta2: float = 0.8       # loading on λ_g in hazard

    # Constant b in the SDF 
    b_sdf: float = None    

    def compute_b_sdf(self):
        if self.b_sdf is not None:
            return self.b_sdf

        beta = self.beta
        kappa = self.kappa
        v = self.v
        sigma_l = self.sigma_lambda
        gamma = self.gamma
        Z = self.Z

        def f(b):
            return (-(beta + kappa + v)*b
                    + 0.5 * sigma_l**2 * b**2
                    + np.exp(b*v + (1-gamma)*Z)
                    - 1)

        def fprime(b):
            return (-(beta + kappa + v)
                    + sigma_l**2 * b
                    + v * np.exp(b*v + (1-gamma)*Z))

        b = 0.01 
        for _ in range(50):
            try:
                fb = f(b)
                fpb = fprime(b)

                if not np.isfinite(fb) or not np.isfinite(fpb) or abs(fpb) < 1e-10:
                    b = 0.5 * b  
                    continue

                step = fb / fpb
                b_new = b - step

                if abs(b_new - b) > 1.0:
                    b_new = b - np.sign(step) * 1.0
                b = b_new

            except FloatingPointError:
                b *= 0.5

        self.b_sdf = float(b)
        return self.b_sdf

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

def plot_rf_vs_defaultable(params: DisasterModelParams, tau: float = 5.0):
    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_d, b_f, b_g = get_defaultable_coeffs(params, tau)

    lam_g_fixed = params.lam_bar_g
    lam_f_vals = np.linspace(0.0, 0.2, 200)

    lambda_star_vals = lam_f_vals + lam_g_fixed

    B_rf_vals = np.exp(a_rf + b_rf * lambda_star_vals)
    B_d_vals = np.exp(a_d + b_f * lam_f_vals + b_g * lam_g_fixed)

    plt.figure()
    plt.plot(lam_f_vals, B_rf_vals, label="Risk-free ZCB")
    plt.plot(lam_f_vals, B_d_vals, label="Defaultable ZCB")
    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Bond price")
    plt.title(rf"Risk-free vs defaultable bond ($\tau = {tau}$ years, $\lambda^g$ fixed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
    plt.xlabel(r"$\lambda^f$ (foreign intensity)")
    plt.ylabel(r"$\lambda^g$ (global intensity)")
    plt.title(rf"Defaultable bond price surface ($\tau$ = {tau} years)")
    plt.tight_layout()
    plt.show()

def plot_gamma_sensitivity(params_base: DisasterModelParams, tau: float = 5.0):
    lam_f_vals = np.linspace(0.0, 0.2, 200)
    lam_g_fixed = params_base.lam_bar_g
    gammas = [2.0, 4.0, 8.0]

    plt.figure()
    for g in gammas:
        p = replace(params_base, gamma=g, b_sdf=None)
        a_d, b_f, b_g = get_defaultable_coeffs(p, tau)
        B_d_vals = np.exp(a_d + b_f * lam_f_vals + b_g * lam_g_fixed)
        plt.plot(lam_f_vals, B_d_vals, label=rf"$\gamma = {g}$")

    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Defaultable bond price")
    plt.title(rf"Sensitivity to risk aversion $\gamma$ ($\tau = {tau}$ years, $\lambda^g$ fixed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    params = DisasterModelParams()
    tau = 5.0  

    plot_rf_vs_defaultable(params, tau=tau)
    plot_defaultable_heatmap(params, tau=tau)
    plot_gamma_sensitivity(params, tau=tau)
