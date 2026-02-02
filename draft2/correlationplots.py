import numpy as np
import matplotlib.pyplot as plt
from params.modelparams import DisasterModelParams

def _rho_P(params: DisasterModelParams,
           lam_h: float,
           lam_f: float,
           lam_g: float) -> float:
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

    num = eta1 * lam_f * (b * sigma_lambda**2 + v * (b * v - gamma * Z))

    var_intensity = (sigma_lambda**2 + v**2) * (eta1**2 * lam_f + eta2**2 * lam_g)
    var_fx = 2 * (gamma * sigma)**2 * (1 - rho_C) + (
        (b * sigma_lambda)**2 + (b * v - gamma * Z)**2
    ) * (lam_h + lam_f)

    denom = np.sqrt(var_intensity * var_fx)
    if denom <= 0:
        return np.nan
    return num / denom

def _rho_Q(params: DisasterModelParams,
           lam_h: float,
           lam_f: float,
           lam_g: float) -> float:
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

    num = eta1 * lam_f * (b * sigma_lambda**2 + v * (b * v - gamma * Z))

    tilt = np.exp(b * v - gamma * Z)

    var_intensity = (
        sigma_lambda**2 * (eta1**2 * lam_f + eta2**2 * lam_g)
        + v**2 * (eta1**2 * lam_f + eta2**2 * tilt * lam_g)
    )

    var_fx = (
        2 * (gamma * sigma)**2 * (1 - rho_C)
        + (b * sigma_lambda)**2 * (lam_h + lam_f)
        + (b * v - gamma * Z)**2 * (tilt * lam_h + lam_f)
    )

    denom = np.sqrt(var_intensity * var_fx)
    if denom <= 0:
        return np.nan
    return num / denom

def _rho_Q_star(params: DisasterModelParams,
                lam_h: float,
                lam_f: float,
                lam_g: float) -> float:
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

    num = eta1 * lam_f * (b * sigma_lambda**2 + v * (b * v - gamma * Z) * tilt)

    var_intensity = (sigma_lambda**2 + tilt * v**2) * (eta1**2 * lam_f + eta2**2 * lam_g)

    var_fx = (
        2 * (gamma * sigma)**2 * (1 - rho_C)
        + (b * sigma_lambda)**2 * (lam_h + lam_f)
        + (b * v - gamma * Z)**2 * (lam_h + tilt * lam_f)
    )

    denom = np.sqrt(var_intensity * var_fx)
    if denom <= 0:
        return np.nan
    return num / denom

def plot_rho_vs_lambda_f(params: DisasterModelParams,
                         lam_f_max: float = 0.2,
                         n_points: int = 200):
    lam_g = params.lam_bar_g
    lam_h = params.lam_bar_h

    lam_f_vals = np.linspace(1e-6, lam_f_max, n_points)
    rho_P_vals = np.array([_rho_P(params, lam_h, lf, lam_g) for lf in lam_f_vals])
    rho_Q_vals = np.array([_rho_Q(params, lam_h, lf, lam_g) for lf in lam_f_vals])
    rho_Qs_vals = np.array([_rho_Q_star(params, lam_h, lf, lam_g) for lf in lam_f_vals])

    plt.figure()
    plt.plot(lam_f_vals, rho_P_vals, label=r"$\rho^\mathbb{P}$")
    plt.plot(lam_f_vals, rho_Q_vals, label=r"$\rho^\mathbb{Q}$")
    plt.plot(lam_f_vals, rho_Qs_vals, label=r"$\rho^{\mathbb{Q}^*}$")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.xlabel(r"Foreign disaster intensity $\lambda_t^f$")
    plt.ylabel(r"Instantaneous corr. with $h_t^*$")
    plt.title(r"FX–hazard correlation vs $\lambda_t^f$")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/corr_vs_lambdaf.png")
    plt.show()

def plot_rho_vs_lambda_g(params: DisasterModelParams,
                         lam_g_max: float = 0.2,
                         n_points: int = 200):
    lam_f = params.lam_bar_f
    lam_h = params.lam_bar_h

    lam_g_vals = np.linspace(1e-6, lam_g_max, n_points)
    rho_P_vals = np.array([_rho_P(params, lam_h, lam_f, lg) for lg in lam_g_vals])
    rho_Q_vals = np.array([_rho_Q(params, lam_h, lam_f, lg) for lg in lam_g_vals])
    rho_Qs_vals = np.array([_rho_Q_star(params, lam_h, lam_f, lg) for lg in lam_g_vals])

    plt.figure()
    plt.plot(lam_g_vals, rho_P_vals, label=r"$\rho^\mathbb{P}$")
    plt.plot(lam_g_vals, rho_Q_vals, label=r"$\rho^\mathbb{Q}$")
    plt.plot(lam_g_vals, rho_Qs_vals, label=r"$\rho^{\mathbb{Q}^*}$")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.xlabel(r"Global disaster intensity $\lambda_t^g$")
    plt.ylabel(r"Instantaneous corr. with $h_t^*$")
    plt.title(r"FX–hazard correlation vs $\lambda_t^g$")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/corr_vs_lambdag.png")
    plt.show()

def plot_rho_vs_lambda_h(params: DisasterModelParams,
                         lam_h_max: float = 0.2,
                         n_points: int = 200):
    lam_f = params.lam_bar_f
    lam_g = params.lam_bar_g

    lam_h_vals = np.linspace(1e-6, lam_h_max, n_points)
    rho_P_vals = np.array([_rho_P(params, lh, lam_f, lam_g) for lh in lam_h_vals])
    rho_Q_vals = np.array([_rho_Q(params, lh, lam_f, lam_g) for lh in lam_h_vals])
    rho_Qs_vals = np.array([_rho_Q_star(params, lh, lam_f, lam_g) for lh in lam_h_vals])

    plt.figure()
    plt.plot(lam_h_vals, rho_P_vals, label=r"$\rho^\mathbb{P}$")
    plt.plot(lam_h_vals, rho_Q_vals, label=r"$\rho^\mathbb{Q}$")
    plt.plot(lam_h_vals, rho_Qs_vals, label=r"$\rho^{\mathbb{Q}^*}$")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.xlabel(r"Home disaster intensity $\lambda_t^h$")
    plt.ylabel(r"Instantaneous corr. with $h_t^*$")
    plt.title(r"FX–hazard correlation vs $\lambda_t^h$")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/corr_vs_lambdah.png")
    plt.show()

def plot_rho_param_sensitivity(params_base: DisasterModelParams,
                               param_name: str,
                               param_values,
                               measure: str = "Q",
                               lam_f_max: float = 0.2,
                               n_points: int = 200):
    lam_g = params_base.lam_bar_g
    lam_h = params_base.lam_bar_h
    lam_f_vals = np.linspace(1e-6, lam_f_max, n_points)

    measure = measure.upper()
    if measure not in ["P", "Q", "Q*"]:
        raise ValueError("measure must be 'P', 'Q', or 'Q*'")

    plt.figure()
    for val in param_values:
        params = params_base.__class__(**{**params_base.__dict__, param_name: val, "b_sdf": None})
        if measure == "P":
            rho_vals = np.array([_rho_P(params, lam_h, lf, lam_g) for lf in lam_f_vals])
            label = rf"{param_name}={val}, $\rho^\mathbb{{P}}$"
        elif measure == "Q":
            rho_vals = np.array([_rho_Q(params, lam_h, lf, lam_g) for lf in lam_f_vals])
            label = rf"{param_name}={val}, $\rho^\mathbb{{Q}}$"
        else:
            rho_vals = np.array([_rho_Q_star(params, lam_h, lf, lam_g) for lf in lam_f_vals])
            label = rf"{param_name}={val}, $\rho^{{\mathbb{{Q}}^*}}$"

        plt.plot(lam_f_vals, rho_vals, label=label)

    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    plt.xlabel(r"Foreign disaster intensity $\lambda_t^f$")
    plt.ylabel(r"Instantaneous corr. with $h_t^*$")
    plt.title(rf"Sensitivity of FX–hazard correlation to {param_name} ({measure})")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/Users/aryaman/honors/draft2/figures/corr_sensitivity_{param_name}_{measure.replace('*', 'star')}.png")
    plt.show()

if __name__ == "__main__":
    params = DisasterModelParams()
    plot_rho_vs_lambda_f(params)
    plot_rho_vs_lambda_g(params)
    plot_rho_vs_lambda_h(params)
    plot_rho_param_sensitivity(params, "gamma", [2.0, 4.0, 8.0], measure="Q*")
    plot_rho_param_sensitivity(params, "v", [0, 0.05, 0.1], measure="Q*")
    plot_rho_param_sensitivity(params, "Z", [-0.1, -0.2, -0.5], measure="Q*")
