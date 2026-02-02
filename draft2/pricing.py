import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from scipy.integrate import solve_ivp

from params.modelparams import DisasterModelParams


# ---------------------------------------------------------------------
# 1. RISK-FREE FOREIGN BOND (Du 2010)
# ---------------------------------------------------------------------

def get_riskfree_coeffs(params: DisasterModelParams, tau: float):
    """
    Foreign risk-free ZCB: B^*(t, T) = exp(a_rf(τ) + b_rf(τ) * λ_t^*),
    where λ_t^* = λ_t^f + λ_t^g under the foreign short rate.
    This uses the symmetric Du (2010) structure.
    """
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

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0])
    a_tau, b_tau = sol.y[:, -1]
    return a_tau, b_tau


# ---------------------------------------------------------------------
# 2. FOREIGN DEFAULTABLE BOND (CREDIT RISK IN FOREIGN CURRENCY)
# ---------------------------------------------------------------------

def hazard_affine_coeffs(params: DisasterModelParams):
    """
    Affine decomposition of r_t^* + (1-R) h_t^* under Q^*:

        r_t^* + (1-R) h_t^*
        = A0
          + Af λ_t^f
          + Ag λ_t^g

    with λ_t^* = λ_t^f + λ_t^g and
    h_t^* = h0_star + eta1 λ_t^f + eta2 λ_t^g.

    This matches the A_0, A_f, A_g in the thesis for B_D^*(t, T).
    """
    params.compute_b_sdf()
    bbar = params.b_sdf
    # C = e^{b v - γ Z} (e^Z - 1)
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
    """
    Foreign defaultable ZCB (in foreign currency):

        B_D^*(t, T) = exp(a_D(τ) + b_f(τ) λ_t^f + b_g(τ) λ_t^g)

    under the foreign risk-neutral measure Q^*.
    Symmetric treatment of λ^f and λ^g as in Proposition 3.3.
    """
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

        da = (
            params.kappa
            * (params.lam_bar_f * b_f + params.lam_bar_g * b_g)
            - A0
        )

        return [da, db_f, db_g]

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0, 0.0])
    a_tau, b_f_tau, b_g_tau = sol.y[:, -1]
    return a_tau, b_f_tau, b_g_tau


# ---------------------------------------------------------------------
# 3. QUANTO DEFAULTABLE BOND (ALWAYS DENOMINATED IN DOMESTIC CURRENCY)
# ---------------------------------------------------------------------

def quanto_affine_coeffs(params: DisasterModelParams):
    """
    Affine decomposition of the DISCOUNT RATE for the quanto bond under Q:

        r_t + (1-R) h_t^*
        = A0_q
          + Ah_q λ_t^h
          + Ag_q λ_t^g
          + Af_q λ_t^f

    where:
        r_t   = β + μ - γ σ_c^2 + (λ_t^h + λ_t^g) * C,
        h_t^* = h0_star + eta1 λ_t^f + eta2 λ_t^g,
        C     = e^{b v - γ Z} (e^Z - 1).

    Hence:
        A0_q = β + μ - γ σ_c^2 + (1-R) h0_star
        Ah_q = C
        Ag_q = C + (1-R) η2
        Af_q = (1-R) η1
    """
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
    """
    Domestic-currency defaultable quanto ZCB:

        \tilde B_D^*(t, T)
        = exp(a_q(τ) + b_h(τ) λ_t^h + b_g(τ) λ_t^g + b_f(τ) λ_t^f),

    under the DOMESTIC risk-neutral measure Q.

    Crucial asymmetry:
      - λ^h, λ^g intensities are tilted under Q by e^{b v - γ Z}.
      - λ^f is NOT tilted under Q (see Appendix A3), so its jump term
        uses the physical intensity in the FK equation. This is what
        generates the quanto effect and encodes the priced covariance
        between FX and default.
    """
    params.compute_b_sdf()
    bbar = params.b_sdf
    sigma_l = params.sigma_lambda
    kappa = params.kappa
    v = params.v

    A0_q, Ah_q, Ag_q, Af_q, C = quanto_affine_coeffs(params)

    def ode(tau_local, y):
        a, b_h, b_g, b_f = y

        # λ^h : tilted under Q, contributes to r_t only (no hazard term).
        db_h = (
            (bbar * sigma_l ** 2 - kappa - v) * b_h
            + 0.5 * sigma_l ** 2 * b_h ** 2
            + np.exp(bbar * v - params.gamma * params.Z) * (np.exp(b_h * v) - np.exp(params.Z))
            - Ah_q
        )

        # λ^g : tilted under Q, enters both r_t and h_t^*.
        db_g = (
            (bbar * sigma_l ** 2 - kappa - v) * b_g
            + 0.5 * sigma_l ** 2 * b_g ** 2
            + np.exp(bbar * v - params.gamma * params.Z) * (np.exp(b_g * v) - np.exp(params.Z))
            - Ag_q
        )

        # λ^f : NOT tilted under Q, only enters via (1-R) h_t^*.
        # Jump term uses physical intensity:
        #   λ^f [exp(b_f v) - 1],
        # and discount contributes -Af_q λ^f.
        db_f = (
            (-kappa - v) * b_f
            + 0.5 * sigma_l ** 2 * b_f ** 2
            + (np.exp(b_f * v) - 1.0)
            - Af_q
        )

        # Constant term (time-homogeneous Markov structure)
        da = (
            kappa
            * (
                params.lam_bar_h * b_h
                + params.lam_bar_g * b_g
                + params.lam_bar_f * b_f
            )
            - A0_q
        )

        return [da, db_h, db_g, db_f]

    sol = solve_ivp(ode, (0.0, tau), y0=[0.0, 0.0, 0.0, 0.0])
    a_tau, b_h_tau, b_g_tau, b_f_tau = sol.y[:, -1]
    return a_tau, b_h_tau, b_g_tau, b_f_tau


# ---------------------------------------------------------------------
# 4. PRICE / YIELD HELPERS
# ---------------------------------------------------------------------

def compute_bond_prices_vs_lambda_f(
    params: DisasterModelParams,
    tau: float,
    lam_f_vals: np.ndarray,
    lam_g_fixed: float = None,
):
    """
    Compute risk-free (foreign) and foreign defaultable ZCB prices as functions
    of the foreign country-specific disaster intensity λ^f, holding λ^g fixed.
    """
    if lam_g_fixed is None:
        lam_g_fixed = params.lam_bar_g

    a_rf, b_rf = get_riskfree_coeffs(params, tau)
    a_d, b_f, b_g = get_defaultable_coeffs(params, tau)

    lambda_star_vals = lam_f_vals + lam_g_fixed

    B_rf_vals = np.exp(a_rf + b_rf * lambda_star_vals)
    B_d_vals = np.exp(a_d + b_f * lam_f_vals + b_g * lam_g_fixed)

    return B_rf_vals, B_d_vals


def compute_quanto_bond_prices_vs_lambda_f(
    params: DisasterModelParams,
    tau: float,
    lam_f_vals: np.ndarray,
    lam_g_fixed: float = None,
    lam_h_fixed: float = None,
):
    """
    Compute domestic-currency quanto defaultable ZCB prices as functions
    of λ^f, holding λ^g and λ^h fixed.
    """
    if lam_g_fixed is None:
        lam_g_fixed = params.lam_bar_g
    if lam_h_fixed is None:
        lam_h_fixed = params.lam_bar_h

    a_q, b_h, b_g, b_f = get_quanto_defaultable_coeffs(params, tau)
    B_q_vals = np.exp(
        a_q + b_h * lam_h_fixed + b_g * lam_g_fixed + b_f * lam_f_vals
    )
    return B_q_vals


def compute_yields_from_prices(
    B_rf_vals: np.ndarray,
    B_d_vals: np.ndarray,
    tau: float,
):
    """
    Convert price arrays to annualized yields and credit spread in percentage.
    """
    y_rf = -np.log(B_rf_vals) / tau * 100.0
    y_d = -np.log(B_d_vals) / tau * 100.0
    spread = y_d - y_rf
    return y_rf, y_d, spread


# ---------------------------------------------------------------------
# 5. PLOTS: ORIGINAL
# ---------------------------------------------------------------------

def plot_rf_vs_defaultable(params: DisasterModelParams, tau: float = 5.0):
    lam_f_vals = np.linspace(0.0, 0.2, 200)
    B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(params, tau, lam_f_vals)

    plt.figure()
    plt.plot(lam_f_vals, B_rf_vals, label="Risk-free ZCB")
    plt.plot(lam_f_vals, B_d_vals, label="Defaultable ZCB")
    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Bond price")
    plt.title(
        rf"Risk-free vs defaultable bond ($\tau = {tau}$ years, $\lambda^g$ fixed)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/rf_vs_defaultable_prices.png"
    )
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
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/defaultable_heatmap.png"
    )
    plt.show()


def plot_gamma_sensitivity(params_base: DisasterModelParams, tau: float = 5.0):
    lam_f_vals = np.linspace(0.0, 0.2, 200)
    lam_g_fixed = params_base.lam_bar_g
    gammas = [2.0, 4.0, 8.0]

    plt.figure()
    for g in gammas:
        p = replace(params_base, gamma=g, b_sdf=None)
        B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(
            p, tau, lam_f_vals, lam_g_fixed
        )
        plt.plot(lam_f_vals, B_d_vals, label=rf"$\gamma={g}$")

    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Defaultable bond price")
    plt.title(rf"Sensitivity to risk aversion $\gamma$ ($\tau={tau}$ years)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/defaultable_gamma_sensitivity.png"
    )
    plt.show()


def plot_yields_and_spread_vs_lambda_f(
    params: DisasterModelParams,
    tau: float = 5.0,
    lam_f_max: float = 0.1,
):
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
    ax1.plot(
        lam_f_vals,
        spread,
        label=r"Yield spread (%)",
        color="black",
        alpha=0.7,
        linestyle="--",
    )
    ax1.plot(
        lam_f_vals,
        np.zeros(len(lam_f_vals)),
        color="black",
        alpha=0.7,
        linewidth=0.7,
    )
    ax1.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/yields_and_spread_vs_lambda_f.png"
    )
    plt.show()


def plot_param_sensitivity_defaultable(
    params_base: DisasterModelParams,
    tau: float,
    param_name: str,
    param_values,
    lam_f_max: float = 0.1,
):
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)
    lam_g_fixed = params_base.lam_bar_g

    plt.figure()
    for val in param_values:
        p = replace(params_base, **{param_name: val}, b_sdf=None)
        B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(
            p, tau, lam_f_vals, lam_g_fixed
        )
        if param_name not in ["R", "Z", "v"]:
            plt.plot(lam_f_vals, B_d_vals, label=rf"$\{param_name}={val}$")
        else:
            plt.plot(lam_f_vals, B_d_vals, label=rf"${param_name}={val}$")

    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel("Defaultable bond price")

    if param_name not in ["R", "Z", "v"]:
        plt.title(
            rf"Sensitivity of defaultable bond to $\{param_name}$ ($\tau={tau}$ years)"
        )
    else:
        plt.title(
            rf"Sensitivity of defaultable bond to ${param_name}$ ($\tau={tau}$ years)"
        )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"/Users/aryaman/honors/draft2/figures/defaultable_sensitivity_{param_name}.png"
    )
    plt.show()


def plot_short_rate_vs_lambda_f(params: DisasterModelParams, lam_f_max: float = 0.1):
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)

    params.compute_b_sdf()
    C_r = (
        np.exp(params.b_sdf * params.v - params.gamma * params.Z)
        * (np.exp(params.Z) - 1)
    )
    A_r = params.beta + params.mu - params.gamma * params.sigma_c ** 2
    lam_g = params.lam_bar_g
    r_vals = A_r + C_r * (lam_f_vals + lam_g)

    plt.figure()
    plt.plot(lam_f_vals, r_vals, label=r"Foreign short rate $r_t^*$")
    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel(r"Short rate $r_t^*$")
    plt.title(r"Foreign short rate $r_t^*$ as function of $\lambda^f$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/short_rate_vs_lambda_f.png"
    )
    plt.show()


def plot_short_rate_hazard_and_sum(
    params: DisasterModelParams,
    lam_f_max: float = 0.1,
):
    lam_f_vals = np.linspace(0.0, lam_f_max, 300)
    params.compute_b_sdf()

    C_r = (
        np.exp(params.b_sdf * params.v - params.gamma * params.Z)
        * (np.exp(params.Z) - 1)
    )
    A_r = params.beta + params.mu - params.gamma * params.sigma_c ** 2
    lam_g = params.lam_bar_g

    r_vals = A_r + C_r * (lam_f_vals + lam_g)
    h_vals = (
        params.h0_star + params.eta1 * lam_f_vals + params.eta2 * lam_g
    )
    adj_vals = r_vals + (1 - params.R) * h_vals

    plt.figure()
    plt.plot(lam_f_vals, r_vals, label=r"$r_t^*$", linewidth=2)
    plt.plot(
        lam_f_vals,
        (1 - params.R) * h_vals,
        label=r"$(1-R)h_t^*$",
        linewidth=2,
    )
    plt.plot(
        lam_f_vals,
        adj_vals,
        label=r"$r_t^* + (1-R)h_t^*$",
        linewidth=2,
    )

    plt.xlabel(r"Foreign disaster intensity $\lambda^f$")
    plt.ylabel(r"Rate (annualized)")
    plt.title(r"Short rate, hazard-adjusted short rate, and sum vs $\lambda^f$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/short_rate_hazard_sum_vs_lambda_f.png"
    )
    plt.show()


# ---------------------------------------------------------------------
# 6. NEW: QUANTO YIELDS AND QUANTO BASIS (GRAPHING THE QUANTO EFFECT)
# ---------------------------------------------------------------------

def plot_quanto_yields_and_basis_vs_lambda_f(
    params: DisasterModelParams,
    tau: float = 5.0,
    lam_f_max: float = 0.1,
):
    """
    Plot foreign defaultable vs domestic-currency quanto defaultable yields
    as functions of λ^f, and the quanto basis:

        q(t, T) = \tilde y_D^*(t, T) - y_D^*(t, T),

    which isolates the priced covariance between FX and default.
    """
    lam_f_vals = np.linspace(0.0, lam_f_max, 200)

    # Foreign risk-free and foreign defaultable prices/yields
    B_rf_vals, B_d_vals = compute_bond_prices_vs_lambda_f(params, tau, lam_f_vals)
    y_rf, y_d, spread_foreign = compute_yields_from_prices(
        B_rf_vals, B_d_vals, tau
    )

    # Domestic-currency quanto defaultable prices/yields
    B_q_vals = compute_quanto_bond_prices_vs_lambda_f(params, tau, lam_f_vals)
    y_q = -np.log(B_q_vals) / tau * 100.0

    # Quanto basis: domestic quanto yield minus foreign defaultable yield
    quanto_basis = y_q - y_d

    fig, ax1 = plt.subplots()
    ax1.plot(lam_f_vals, y_d, label=r"Foreign defaultable yield $y_D^*$")
    ax1.plot(
        lam_f_vals,
        y_q,
        label=r"Domestic-currency quanto yield $\tilde y_D^*$",
    )
    ax1.set_xlabel(r"Foreign disaster intensity $\lambda^f$")
    ax1.set_ylabel(r"Yield (annual %)")
    ax1.set_title(
        rf"Quanto effect: yields and quanto basis vs $\lambda^f$ ($\tau={tau}$ years)"
    )
    ax1.grid(True, alpha=0.3)

    # Secondary axis for quanto basis
    ax2 = ax1.twinx()
    ax2.plot(
        lam_f_vals,
        quanto_basis,
        label=r"Quanto basis $q(t,T)$",
        linestyle="--",
        color="black",
        alpha=0.8,
    )
    ax2.set_ylabel("Quanto basis (percentage points)")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    plt.savefig(
        "/Users/aryaman/honors/draft2/figures/quanto_yields_and_basis_vs_lambda_f.png"
    )
    plt.show()

# ---------------------------------------------------------------------
# 7. AFFINE COEFFICIENT VISUALIZATION
# ---------------------------------------------------------------------

def get_all_coeffs_vs_tau(params: DisasterModelParams, tau_grid):
    """
    Compute all affine coefficients across a grid of maturities.
    Returns dict:
        {
           "rf": {"a":[], "b": []},
           "defaultable": {"a":[], "b_f":[], "b_g":[]},
           "quanto": {"a":[], "b_h":[], "b_g":[], "b_f":[]}
        }
    """
    out = {
        "rf": {"a": [], "b": []},
        "defaultable": {"a": [], "b_f": [], "b_g": []},
        "quanto": {"a": [], "b_h": [], "b_g": [], "b_f": []},
    }

    for tau in tau_grid:
        # RF
        a_rf, b_rf = get_riskfree_coeffs(params, tau)
        out["rf"]["a"].append(a_rf)
        out["rf"]["b"].append(b_rf)

        # Defaultable
        a_d, b_f_d, b_g_d = get_defaultable_coeffs(params, tau)
        out["defaultable"]["a"].append(a_d)
        out["defaultable"]["b_f"].append(b_f_d)
        out["defaultable"]["b_g"].append(b_g_d)

        # Quanto
        a_q, b_h_q, b_g_q, b_f_q = get_quanto_defaultable_coeffs(params, tau)
        out["quanto"]["a"].append(a_q)
        out["quanto"]["b_h"].append(b_h_q)
        out["quanto"]["b_g"].append(b_g_q)
        out["quanto"]["b_f"].append(b_f_q)

    return out


# ---------------------------------------------------------------------
# PLOT: Each security’s (a, b(.)) curves separately
# ---------------------------------------------------------------------

def plot_riskfree_coeffs(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["rf"]["a"], label="a_rf(τ)")
    plt.plot(tau_grid, coeffs["rf"]["b"], label="b_rf(τ)")
    plt.title("Foreign Risk-Free Affine Coefficients")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/rf_affine_coeffs.png")
    plt.show()


def plot_defaultable_coeffs(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["defaultable"]["a"], label="a_D(τ)")
    plt.plot(tau_grid, coeffs["defaultable"]["b_f"], label="b_f(τ)")
    plt.plot(tau_grid, coeffs["defaultable"]["b_g"], label="b_g(τ)")
    plt.title("Foreign Defaultable Affine Coefficients")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/defaultable_affine_coeffs.png")
    plt.show()


def plot_quanto_coeffs(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["quanto"]["a"], label="a_q(τ)")
    plt.plot(tau_grid, coeffs["quanto"]["b_h"], label="b_h(τ)")
    plt.plot(tau_grid, coeffs["quanto"]["b_g"], label="b_g(τ)")
    plt.plot(tau_grid, coeffs["quanto"]["b_f"], label="b_f(τ)")
    plt.title("Quanto Defaultable Affine Coefficients (Domestic Currency)")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/quanto_affine_coeffs.png")
    plt.show()


# ---------------------------------------------------------------------
# 8. COMPARE LOADING ON λ^f ACROSS SECURITIES (Key Economic Graph!)
# ---------------------------------------------------------------------

def plot_loading_on_lambda_f(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["defaultable"]["b_f"], label="Foreign defaultable: b_f(τ)")
    plt.plot(tau_grid, coeffs["quanto"]["b_f"], label="Quanto (domestic-currency): b_f_q(τ)")
    plt.title("Loading on λ^f Across Securities")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient multiplying λ^f")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/loading_lambda_f.png")
    plt.show()


# ---------------------------------------------------------------------
# 9. COMPARE λ^g LOADINGS
# ---------------------------------------------------------------------

def plot_loading_on_lambda_g(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["defaultable"]["b_g"], label="Foreign defaultable: b_g(τ)")
    plt.plot(tau_grid, coeffs["quanto"]["b_g"], label="Quanto: b_g_q(τ)")
    plt.title("Loading on λ^g Across Securities")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient multiplying λ^g")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/loading_lambda_g.png")
    plt.show()


# ---------------------------------------------------------------------
# 10. COMPARE λ^h LOADINGS (Only quanto has this!)
# ---------------------------------------------------------------------

def plot_loading_on_lambda_h(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["quanto"]["b_h"], label="Quanto: b_h(τ)")
    plt.title("Quanto-Only Exposure: Loading on λ^h")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient multiplying λ^h")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/loading_lambda_h.png")
    plt.show()


# ---------------------------------------------------------------------
# 11. MASTER PLOT: ALL SECURITIES' b_f(τ) SHOWN TOGETHER
# ---------------------------------------------------------------------

def plot_bf_comparison_all(params: DisasterModelParams, max_tau = 10):
    tau_grid = np.linspace(0.01, max_tau, 400)
    coeffs = get_all_coeffs_vs_tau(params, tau_grid)

    plt.figure()
    plt.plot(tau_grid, coeffs["defaultable"]["b_f"], label="Foreign defaultable b_f")
    plt.plot(tau_grid, coeffs["quanto"]["b_f"], label="Quanto b_f_q")

    plt.title("b_f(τ) Across Securities")
    plt.xlabel("Maturity τ")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/aryaman/honors/draft2/figures/all_bf_comparison.png")
    plt.show()


# ---------------------------------------------------------------------
# MAIN (for batch figure generation)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    params = DisasterModelParams()
    tau = 5.0

    plot_quanto_yields_and_basis_vs_lambda_f(params, tau=tau)

    plot_riskfree_coeffs(params)
    plot_defaultable_coeffs(params)
    plot_quanto_coeffs(params)

    plot_loading_on_lambda_f(params)
    plot_loading_on_lambda_g(params)
    plot_loading_on_lambda_h(params)
    plot_bf_comparison_all(params)

    plot_short_rate_hazard_and_sum(params)
    plot_yields_and_spread_vs_lambda_f(params, tau=tau)
    plot_rf_vs_defaultable(params, tau=tau)
    # plot_defaultable_heatmap(params, tau=tau)
    plot_gamma_sensitivity(params, tau=tau)
    plot_short_rate_vs_lambda_f(params)

    plot_param_sensitivity_defaultable(params, tau, "beta", [0.01, 0.02, 0.03])
    plot_param_sensitivity_defaultable(params, tau, "mu", [0.005, 0.015, 0.025])
    plot_param_sensitivity_defaultable(
        params, tau, "sigma_c", [0.01, 0.02, 0.04]
    )
    plot_param_sensitivity_defaultable(params, tau, "Z", [-0.10, -0.172, -0.25])
    plot_param_sensitivity_defaultable(
        params, tau, "kappa", [0.05, 0.15, 0.3]
    )
    # plot_param_sensitivity_defaultable(params, tau, "sigma_lambda", [0.05, 0.1, 0.2])
    plot_param_sensitivity_defaultable(
        params, tau, "v", [0.005, 0.01, 0.05]
    )
    plot_param_sensitivity_defaultable(
        params, tau, "R", [0.2, 0.4, 0.6]
    )
    plot_param_sensitivity_defaultable(
        params, tau, "eta1", [1, 1.5, 2, 2.5]
    )
    plot_param_sensitivity_defaultable(
        params, tau, "eta2", [1, 1.5, 2, 2.5]
    )

    
