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


def stable_delta(phi, delta, tau):
    exp_neg = np.exp(-delta * tau)
    return (delta - phi) + (phi + delta) * exp_neg


def stable_log_term(phi, delta, tau):
    d = stable_delta(phi, delta, tau)
    return (delta - phi) * tau + 2.0 * (np.log(2.0 * delta) - np.log(d) - delta * tau)


def stable_frac_exp(delta, tau):
    u = delta * tau
    if u > 50.0:
        return 1.0
    return (np.exp(u) - 1.0) / np.exp(u)


def stable_delta_plusminus(delta, kappa, tau):
    exp_neg = np.exp(-delta * tau)
    return (delta + kappa) + (delta - kappa) * exp_neg


def stable_log_term_plus(delta, kappa, tau):
    d = stable_delta_plusminus(delta, kappa, tau)
    return (delta + kappa) * tau + 2.0 * (np.log(2.0 * delta) - np.log(d) - delta * tau)


def stable_frac_exp_plus(delta, tau):
    u = delta * tau
    if u > 50.0:
        return 1.0
    return (np.exp(u) - 1.0) / np.exp(u)


def trig_b(phi, sig2, delta_trig, tau):
    a0 = np.arctan(phi / delta_trig)
    ang = 0.5 * delta_trig * tau + a0
    return (delta_trig * np.tan(ang) - phi) / sig2


def trig_log_term(phi, delta_trig, tau):
    a0 = np.arctan(phi / delta_trig)
    ang = 0.5 * delta_trig * tau + a0
    c0 = np.cos(a0)
    c1 = np.cos(ang)
    return (-phi) * tau - 2.0 * np.log(c1 / c0)


def delta_base_sq(params, b):
    sig2 = params.sigma_lambda ** 2
    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
    phi = b * sig2 - params.kappa
    return phi * phi - 2.0 * sig2 * K0


def delta_i_sq(params, b, A_i):
    sig2 = params.sigma_lambda ** 2
    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
    phi = b * sig2 - params.kappa
    return phi * phi - 2.0 * sig2 * (K0 - A_i)


def classify_branch(delta_sq):
    if delta_sq > 0.0:
        return "exponential"
    if delta_sq < 0.0:
        return "trigonometric"
    return "degenerate"


def check_parameter_admissibility(params, tau_probe=100.0):
    beta = float(params.beta)
    kappa = float(params.kappa)
    sigl = float(params.sigma_lambda)
    sig2 = sigl ** 2
    gamma = float(params.gamma)
    Z = float(params.Z)
    R = float(params.R)
    eta_f = float(params.eta1)
    eta_g = float(params.eta2)

    failures = []

    def req(cond, msg):
        if not cond:
            failures.append(msg)

    req(np.isfinite(beta) and beta > 0.0, f"beta must be > 0. Got {beta}")
    req(np.isfinite(kappa) and kappa > 0.0, f"kappa must be > 0. Got {kappa}")
    req(np.isfinite(sigl) and sigl > 0.0, f"sigma_lambda must be > 0. Got {sigl}")
    req(np.isfinite(gamma) and gamma > 0.0, f"gamma must be > 0. Got {gamma}")
    req(np.isfinite(Z) and Z < 0.0, f"Z must be < 0. Got {Z}")
    req(np.isfinite(R) and 0.0 <= R <= 1.0, f"R must be in [0,1]. Got {R}")
    req(np.isfinite(eta_f) and eta_f >= 0.0, f"eta1 must be >= 0. Got {eta_f}")
    req(np.isfinite(eta_g) and eta_g >= 0.0, f"eta2 must be >= 0. Got {eta_g}")

    D0 = np.exp((1.0 - gamma) * Z) - 1.0
    disc_b = (beta + kappa) ** 2 - 2.0 * sig2 * D0
    req(np.isfinite(disc_b) and disc_b > 0.0, f"b discriminant must be > 0. Got {disc_b}")

    params.compute_b_sdf()
    b = getattr(params, "b_sdf", None)
    req(b is not None and np.isfinite(b), f"b_sdf must be finite. Got {b}")

    if failures:
        print("\nPARAMETER ADMISSIBILITY FAILURE\n")
        for f in failures:
            print(" -", f)
        print("\nParameter values:")
        for attr in vars(params):
            print(f"{attr} =", getattr(params, attr))
        raise ValueError("Model parameter restrictions violated.")

    b = float(b)
    phi = b * sig2 - kappa
    req(np.isfinite(phi), f"phi must be finite. Got {phi}")
    req(phi < 0.0, f"phi=b*sigma_lambda^2-kappa must be < 0. Got phi={phi}")

    K0 = np.exp(-gamma * Z) * (1.0 - np.exp(Z))
    req(np.isfinite(K0) and K0 > 0.0, f"K0 must be finite and > 0. Got {K0}")

    Ah = np.exp(-gamma * Z) * (np.exp(Z) - 1.0)
    Ag_def = Ah + (1.0 - R) * eta_g
    Af_def = Ah + (1.0 - R) * eta_f
    Af_quanto = (1.0 - R) * eta_f

    disc_base = phi * phi - 2.0 * sig2 * K0
    disc_h = phi * phi - 2.0 * sig2 * (K0 - Ah)
    disc_g = phi * phi - 2.0 * sig2 * (K0 - Ag_def)
    disc_f = phi * phi - 2.0 * sig2 * (K0 - Af_def)

    disc_qf = kappa * kappa + 2.0 * sig2 * Af_quanto
    req(np.isfinite(disc_qf) and disc_qf > 0.0, f"tilde_delta_f^2 must be > 0. Got {disc_qf}")

    branches = {
        "base (RF)": classify_branch(disc_base),
        "quanto h": classify_branch(disc_h),
        "def g": classify_branch(disc_g),
        "def f": classify_branch(disc_f),
        "quanto f": "exponential",
    }

    print("\nBranch requirements from parameters:")
    for k, v in branches.items():
        print(f" - {k}: {v}")

    def denom_checks(label, delta2):
        if not (np.isfinite(delta2) and delta2 > 0.0):
            return
        delta = float(np.sqrt(delta2))
        d0 = stable_delta(phi, delta, 0.0)
        dT = stable_delta(phi, delta, float(tau_probe))
        req(np.isfinite(d0) and d0 > 0.0, f"{label} denom must be >0 at tau=0. Got {d0}")
        req(np.isfinite(dT) and dT > 0.0, f"{label} denom must be >0 at tau={tau_probe}. Got {dT}")

    denom_checks("base", disc_base)
    denom_checks("h", disc_h)
    denom_checks("g", disc_g)
    denom_checks("f", disc_f)

    if failures:
        print("\nPARAMETER ADMISSIBILITY FAILURE\n")
        for f in failures:
            print(" -", f)
        print("\nParameter values:")
        for attr in vars(params):
            print(f"{attr} =", getattr(params, attr))
        raise ValueError("Model parameter restrictions violated.")


def a_star(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2)
    lam_bar = params.lam_bar_g + params.lam_bar_f

    disc = delta_base_sq(params, b)
    if disc > 0.0:
        delta = float(np.sqrt(disc))
        term = stable_log_term(phi, delta, tau)
    elif disc < 0.0:
        delta_trig = float(np.sqrt(-disc))
        term = trig_log_term(phi, delta_trig, tau)
    else:
        term = (-phi) * tau

    return -A0 * tau + (params.kappa * lam_bar / sig2) * term


def b_star(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
    disc = delta_base_sq(params, b)

    if disc > 0.0:
        delta = float(np.sqrt(disc))
        d = stable_delta(phi, delta, tau)
        frac = stable_frac_exp(delta, tau)
        num = 2.0 * K0 * frac
        return num / d

    if disc < 0.0:
        delta_trig = float(np.sqrt(-disc))
        return trig_b(phi, sig2, delta_trig, tau) * sig2

    return (K0 * tau) / (1.0 - phi * tau)


def a_dom(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2)
    lam_bar = params.lam_bar_g + params.lam_bar_h

    disc = delta_base_sq(params, b)
    if disc > 0.0:
        delta = float(np.sqrt(disc))
        term = stable_log_term(phi, delta, tau)
    elif disc < 0.0:
        delta_trig = float(np.sqrt(-disc))
        term = trig_log_term(phi, delta_trig, tau)
    else:
        term = (-phi) * tau

    return -A0 * tau + (params.kappa * lam_bar / sig2) * term


def defaultable_foreign_coeffs(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2) + (1.0 - params.R) * params.h0_star
    Af = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0) + (1.0 - params.R) * params.eta1
    Ag = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0) + (1.0 - params.R) * params.eta2

    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))

    disc_f = delta_i_sq(params, b, Af)
    disc_g = delta_i_sq(params, b, Ag)

    if disc_f > 0.0:
        delta_f = float(np.sqrt(disc_f))
        df = stable_delta(phi, delta_f, tau)
        frac_f = stable_frac_exp(delta_f, tau)
        bf = 2.0 * (K0 - Af) * frac_f / df
        term_f = stable_log_term(phi, delta_f, tau)
    else:
        delta_trig = float(np.sqrt(-disc_f))
        bf = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_f = trig_log_term(phi, delta_trig, tau)

    if disc_g > 0.0:
        delta_g = float(np.sqrt(disc_g))
        dg = stable_delta(phi, delta_g, tau)
        frac_g = stable_frac_exp(delta_g, tau)
        bg = 2.0 * (K0 - Ag) * frac_g / dg
        term_g = stable_log_term(phi, delta_g, tau)
    else:
        delta_trig = float(np.sqrt(-disc_g))
        bg = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_g = trig_log_term(phi, delta_trig, tau)

    a = -A0 * tau
    a += (params.kappa * params.lam_bar_f / sig2) * term_f
    a += (params.kappa * params.lam_bar_g / sig2) * term_g

    return a, bf, bg


def quanto_coeffs(params, tau):
    params.compute_b_sdf()
    b = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi = b * sig2 - params.kappa

    A0 = params.beta + params.mu - params.gamma * (params.sigma_c ** 2) + (1.0 - params.R) * params.h0_star
    Ah = np.exp(-params.gamma * params.Z) * (np.exp(params.Z) - 1.0)
    Ag = Ah + (1.0 - params.R) * params.eta2
    Af = (1.0 - params.R) * params.eta1

    K0 = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))

    disc_h = delta_i_sq(params, b, Ah)
    disc_g = delta_i_sq(params, b, Ag)

    if disc_h > 0.0:
        delta_h = float(np.sqrt(disc_h))
        dh = stable_delta(phi, delta_h, tau)
        frac_h = stable_frac_exp(delta_h, tau)
        bh = 2.0 * (K0 - Ah) * frac_h / dh
        term_h = stable_log_term(phi, delta_h, tau)
    else:
        delta_trig = float(np.sqrt(-disc_h))
        bh = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_h = trig_log_term(phi, delta_trig, tau)

    if disc_g > 0.0:
        delta_g = float(np.sqrt(disc_g))
        dg = stable_delta(phi, delta_g, tau)
        frac_g = stable_frac_exp(delta_g, tau)
        bg = 2.0 * (K0 - Ag) * frac_g / dg
        term_g = stable_log_term(phi, delta_g, tau)
    else:
        delta_trig = float(np.sqrt(-disc_g))
        bg = trig_b(phi, sig2, delta_trig, tau) * sig2
        term_g = trig_log_term(phi, delta_trig, tau)

    delta_f = float(np.sqrt(params.kappa ** 2 + 2.0 * sig2 * Af))
    df = stable_delta_plusminus(delta_f, params.kappa, tau)
    frac_f = stable_frac_exp_plus(delta_f, tau)
    bf = -2.0 * (1.0 - params.R) * params.eta1 * frac_f / df
    term_f = stable_log_term_plus(delta_f, params.kappa, tau)

    a = -A0 * tau
    a += (params.kappa * params.lam_bar_h / sig2) * term_h
    a += (params.kappa * params.lam_bar_g / sig2) * term_g
    a += (params.kappa * params.lam_bar_f / sig2) * term_f

    return a, bh, bg, bf


def yields_and_spreads(params, tau_grid):
    lam_h = params.lam_bar_h
    lam_g = params.lam_bar_g
    lam_f = params.lam_bar_f

    y_star = np.empty_like(tau_grid, dtype=float)
    y_dom = np.empty_like(tau_grid, dtype=float)
    y_D = np.empty_like(tau_grid, dtype=float)
    y_q = np.empty_like(tau_grid, dtype=float)

    s_star = np.empty_like(tau_grid, dtype=float)
    s_q = np.empty_like(tau_grid, dtype=float)
    q_basis = np.empty_like(tau_grid, dtype=float)

    for i, tau in enumerate(tau_grid):
        a1 = a_star(params, tau)
        b1 = b_star(params, tau)
        logB_star = a1 + b1 * (lam_f + lam_g)
        y_star[i] = -logB_star / tau

        a2 = a_dom(params, tau)
        b2 = b_star(params, tau)
        logB_dom = a2 + b2 * (lam_h + lam_g)
        y_dom[i] = -logB_dom / tau

        a3, bf, bg = defaultable_foreign_coeffs(params, tau)
        logB_D = a3 + bf * lam_f + bg * lam_g
        y_D[i] = -logB_D / tau

        a4, bh, bgq, bfq = quanto_coeffs(params, tau)
        logB_q = a4 + bh * lam_h + bgq * lam_g + bfq * lam_f
        y_q[i] = -logB_q / tau

        s_star[i] = y_D[i] - y_star[i]
        s_q[i] = y_q[i] - y_dom[i]
        q_basis[i] = y_q[i] - y_D[i]

    return y_star, y_dom, y_D, y_q, s_star, s_q, q_basis


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
        B_star_vals.append(np.exp(a1 + b1 * (lam_f + lam_g)))

        a2 = a_dom(params, tau)
        b2 = b_star(params, tau)
        B_dom_vals.append(np.exp(a2 + b2 * (lam_h + lam_g)))

        a3, bf, bg = defaultable_foreign_coeffs(params, tau)
        B_D_vals.append(np.exp(a3 + bf * lam_f + bg * lam_g))

        a4, bh, bgq, bfq = quanto_coeffs(params, tau)
        B_q_vals.append(np.exp(a4 + bh * lam_h + bgq * lam_g + bfq * lam_f))

    return (
        np.array(B_star_vals),
        np.array(B_dom_vals),
        np.array(B_D_vals),
        np.array(B_q_vals),
    )


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    tau_grid = np.linspace(0.25, 30.0, 800)
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

    y_star, y_dom, y_D, y_q, s_star, s_q, q_basis = yields_and_spreads(params, tau_grid)

    plt.figure(figsize=(9, 5))
    plt.plot(tau_grid, 100.0 * y_star, linewidth=2.2, label="y* (Foreign RF)")
    plt.plot(tau_grid, 100.0 * y_dom, linewidth=2.2, label="y (Domestic RF)")
    plt.plot(tau_grid, 100.0 * y_D, linewidth=2.2, linestyle="--", label="y_D* (Foreign Risky)")
    plt.plot(tau_grid, 100.0 * y_q, linewidth=2.2, linestyle="--", label="y~ (Quanto)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "yields.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(tau_grid, 1e4 * s_star, linewidth=2.2, label="s* (Foreign Credit Spread)")
    plt.plot(tau_grid, 1e4 * s_q, linewidth=2.2, label="s~ (Quanto Credit Spread)")
    plt.plot(tau_grid, 1e4 * q_basis, linewidth=2.2, label="q~ (Quanto Basis)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Spread (bp)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "spreads.png"), dpi=220)
    plt.close()

    tail = 10
    print("\nTail yields (%), last", tail, "points:")
    print("y* :", (100.0 * y_star[-tail:]).tolist())
    print("y  :", (100.0 * y_dom[-tail:]).tolist())
    print("yD*:", (100.0 * y_D[-tail:]).tolist())
    print("y~ :", (100.0 * y_q[-tail:]).tolist())
    print("Tail spreads (bp), last", tail, "points:")
    print("s* :", (1e4 * s_star[-tail:]).tolist())
    print("s~ :", (1e4 * s_q[-tail:]).tolist())
    print("q~ :", (1e4 * q_basis[-tail:]).tolist())


if __name__ == "__main__":
    main()