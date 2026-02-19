import numpy as np
from scipy.integrate import solve_ivp
from params.modelparams import DisasterModelParams

def _solve_ode(ode, tau, y0):
    sol = solve_ivp(
        ode,
        (0.0, tau),
        y0=y0,
        method="Radau",
        rtol=1e-9,
        atol=1e-12,
    )
    return sol.y[:, -1]

def get_riskfree_coeffs(params: DisasterModelParams, tau: float):
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

    if tau == 0.0:
        return 0.0, 0.0

    a_tau, b_tau = _solve_ode(ode, tau, [0.0, 0.0])
    return a_tau, b_tau

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

    a_tau, b_tau = _solve_ode(ode, tau, [0.0, 0.0])
    return a_tau, b_tau

def hazard_affine_coeffs(params: DisasterModelParams):
    params.compute_b_sdf()
    bbar = params.b_sdf
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

        da = params.kappa * (
            params.lam_bar_f * b_f + params.lam_bar_g * b_g
        ) - A0

        return [da, db_f, db_g]

    if tau == 0.0:
        return 0.0, 0.0, 0.0

    a_tau, b_f_tau, b_g_tau = _solve_ode(ode, tau, [0.0, 0.0, 0.0])
    return a_tau, b_f_tau, b_g_tau

def quanto_affine_coeffs(params: DisasterModelParams):
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
    params.compute_b_sdf()
    bbar = params.b_sdf
    sigma_l = params.sigma_lambda
    kappa = params.kappa
    v = params.v

    A0_q, Ah_q, Ag_q, Af_q, C = quanto_affine_coeffs(params)

    def ode(tau_local, y):
        a, b_h, b_g, b_f = y

        db_h = (
            (bbar * sigma_l ** 2 - kappa - v) * b_h
            + 0.5 * sigma_l ** 2 * b_h ** 2
            + np.exp(bbar * v - params.gamma * params.Z)
            * (np.exp(b_h * v) - np.exp(params.Z))
            - Ah_q
        )

        db_g = (
            (bbar * sigma_l ** 2 - kappa - v) * b_g
            + 0.5 * sigma_l ** 2 * b_g ** 2
            + np.exp(bbar * v - params.gamma * params.Z)
            * (np.exp(b_g * v) - np.exp(params.Z))
            - Ag_q
        )

        db_f = (
            (-kappa - v) * b_f
            + 0.5 * sigma_l ** 2 * b_f ** 2
            + (np.exp(b_f * v) - 1.0)
            - Af_q
        )

        da = kappa * (
            params.lam_bar_h * b_h
            + params.lam_bar_g * b_g
            + params.lam_bar_f * b_f
        ) - A0_q

        return [da, db_h, db_g, db_f]

    if tau == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    a_tau, b_h_tau, b_g_tau, b_f_tau = _solve_ode(
        ode, tau, [0.0, 0.0, 0.0, 0.0]
    )
    return a_tau, b_h_tau, b_g_tau, b_f_tau


def _hit_time_to_level(F, b0, level, tau_max, method="Radau"):
    def event(t, y):
        return y[0] - level
    event.terminal = True
    event.direction = 1.0

    sol = solve_ivp(
        lambda t, y: [F(y[0])],
        (0.0, tau_max),
        y0=[b0],
        method=method,
        rtol=1e-10,
        atol=1e-13,
        max_step=0.05,
        events=event,
    )

    if sol.status == 1 and sol.t_events and len(sol.t_events[0]) > 0:
        return float(sol.t_events[0][0]), True
    return float(tau_max), False


def _integral_remaining_time(F, b_start, b_end, n=200000):
    grid = np.linspace(b_start, b_end, int(n))
    vals = np.array([F(b) for b in grid], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("F(b) not finite on integration grid.")
    if np.min(vals) <= 0.0:
        raise RuntimeError("F(b) not strictly positive on integration grid.")
    return float(np.trapz(1.0 / vals, grid))


def _estimate_blowup_time_scalar_ode(F, b0=0.0, tau_max=500.0, b_danger=200.0, b_big=4000.0):
    tau_hit, hit = _hit_time_to_level(F, b0=b0, level=b_danger, tau_max=tau_max)
    if not hit:
        return None

    F_d = float(F(b_danger))
    if not np.isfinite(F_d) or F_d <= 0.0:
        return None

    try:
        dt = _integral_remaining_time(F, b_danger, b_big, n=200000)
    except RuntimeError:
        return None

    tau_lo = tau_hit
    tau_hi = tau_hit + dt
    return tau_lo, tau_hi


def _estimate_blowup_time_riskfree(params: DisasterModelParams, domestic=False, tau_max=500.0):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)
    lam_bar_total = (params.lam_bar_h + params.lam_bar_g) if domestic else (params.lam_bar_f + params.lam_bar_g)

    def F(b):
        return (
            -(params.kappa + params.v) * b
            + bbar * params.sigma_lambda ** 2 * b
            + 0.5 * params.sigma_lambda ** 2 * b ** 2
            + C * (np.exp(b * params.v) - np.exp(params.Z))
        )

    return _estimate_blowup_time_scalar_ode(F, b0=0.0, tau_max=tau_max, b_danger=200.0, b_big=4000.0)


def _estimate_blowup_time_defaultable(params: DisasterModelParams, tau_max=500.0):
    params.compute_b_sdf()
    bbar = params.b_sdf
    C = np.exp(bbar * params.v - params.gamma * params.Z)
    A0, Af, Ag = hazard_affine_coeffs(params)

    def F_f(b):
        return (
            -(params.kappa + params.v) * b
            + bbar * params.sigma_lambda ** 2 * b
            + 0.5 * params.sigma_lambda ** 2 * b ** 2
            + C * (np.exp(b * params.v) - np.exp(params.Z))
            - Af
        )

    def F_g(b):
        return (
            -(params.kappa + params.v) * b
            + bbar * params.sigma_lambda ** 2 * b
            + 0.5 * params.sigma_lambda ** 2 * b ** 2
            + C * (np.exp(b * params.v) - np.exp(params.Z))
            - Ag
        )

    est_f = _estimate_blowup_time_scalar_ode(F_f, b0=0.0, tau_max=tau_max, b_danger=200.0, b_big=4000.0)
    est_g = _estimate_blowup_time_scalar_ode(F_g, b0=0.0, tau_max=tau_max, b_danger=200.0, b_big=4000.0)

    if est_f is None and est_g is None:
        return None

    candidates = [x for x in [est_f, est_g] if x is not None]
    tau_lo = min(x[0] for x in candidates)
    tau_hi = min(x[1] for x in candidates)
    return tau_lo, tau_hi


def _estimate_blowup_time_quanto(params: DisasterModelParams, tau_max=500.0):
    params.compute_b_sdf()
    bbar = params.b_sdf
    sigma_l = params.sigma_lambda
    kappa = params.kappa
    v = params.v
    A0_q, Ah_q, Ag_q, Af_q, C0 = quanto_affine_coeffs(params)

    def F_h(b):
        return (
            (bbar * sigma_l ** 2 - kappa - v) * b
            + 0.5 * sigma_l ** 2 * b ** 2
            + np.exp(bbar * v - params.gamma * params.Z) * (np.exp(b * v) - np.exp(params.Z))
            - Ah_q
        )

    def F_g(b):
        return (
            (bbar * sigma_l ** 2 - kappa - v) * b
            + 0.5 * sigma_l ** 2 * b ** 2
            + np.exp(bbar * v - params.gamma * params.Z) * (np.exp(b * v) - np.exp(params.Z))
            - Ag_q
        )

    def F_f(b):
        return (
            (-kappa - v) * b
            + 0.5 * sigma_l ** 2 * b ** 2
            + (np.exp(b * v) - 1.0)
            - Af_q
        )

    est_h = _estimate_blowup_time_scalar_ode(F_h, b0=0.0, tau_max=tau_max, b_danger=200.0, b_big=4000.0)
    est_g = _estimate_blowup_time_scalar_ode(F_g, b0=0.0, tau_max=tau_max, b_danger=200.0, b_big=4000.0)
    est_f = _estimate_blowup_time_scalar_ode(F_f, b0=0.0, tau_max=tau_max, b_danger=200.0, b_big=4000.0)

    if est_h is None and est_g is None and est_f is None:
        return None

    candidates = [x for x in [est_h, est_g, est_f] if x is not None]
    tau_lo = min(x[0] for x in candidates)
    tau_hi = min(x[1] for x in candidates)
    return tau_lo, tau_hi


def print_blowup_estimates(params: DisasterModelParams, tau_max=500.0):
    rf_star = _estimate_blowup_time_riskfree(params, domestic=False, tau_max=tau_max)
    rf_dom = _estimate_blowup_time_riskfree(params, domestic=True, tau_max=tau_max)
    def_star = _estimate_blowup_time_defaultable(params, tau_max=tau_max)
    quanto = _estimate_blowup_time_quanto(params, tau_max=tau_max)

    def _fmt(x):
        if x is None:
            return f"> {tau_max:.1f}y (no reliable blow-up bracket)"
        lo, hi = x
        return f"[{lo:.6f}, {hi:.6f}] years"

    print("Estimated blow-up maturity ranges (years):")
    print(f"  y*(tau)   (foreign risk-free):   {_fmt(rf_star)}")
    print(f"  y(tau)    (domestic risk-free):  {_fmt(rf_dom)}")
    print(f"  y_D*(tau) (foreign defaultable): {_fmt(def_star)}")
    print(f"  y~(tau)   (quanto):              {_fmt(quanto)}")


if __name__ == "__main__":
    params = DisasterModelParams()
    print_blowup_estimates(params, tau_max=500.0)
