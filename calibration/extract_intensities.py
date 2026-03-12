import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from params.modelparams import DisasterModelParams

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

FIGDIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = "/Users/aryaman/honors/calibration/data"


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
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    return tau_grid, sol.y[1], sol.y[0]


def simulate_cir_mean_log_pd(params: DisasterModelParams, log_pd_func, years=200000, dt=1 / 12, seed=12345):
    n = int(years / dt)
    lam_bar = params.lam_bar_f + params.lam_bar_g
    lam = lam_bar
    rng = np.random.default_rng(seed)
    burn = min(int(500 / dt), max(1000, n // 20))
    vals = np.empty(n - burn)

    for t in range(n):
        lam_pos = max(lam, 0.0)
        z = rng.standard_normal()
        lam = (
            lam
            + params.kappa * (lam_bar - lam_pos) * dt
            + params.sigma_lambda * np.sqrt(lam_pos) * np.sqrt(dt) * z
        )
        lam = max(lam, 0.0)
        if t >= burn:
            vals[t - burn] = log_pd_func(lam)

    return float(np.mean(vals))


def build_log_pd_mapping(params: DisasterModelParams, tau, aphi, bphi, lambda_max=0.15, n_lambda=2000):
    lambda_grid = np.linspace(0.0, lambda_max, n_lambda)

    expo = aphi[:, None] + np.outer(bphi, lambda_grid)
    expo = np.clip(expo, -745.0, 700.0)
    model_pd_vals = np.trapz(np.exp(expo), tau, axis=0)

    if np.any(~np.isfinite(model_pd_vals)) or np.any(model_pd_vals <= 0.0):
        raise RuntimeError("Non-finite or non-positive price-dividend values encountered.")

    log_model_pd = np.log(model_pd_vals)

    if not np.all(np.diff(log_model_pd) < 0):
        diffs = np.diff(log_model_pd)
        if np.any(diffs >= 0):
            raise RuntimeError("Model log price-dividend mapping is not strictly decreasing in lambda.")

    interp = PchipInterpolator(lambda_grid, log_model_pd, extrapolate=False)

    return lambda_grid, log_model_pd, interp


def invert_log_pd_targets(targets, lambda_grid, log_model_pd, interp):
    lower_lambda = float(lambda_grid[0])
    upper_lambda = float(lambda_grid[-1])
    lower_logpd = float(log_model_pd[0])
    upper_logpd = float(log_model_pd[-1])

    implied = np.empty_like(targets, dtype=float)

    for i, target in enumerate(targets):
        if not np.isfinite(target):
            implied[i] = np.nan
        elif target >= lower_logpd:
            implied[i] = lower_lambda
        elif target <= upper_logpd:
            implied[i] = upper_lambda
        else:
            root_fn = lambda lam: float(interp(lam) - target)
            implied[i] = brentq(root_fn, lower_lambda, upper_lambda, xtol=1e-12, rtol=1e-10, maxiter=200)

    return np.maximum(implied, 0.0)


def parse_shiller_dates(date_series: pd.Series):
    date_floats = date_series.astype(float).to_numpy()
    years = np.floor(date_floats).astype(int)
    months = np.round((date_floats - years) * 100).astype(int)
    months = np.clip(months, 1, 12)
    return pd.to_datetime(
        {"year": years, "month": months, "day": np.ones_like(years, dtype=int)}
    )


def build_and_save_lambda_series(output_path=os.path.join(DATA_DIR, "lambda_t_series.csv")):
    params = DisasterModelParams()

    cape_df = pd.read_csv(os.path.join(DATA_DIR, "shiller CAPE.csv"))
    cape_df["Date"] = parse_shiller_dates(cape_df["Date"])
    cape_df = cape_df.sort_values("Date").reset_index(drop=True)

    if "CAPE" not in cape_df.columns:
        raise KeyError("Input CSV must contain a 'CAPE' column.")

    cape_df = cape_df[np.isfinite(cape_df["CAPE"]) & (cape_df["CAPE"] > 0)].copy()
    cape_df["log_cape"] = np.log(cape_df["CAPE"])
    demeaned_log_cape = cape_df["log_cape"] - cape_df["log_cape"].mean()

    tau, aphi, bphi = solve_a_b(params, tau_max=250.0, n_tau=8000)
    lambda_grid, log_model_pd, interp = build_log_pd_mapping(
        params=params,
        tau=tau,
        aphi=aphi,
        bphi=bphi,
        lambda_max=0.15,
        n_lambda=2500,
    )

    model_mean_log_pd = simulate_cir_mean_log_pd(
        params=params,
        log_pd_func=lambda lam: float(interp(min(max(lam, lambda_grid[0]), lambda_grid[-1]))),
        years=200000,
        dt=1 / 12,
        seed=12345,
    )

    target_log_pd = demeaned_log_cape.to_numpy() + model_mean_log_pd
    cape_df["lambda_t"] = invert_log_pd_targets(target_log_pd, lambda_grid, log_model_pd, interp)

    lambda_df = cape_df[["Date", "lambda_t"]].copy()
    lambda_df.to_csv(output_path, index=False)

    return lambda_df


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    lambda_df = build_and_save_lambda_series()

    print("Head of Lambda_t series:")
    print(lambda_df.head())
    print("\nTail of Lambda_t series:")
    print(lambda_df.tail())

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(lambda_df["Date"], lambda_df["lambda_t"], color="black", lw=1.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.title(r"Implied Disaster Intensity $\lambda_t$ from Shiller CAPE", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(r"Disaster Intensity ($\lambda_t$)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "pd_ratio.png"), dpi=300)


if __name__ == "__main__":
    main()