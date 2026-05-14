"""
extract_intensities.py
======================
Multi-country CAPE panel -> lambda^{i,total}, lambda^g (global),
and lambda^f_i (country-specific).

Global factor: PCA regression decomposition
--------------------------------------------
1. Build a date x country panel of total disaster intensities lambda^{i,total}.
2. Demean each country series before PCA (optionally standardize by std).
3. Extract PC1 as the zero-mean common global disaster-risk fluctuation factor F_t.
4. Sign-correct F_t so it is positively correlated with the cross-sectional mean.
5. Construct lambda_global as a positive affine rescaling of F_t (for the hazard
   equation only; it is NOT used in the regression). Its unconditional mean is
   anchored to params.lam_bar_g, which should be set to the structural Du/Barro
   global disaster mean.
6. OLS regression with intercept for each country:
       lambda_total_i = alpha_i + beta_i * F_t + residual_i
7. Country-specific intensity:
       lambda_country_i_t = lambda_total_i_t - beta_i * F_t
                          = alpha_i + residual_i_t
   Apply small floor:
       lambda_country_i_t = max(lambda_country_i_t, eps)
8. Save:
       processed_data/intensities.csv
       processed_data/pca_loadings.csv
       processed_data/decomposition_betas.csv  (regression + PCA diagnostics)

CLI
---
    python -m calibration.extract_intensities
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.parameters import DisasterModelParams


CAPE_PATH = os.path.join(BASE_DIR, "raw_data", "country_cape_panel.csv")
OUT_PATH = os.path.join(BASE_DIR, "processed_data", "intensities.csv")
LOADINGS_PATH = os.path.join(BASE_DIR, "processed_data", "pca_loadings.csv")
BETAS_PATH = os.path.join(BASE_DIR, "processed_data", "decomposition_betas.csv")

LAMBDA_MAX = 1.0            # upper bound of inversion grid
N_LAMBDA = 5000             # grid resolution
FLAG_THRESH = 0.30          # warn if lambda_total exceeds this annualized level
PCA_STANDARDIZE = True      # True: demean + scale by std; False: demean only
MIN_OBS_PER_COUNTRY = 36    # minimum monthly observations for PCA inclusion
MIN_OBS_FOR_OLS = 12        # minimum observations for country regressions
NONNEG_EPS = 1e-6           # strict-positive numerical floor


# ---------------------------------------------------------------------------
# P/D ODE solver
# ---------------------------------------------------------------------------

def _solve_a_b(params: DisasterModelParams, tau_max: float, n_tau: int):
    tau_grid = np.linspace(0.0, tau_max, n_tau)
    params.compute_b_sdf()

    def rhs(tau, y):
        bphi, aphi = y
        db = (
            -params.kappa * bphi
            + (0.5 * bphi * bphi + params.b_sdf * bphi)
            * (params.sigma_lambda ** 2)
            - np.exp(-params.gamma * params.Z)
            * (np.exp(params.Z) - np.exp(params.phi * params.Z))
        )
        da = (
            -(
                params.beta
                + params.mu
                - params.mu_D
                + params.gamma * (params.phi - 1.0) * (params.sigma_c ** 2)
            )
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


def _simulate_cir_mean_log_pd(
    params: DisasterModelParams,
    log_pd_func,
    years: int = 200_000,
    dt: float = 1 / 12,
    seed: int = 12345,
) -> float:
    n = int(years / dt)
    lam_bar = params.lam_bar_f + params.lam_bar_g
    lam = lam_bar
    rng = np.random.default_rng(seed)
    burn = min(int(500 / dt), max(1000, n // 20))
    vals = np.empty(n - burn)

    for t in range(n):
        lam_pos = max(lam, 0.0)
        lam = (
            lam
            + params.kappa * (lam_bar - lam_pos) * dt
            + params.sigma_lambda
            * np.sqrt(lam_pos)
            * np.sqrt(dt)
            * rng.standard_normal()
        )
        lam = max(lam, 0.0)
        if t >= burn:
            vals[t - burn] = log_pd_func(lam)

    return float(np.mean(vals))


def _build_log_pd_mapping(
    params: DisasterModelParams,
    tau: np.ndarray,
    aphi: np.ndarray,
    bphi: np.ndarray,
    lambda_max: float = LAMBDA_MAX,
    n_lambda: int = N_LAMBDA,
):
    lambda_grid = np.linspace(0.0, lambda_max, n_lambda)
    expo = aphi[:, None] + np.outer(bphi, lambda_grid)
    expo = np.clip(expo, -745.0, 700.0)

    if hasattr(np, "trapezoid"):
        model_pd = np.trapezoid(np.exp(expo), tau, axis=0)
    else:
        model_pd = np.trapz(np.exp(expo), tau, axis=0)

    if not np.all(np.isfinite(model_pd)) or not np.all(model_pd > 0):
        raise RuntimeError("Non-finite or non-positive model P/D values.")

    log_model_pd = np.log(model_pd)

    if not np.all(np.diff(log_model_pd) < 0):
        raise RuntimeError("Model log P/D mapping is not strictly decreasing in lambda.")

    interp = PchipInterpolator(lambda_grid, log_model_pd, extrapolate=False)
    return lambda_grid, log_model_pd, interp


def _invert_log_pd_targets(
    targets: np.ndarray,
    lambda_grid: np.ndarray,
    log_model_pd: np.ndarray,
    interp,
) -> np.ndarray:
    lo_lam, hi_lam = float(lambda_grid[0]), float(lambda_grid[-1])
    lo_pd, hi_pd = float(log_model_pd[0]), float(log_model_pd[-1])
    implied = np.empty_like(targets, dtype=float)
    n_at_upper = 0

    for i, tgt in enumerate(targets):
        if not np.isfinite(tgt):
            implied[i] = np.nan
        elif tgt >= lo_pd:
            implied[i] = lo_lam
        elif tgt <= hi_pd:
            implied[i] = hi_lam
            n_at_upper += 1
        else:
            implied[i] = brentq(
                lambda lam: float(interp(lam)) - tgt,
                lo_lam,
                hi_lam,
                xtol=1e-12,
                rtol=1e-10,
                maxiter=200,
            )

    if n_at_upper > 0:
        warnings.warn(
            f"{n_at_upper} observations hit the upper bound of the lambda grid "
            f"(lambda_max={hi_lam:.2f}). Consider increasing LAMBDA_MAX if "
            "this is more than a handful of extreme observations."
        )

    return np.maximum(implied, 0.0)


# ---------------------------------------------------------------------------
# Public: build P/D mapping
# ---------------------------------------------------------------------------

def compute_model_pd_mapping(
    params: DisasterModelParams,
    tau_max: float = 250.0,
    n_tau: int = 8000,
    lambda_max: float = LAMBDA_MAX,
    n_lambda: int = N_LAMBDA,
    sim_years: int = 200_000,
    sim_seed: int = 12345,
):
    """Return (lambda_grid, log_model_pd, interp, model_mean_log_pd)."""
    tau, aphi, bphi = _solve_a_b(params, tau_max, n_tau)
    lambda_grid, log_model_pd, interp = _build_log_pd_mapping(
        params, tau, aphi, bphi, lambda_max, n_lambda
    )
    model_mean_log_pd = _simulate_cir_mean_log_pd(
        params,
        log_pd_func=lambda lam: float(
            interp(min(max(lam, lambda_grid[0]), lambda_grid[-1]))
        ),
        years=sim_years,
        seed=sim_seed,
    )
    return lambda_grid, log_model_pd, interp, model_mean_log_pd


# ---------------------------------------------------------------------------
# Public: per-country lambda^{i,total}
# ---------------------------------------------------------------------------

def compute_lambda_total_for_country(
    cape_series: pd.Series,
    model_mean_log_pd: float,
    lambda_grid: np.ndarray,
    log_model_pd: np.ndarray,
    interp,
) -> pd.Series:
    """Map a country's CAPE/P-E series to lambda^{i,total}."""
    cape = cape_series.dropna()
    cape = cape[cape > 0]
    if cape.empty:
        return pd.Series(np.nan, index=cape_series.index, dtype=float)

    log_cape = np.log(cape)
    demeaned = log_cape - log_cape.mean()
    target_log_pd = demeaned.to_numpy() + model_mean_log_pd

    lambda_vals = _invert_log_pd_targets(
        target_log_pd, lambda_grid, log_model_pd, interp
    )

    result = pd.Series(np.nan, index=cape_series.index, dtype=float)
    result.loc[cape.index] = lambda_vals
    return result


# ---------------------------------------------------------------------------
# Public: PCA regression decomposition
# ---------------------------------------------------------------------------

def decompose_intensities_pca(
    lambda_total_panel: pd.DataFrame,
    standardize: bool = PCA_STANDARDIZE,
    min_obs_per_country: int = MIN_OBS_PER_COUNTRY,
    min_obs_for_ols: int = MIN_OBS_FOR_OLS,
    eps: float = NONNEG_EPS,
    target_global_mean: float | None = None,
    target_global_std: float | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Decompose lambda^{i,total} into a PCA global factor and country components.

    Returns
    -------
    lambda_g : pd.Series
        Global intensity series (positive affine rescaling of F_t for hazard eq).
    lambda_f : pd.DataFrame
        Country-specific intensity panel, date x country.
    diagnostics : dict
        PCA loadings and regression diagnostics DataFrames.
    """
    print("\n" + "=" * 60)
    print("PCA REGRESSION DECOMPOSITION")
    print("=" * 60)
    print(
        f"Input panel: {lambda_total_panel.shape[0]} dates x "
        f"{lambda_total_panel.shape[1]} countries"
    )
    print(
        f"Date range: {lambda_total_panel.index.min().date()} -> "
        f"{lambda_total_panel.index.max().date()}"
    )
    print(
        f"Standardise before PCA: {standardize}  "
        f"(demean + scale by std)" if standardize
        else f"Standardise before PCA: {standardize}  (demean only)"
    )
    print(f"Min obs per country (PCA inclusion): {min_obs_per_country}")

    # 1. Filter countries with sufficient observations.
    obs_counts = lambda_total_panel.notna().sum()
    included = obs_counts[obs_counts >= min_obs_per_country].index.tolist()
    excluded = obs_counts[obs_counts < min_obs_per_country].index.tolist()

    if excluded:
        print(
            f"\nWARNING: {len(excluded)} country/countries excluded from PCA "
            f"(< {min_obs_per_country} valid obs):"
        )
        for c in excluded:
            print(f"  {c}: {int(obs_counts[c])} obs")

    if len(included) < 2:
        raise ValueError(
            f"Only {len(included)} countries have >= {min_obs_per_country} observations. "
            "PCA requires at least 2 countries."
        )

    panel = lambda_total_panel[included].copy()
    print(f"\nPCA panel: {panel.shape[0]} dates x {len(included)} countries")
    print(f"Countries included: {included}")

    # 2. Missing data summary.
    n_missing = int(panel.isna().sum().sum())
    pct_missing = 100.0 * n_missing / panel.size
    print(f"\nMissing values in PCA panel: {n_missing} / {panel.size} ({pct_missing:.1f}%)")
    print("Valid obs per country:")
    for c in included:
        n_valid = int(panel[c].notna().sum())
        d_min = panel[c].dropna().index.min().date() if n_valid > 0 else "n/a"
        d_max = panel[c].dropna().index.max().date() if n_valid > 0 else "n/a"
        print(f"  {c}: {n_valid} obs  ({d_min} -> {d_max})")

    # 3. Impute missing values with country means for PCA only.
    col_means = panel.mean()
    col_stds = panel.std().replace(0.0, 1.0)
    panel_filled = panel.fillna(col_means)

    # 4. Demean (and optionally scale by std) before PCA.
    if standardize:
        print("\nPre-PCA: demeaning and scaling by std (standardise)")
        panel_for_pca = (panel_filled - col_means) / col_stds
    else:
        print("\nPre-PCA: demeaning only (not scaling by std)")
        panel_for_pca = panel_filled - col_means

    # 5. PCA: extract PC1.
    pca = PCA(n_components=1)
    pc_scores = pca.fit_transform(panel_for_pca.values).squeeze()
    raw_loadings = pca.components_[0].copy()
    expl_var = float(pca.explained_variance_ratio_[0])

    print(f"\nPC1 explained variance ratio: {expl_var * 100:.2f}%")
    if expl_var < 0.20:
        warnings.warn(
            f"PC1 explains only {expl_var * 100:.1f}% of total variance. "
            "The global factor may not be well-identified across countries."
        )

    # 6. Sign correction: ensure F_t is positively correlated with cross-sectional mean.
    cross_mean_series = panel_filled.mean(axis=1)
    cross_mean = cross_mean_series.to_numpy()
    corr_raw = float(np.corrcoef(pc_scores, cross_mean)[0, 1])

    if corr_raw < 0:
        pc_scores = -pc_scores
        loadings = -raw_loadings
        print(
            f"\nSign flipped: raw PC1 negatively correlated with cross-country mean "
            f"(r = {corr_raw:.3f})"
        )
    else:
        loadings = raw_loadings.copy()
        print(f"\nSign OK: PC1 positively correlated with cross-country mean (r = {corr_raw:.3f})")

    print("\nPCA loadings by country (sign-corrected):")
    for c, ld in zip(included, loadings):
        print(f"  {c:4s}: {ld:+.4f}")

    # 7. Zero-mean factor F_t (PCA scores are already zero-mean; enforce explicitly).
    F_t = pc_scores - float(pc_scores.mean())
    factor_mean = float(np.mean(F_t))
    factor_std = float(np.std(F_t))

    if not np.isfinite(factor_std) or factor_std <= 0.0:
        raise ValueError("F_t has zero or invalid standard deviation.")

    corr_factor_cross_mean = float(np.corrcoef(F_t, cross_mean)[0, 1])
    print(f"\nGlobal factor F_t (zero-mean):")
    print(f"  mean={factor_mean:.2e}  std={factor_std:.6f}")
    print(f"  min={F_t.min():.6f}  max={F_t.max():.6f}")
    print(f"  Correlation of F_t with cross-sectional mean lambda_total: {corr_factor_cross_mean:.4f}")

    # 8. lambda_global: positive affine transform of F_t for the hazard equation.
    #    The PCA factor identifies time-series variation, not its level. Anchor the
    #    unconditional mean to the structural model's global disaster mean
    #    params.lam_bar_g (Du/Barro normalization), while using the empirical
    #    cross-sectional mean's volatility as the default volatility scale.
    #    Do NOT use lambda_global in the regression—only zero-mean F_t.
    empirical_cross_mean_mean = float(cross_mean_series.mean())
    empirical_cross_mean_std = float(cross_mean_series.std())

    if target_global_mean is None:
        target_mean = empirical_cross_mean_mean
        target_mean_source = "empirical cross-sectional mean"
    else:
        target_mean = float(target_global_mean)
        target_mean_source = "structural params.lam_bar_g"

    if target_global_std is None:
        target_std = empirical_cross_mean_std
        target_std_source = "empirical cross-sectional mean std"
    else:
        target_std = float(target_global_std)
        target_std_source = "user-specified target_global_std"

    if not np.isfinite(target_mean) or target_mean <= 0.0:
        raise ValueError(f"Invalid target global mean: {target_mean}")
    if not np.isfinite(target_std) or target_std <= 0.0:
        raise ValueError(f"Invalid target global std: {target_std}")

    lambda_g_raw = (F_t / factor_std) * target_std + target_mean

    n_g_floor = int((lambda_g_raw < eps).sum())
    pct_g_floor = 100.0 * n_g_floor / len(lambda_g_raw)

    print(f"\nlambda_global (affine rescaling of F_t; for hazard equation only):")
    print(
        f"  target mean={target_mean * 100:.4f}%  ({target_mean_source})  "
        f"target std={target_std * 100:.4f}%  ({target_std_source})"
    )
    print(
        f"  empirical cross-sectional mean: mean={empirical_cross_mean_mean*100:.4f}%  "
        f"std={empirical_cross_mean_std*100:.4f}%"
    )
    print(
        f"  before floor:  min={lambda_g_raw.min()*100:.4f}%  "
        f"mean={lambda_g_raw.mean()*100:.4f}%  "
        f"max={lambda_g_raw.max()*100:.4f}%  "
        f"std={lambda_g_raw.std()*100:.4f}%"
    )

    if n_g_floor > 0:
        warnings.warn(
            f"{n_g_floor} ({pct_g_floor:.1f}%) lambda_global values below eps={eps:.0e}; "
            "flooring rather than shifting the whole series."
        )
    else:
        print("  lambda_global nonnegativity: OK (no values below eps)")

    lambda_g_vals = np.maximum(lambda_g_raw, eps)
    print(
        f"  after floor:   min={lambda_g_vals.min()*100:.4f}%  "
        f"mean={lambda_g_vals.mean()*100:.4f}%  "
        f"max={lambda_g_vals.max()*100:.4f}%  "
        f"std={lambda_g_vals.std()*100:.4f}%"
    )
    print(f"  percent floored: {pct_g_floor:.2f}%")

    ratio = lambda_g_vals / cross_mean
    ratio_finite = ratio[np.isfinite(ratio)]
    if len(ratio_finite) > 0:
        print(
            f"  lambda_global / cross-mean ratio:  "
            f"p10={np.percentile(ratio_finite, 10):.3f}  "
            f"p50={np.percentile(ratio_finite, 50):.3f}  "
            f"p90={np.percentile(ratio_finite, 90):.3f}"
        )

    lambda_g = pd.Series(lambda_g_vals, index=panel.index, name="lambda_global")

    # 9. lambda_total boundary diagnostics.
    print("\nlambda_total boundary diagnostics (near-zero or near upper grid):")
    for country in included:
        y_all = lambda_total_panel[country].dropna().to_numpy()
        if len(y_all) == 0:
            print(f"  {country:4s}: no valid observations")
            continue
        n_tot = len(y_all)
        n_lo = int((y_all <= eps).sum())
        n_hi = int((y_all >= 0.99 * LAMBDA_MAX).sum())
        print(
            f"  {country:4s}: n={n_tot}  "
            f"<=eps={n_lo}({100*n_lo/n_tot:.1f}%)  "
            f">=99%grid={n_hi}({100*n_hi/n_tot:.1f}%)  "
            f"min={y_all.min()*100:.4f}%  "
            f"med={np.median(y_all)*100:.4f}%  "
            f"max={y_all.max()*100:.4f}%"
        )

    # 10. OLS regression: lambda_total_i = alpha_i + beta_i * F_t + residual_i.
    #     lambda_country_i = lambda_total_i - beta_i * F_t  (= alpha_i + residual_i).
    print("\nOLS regression decomposition using zero-mean factor F_t:")
    print("  lambda_total_i = alpha_i + beta_i * F_t + residual_i")
    print("  lambda_country_i = lambda_total_i - beta_i * F_t  [= alpha_i + residual_i]")
    print()
    hdr = (
        f"  {'Cntry':5s}  {'alpha%':>9s}  {'beta':>8s}  "
        f"{'R2':>6s}  {'n':>5s}  {'%floor':>7s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    loading_by_country = dict(zip(included, loadings))
    diag_records: list[dict] = []
    lambda_f_dict: dict[str, np.ndarray] = {}

    for country in included:
        y = lambda_total_panel[country].values
        valid = np.isfinite(y)
        n_valid = int(valid.sum())

        y_valid_all = y[valid]
        n_tot_all = len(y_valid_all)
        n_at_eps = int((y_valid_all <= eps).sum()) if n_tot_all > 0 else 0
        n_at_upper = (
            int((y_valid_all >= 0.99 * LAMBDA_MAX).sum()) if n_tot_all > 0 else 0
        )
        pct_at_eps = 100.0 * n_at_eps / n_tot_all if n_tot_all > 0 else np.nan
        pct_at_upper = 100.0 * n_at_upper / n_tot_all if n_tot_all > 0 else np.nan

        if n_valid < min_obs_for_ols:
            warnings.warn(
                f"{country}: only {n_valid} valid observations "
                f"(minimum {min_obs_for_ols}); skipping regression."
            )
            lambda_f_dict[country] = np.full(len(F_t), np.nan)
            diag_records.append(
                {
                    "country": country,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "r_squared": np.nan,
                    "n_obs": n_valid,
                    "factor_mean": factor_mean,
                    "factor_std": factor_std,
                    "mean_lambda_total": float(np.nanmean(y)) if n_tot_all > 0 else np.nan,
                    "mean_lambda_country_raw": np.nan,
                    "mean_lambda_country_post_floor": np.nan,
                    "pct_lambda_country_floored": np.nan,
                    "min_lambda_country_raw": np.nan,
                    "p05_lambda_country_raw": np.nan,
                    "median_lambda_country_raw": np.nan,
                    "max_lambda_country_raw": np.nan,
                    "corr_lambda_total_factor": np.nan,
                    "corr_lambda_country_factor_post_floor": np.nan,
                    "pct_lambda_total_at_eps": pct_at_eps,
                    "pct_lambda_total_at_upper": pct_at_upper,
                    "pca_loading": loading_by_country.get(country, np.nan),
                    "pca_explained_variance_ratio": expl_var,
                    "target_global_mean": target_mean,
                    "target_global_std": target_std,
                    "empirical_cross_mean_mean": empirical_cross_mean_mean,
                    "empirical_cross_mean_std": empirical_cross_mean_std,
                }
            )
            continue

        F_valid = F_t[valid]
        y_reg = y[valid]

        X = np.column_stack([np.ones(n_valid), F_valid])
        coeff, _, _, _ = np.linalg.lstsq(X, y_reg, rcond=None)
        alpha_i, beta_i = float(coeff[0]), float(coeff[1])

        y_hat = alpha_i + beta_i * F_valid
        ss_res = float(np.sum((y_reg - y_hat) ** 2))
        ss_tot = float(np.sum((y_reg - y_reg.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

        if beta_i < 0:
            warnings.warn(
                f"{country}: negative beta = {beta_i:.4f} "
                "(negative exposure to global factor; meaningful but unusual)."
            )

        # lambda_country_raw = lambda_total - beta_i * F_t  (NaN propagates from y)
        lf_raw = y - beta_i * F_t

        valid_lf = np.isfinite(lf_raw)
        n_lf = int(valid_lf.sum())
        n_floor = int((lf_raw[valid_lf] < eps).sum()) if n_lf > 0 else 0
        pct_floor = 100.0 * n_floor / n_lf if n_lf > 0 else np.nan

        if np.isfinite(pct_floor) and pct_floor > 50.0:
            warnings.warn(
                f"{country}: {pct_floor:.1f}% of lambda_country observations floored (>50%); "
                "inspect data coverage."
            )
        elif np.isfinite(pct_floor) and pct_floor > 25.0:
            warnings.warn(
                f"{country}: {pct_floor:.1f}% of lambda_country observations floored (>25%)."
            )
        elif np.isfinite(pct_floor) and pct_floor > 10.0:
            print(f"  NOTE {country}: {pct_floor:.1f}% of lambda_country floored (>10%).")

        lf_post = np.where(valid_lf, np.maximum(lf_raw, eps), np.nan)
        lambda_f_dict[country] = lf_post

        lf_raw_valid = lf_raw[valid_lf]
        lf_post_valid = lf_post[valid_lf]

        corr_total_factor = (
            float(np.corrcoef(y_reg, F_valid)[0, 1]) if n_valid > 1 else np.nan
        )
        F_for_corr = F_t[valid_lf]
        corr_country_factor = (
            float(np.corrcoef(lf_post_valid, F_for_corr)[0, 1]) if n_lf > 1 else np.nan
        )

        pf_str = f"{pct_floor:6.1f}%" if np.isfinite(pct_floor) else "    n/a"
        print(
            f"  {country:5s}  {alpha_i*100:>9.4f}%  {beta_i:>8.4f}  "
            f"{r2:>6.3f}  {n_valid:>5d}  {pf_str:>7s}"
        )

        diag_records.append(
            {
                "country": country,
                "alpha": alpha_i,
                "beta": beta_i,
                "r_squared": r2,
                "n_obs": n_valid,
                "factor_mean": factor_mean,
                "factor_std": factor_std,
                "mean_lambda_total": float(np.nanmean(y)),
                "mean_lambda_country_raw": float(np.mean(lf_raw_valid)),
                "mean_lambda_country_post_floor": float(np.mean(lf_post_valid)),
                "pct_lambda_country_floored": pct_floor,
                "min_lambda_country_raw": float(np.min(lf_raw_valid)),
                "p05_lambda_country_raw": float(np.percentile(lf_raw_valid, 5)),
                "median_lambda_country_raw": float(np.median(lf_raw_valid)),
                "max_lambda_country_raw": float(np.max(lf_raw_valid)),
                "corr_lambda_total_factor": corr_total_factor,
                "corr_lambda_country_factor_post_floor": corr_country_factor,
                "pct_lambda_total_at_eps": pct_at_eps,
                "pct_lambda_total_at_upper": pct_at_upper,
                "pca_loading": loading_by_country.get(country, np.nan),
                "pca_explained_variance_ratio": expl_var,
                "target_global_mean": target_mean,
                "target_global_std": target_std,
                "empirical_cross_mean_mean": empirical_cross_mean_mean,
                "empirical_cross_mean_std": empirical_cross_mean_std,
            }
        )

    diagnostics_df = pd.DataFrame(diag_records)
    lambda_f = pd.DataFrame(lambda_f_dict, index=panel.index)

    # 11. Pre-floor and post-floor lambda_country summary.
    print("\nlambda_country summary (pre-floor raw vs post-floor):")
    for country in lambda_f.columns:
        col_post = lambda_f[country].values
        valid_post = np.isfinite(col_post)
        if not valid_post.any():
            print(f"  {country:4s}: NO VALID OBSERVATIONS")
            continue
        arr_post = col_post[valid_post]
        row = diagnostics_df[diagnostics_df["country"] == country]
        pf = float(row["pct_lambda_country_floored"].iloc[0]) if not row.empty else np.nan
        raw_min = float(row["min_lambda_country_raw"].iloc[0]) if not row.empty else np.nan
        fl_tag = f"  [{pf:.1f}% at eps]" if np.isfinite(pf) and pf > 0 else ""
        raw_tag = (
            f"  (raw min={raw_min*100:.4f}%)" if np.isfinite(raw_min) and raw_min < eps
            else ""
        )
        print(
            f"  {country:4s}: min={arr_post.min()*100:.4f}%  "
            f"mean={arr_post.mean()*100:.4f}%  "
            f"max={arr_post.max()*100:.4f}%{fl_tag}{raw_tag}"
        )

    print("=" * 60 + "\n")

    loadings_df = pd.DataFrame(
        {
            "country": included,
            "loading": raw_loadings,
            "explained_variance_ratio": expl_var,
            "sign_corrected_loading": loadings,
        }
    )

    diagnostics = {
        "loadings_df": loadings_df,
        "diagnostics_df": diagnostics_df,
        "explained_variance_ratio": expl_var,
    }
    return lambda_g, lambda_f, diagnostics


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _pct_str(arr: np.ndarray) -> str:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "  (no valid observations)"
    return (
        f"  min={arr.min()*100:.3f}%  "
        f"p25={np.percentile(arr,25)*100:.3f}%  "
        f"med={np.median(arr)*100:.3f}%  "
        f"p75={np.percentile(arr,75)*100:.3f}%  "
        f"max={arr.max()*100:.3f}%"
    )


def run_diagnostics(result: pd.DataFrame, halt_on_error: bool = True) -> None:
    """Print intensity diagnostics and raise on structural violations."""
    print("\n" + "=" * 60)
    print("INTENSITY DIAGNOSTICS")
    print("=" * 60)

    print("\nlambda_total by country (annualised):")
    any_flag = False
    for country, grp in result.groupby("country"):
        arr = grp["lambda_total"].dropna().to_numpy()
        if len(arr) == 0:
            print(f"  {country}: NO VALID OBSERVATIONS")
            continue
        n_high = int((arr > FLAG_THRESH).sum())
        flag = f"  *** {n_high} obs > {FLAG_THRESH*100:.0f}% ***" if n_high else ""
        print(f"  {country}: {_pct_str(arr)}{flag}")
        if n_high:
            any_flag = True

    if any_flag:
        warnings.warn(
            f"Some lambda_total values exceed {FLAG_THRESH*100:.0f}%/year. "
            "Inspect CAPE data quality and model mapping."
        )

    print("\nlambda_global (PCA-based affine rescaling):")
    global_ts = result.drop_duplicates("date").set_index("date")["lambda_global"].dropna()
    global_arr = global_ts.to_numpy()
    print(_pct_str(global_arr))

    cs_mean = result.groupby("date")["lambda_total"].mean()
    aligned = cs_mean.reindex(global_ts.index).to_numpy()
    valid = np.isfinite(aligned) & np.isfinite(global_arr)
    if valid.any():
        pct_exceeded = float(np.mean(global_arr[valid] > 10.0 * aligned[valid]))
        if pct_exceeded > 0.01:
            msg = (
                f"lambda_global exceeds 10x cross-sectional mean(lambda_total) in "
                f"{pct_exceeded*100:.1f}% of dates. "
                "This suggests a scaling problem in the global factor."
            )
            if halt_on_error:
                raise ValueError(msg)
            warnings.warn(msg)

    print("\nlambda_country (country-specific, post-floor):")
    for country, grp in result.groupby("country"):
        arr = grp["lambda_country"].dropna().to_numpy()
        if len(arr) == 0:
            continue
        n_floor = int((arr <= NONNEG_EPS + 1e-12).sum())
        pct_floor = 100.0 * n_floor / len(arr)
        extra = f"  [{pct_floor:.1f}% at eps floor]" if n_floor > 0 else ""
        if pct_floor > 50.0:
            warnings.warn(
                f"{country}: {pct_floor:.1f}% of lambda_country at eps floor (>50%)."
            )
        elif pct_floor > 25.0:
            warnings.warn(
                f"{country}: {pct_floor:.1f}% of lambda_country at eps floor (>25%)."
            )
        print(f"  {country}: {_pct_str(arr)}{extra}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Public: full pipeline
# ---------------------------------------------------------------------------

def load_cape_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "country", "cape"]].copy()
    df["cape"] = pd.to_numeric(df["cape"], errors="coerce")
    df = df[df["cape"].notna() & (df["cape"] > 0)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


def build_and_save_intensities(
    cape_path: str = CAPE_PATH,
    output_path: str = OUT_PATH,
    loadings_path: str = LOADINGS_PATH,
    betas_path: str = BETAS_PATH,
    halt_on_error: bool = True,
) -> pd.DataFrame:
    """Full pipeline: CAPE -> intensities via PCA regression decomposition.

    Outputs
    -------
    intensities.csv:
        long format: date, country, lambda_total, lambda_global, lambda_country
    pca_loadings.csv:
        country, loading, explained_variance_ratio, sign_corrected_loading
    decomposition_betas.csv:
        OLS alpha/beta/R2 and per-country floor diagnostics
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("CAPE -> DISASTER INTENSITIES  (PCA regression decomposition)")
    print("=" * 60)

    print(f"\nLoading CAPE panel: {cape_path}")
    cape_df = load_cape_panel(cape_path)
    countries = sorted(cape_df["country"].unique())
    print(f"  {len(countries)} countries: {countries}")
    print(f"  Total rows: {len(cape_df):,}")
    for c in countries:
        grp = cape_df[cape_df["country"] == c]
        valid = int(grp["cape"].notna().sum())
        d_min = grp["date"].min().date()
        d_max = grp["date"].max().date()
        print(f"  {c}: {valid} obs  ({d_min} -> {d_max})")

    print("\nBuilding model P/D mapping (runs once) ...")
    params = DisasterModelParams()
    lambda_grid, log_model_pd, interp, model_mean_log_pd = compute_model_pd_mapping(params)
    print(f"  lambda grid: [0, {lambda_grid[-1]:.2f}], {len(lambda_grid)} points")
    print(f"  Model mean log P/D = {model_mean_log_pd:.4f}")

    print("\nInverting CAPE -> lambda_total per country ...")
    lambda_records: list[pd.DataFrame] = []
    for country in countries:
        cape_series = (
            cape_df[cape_df["country"] == country]
            .set_index("date")["cape"]
            .sort_index()
        )
        lam_total = compute_lambda_total_for_country(
            cape_series, model_mean_log_pd, lambda_grid, log_model_pd, interp
        )
        n_valid = int(lam_total.notna().sum())
        if n_valid > 0:
            arr = lam_total.dropna().to_numpy()
            print(
                f"  {country}: {n_valid} valid  "
                f"mean={arr.mean()*100:.4f}%  max={arr.max()*100:.4f}%"
            )
        else:
            print(f"  {country}: 0 valid observations")

        lambda_records.append(
            pd.DataFrame(
                {
                    "date": lam_total.index,
                    "country": country,
                    "lambda_total": lam_total.values,
                }
            )
        )

    all_lambda = pd.concat(lambda_records, ignore_index=True)
    panel_wide = all_lambda.pivot(index="date", columns="country", values="lambda_total")
    panel_wide.index = pd.to_datetime(panel_wide.index)

    lambda_g, lambda_f, diagnostics = decompose_intensities_pca(
        panel_wide,
        target_global_mean=params.lam_bar_g,
        target_global_std=None,
    )

    diagnostics["loadings_df"].to_csv(loadings_path, index=False)
    print(f"Saved PCA loadings -> {loadings_path}")

    diagnostics["diagnostics_df"].to_csv(betas_path, index=False)
    print(f"Saved decomposition diagnostics -> {betas_path}")

    global_df = lambda_g.rename("lambda_global").reset_index()
    global_df.columns = ["date", "lambda_global"]
    global_df["date"] = pd.to_datetime(global_df["date"])

    lambda_f_long = (
        lambda_f.reset_index().melt(
            id_vars="date", var_name="country", value_name="lambda_country"
        )
    )
    lambda_f_long["date"] = pd.to_datetime(lambda_f_long["date"])

    result = (
        all_lambda.merge(global_df, on="date", how="left")
        .merge(lambda_f_long, on=["date", "country"], how="left")
    )
    result = result.sort_values(["country", "date"]).reset_index(drop=True)

    n_with_both = result.dropna(subset=["lambda_global", "lambda_country"]).shape[0]
    n_total = result.shape[0]
    print(
        f"\nIntensity rows: {n_total:,} total, "
        f"{n_with_both:,} with both lambda_global and lambda_country defined"
    )

    countries_included = lambda_f.columns.tolist()
    countries_excluded = [c for c in countries if c not in countries_included]
    if countries_excluded:
        print(f"Countries excluded from decomposition: {countries_excluded}")

    run_diagnostics(result, halt_on_error=halt_on_error)

    result.to_csv(output_path, index=False)
    print(f"Saved intensities -> {output_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cape_path = CAPE_PATH
    if not os.path.exists(cape_path):
        fallback = os.path.join(PARENT_DIR, "country_cape_panel.csv")
        if os.path.exists(fallback):
            print(f"raw_data/country_cape_panel.csv not found; using fallback: {fallback}")
            cape_path = fallback
        else:
            raise FileNotFoundError(
                f"CAPE panel not found at {CAPE_PATH} or {fallback}. "
                "Run python -m calibration.lseg_cape first."
            )

    build_and_save_intensities(cape_path=cape_path, output_path=OUT_PATH)


if __name__ == "__main__":
    main()
