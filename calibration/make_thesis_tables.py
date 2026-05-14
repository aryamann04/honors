"""
make_thesis_tables.py
=====================
Generate thesis-ready tables for Sections 7.6, 8.2, 8.4.

Tables produced
---------------
Table 7.6.1  calibration_main          — parameters + fit (all 15 countries)
Table 7.6.2  cds_term_structure_fit    — slope diagnostics (core 6)
Table 8.2.1  simulated_moments_5y      — empirical vs model moments (core 6 + POOLED)
Table 8.4.1  bond_term_structure_diag  — discriminants + slopes (all 15)

Plus appendix LaTeX versions with all 15 countries.

CLI
---
    python -m calibration.make_thesis_tables
"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.parameters import DisasterModelParams
from core.closed_form import K_const, _defaultable_Ai, _phi_sig2
from core.riccati import discriminant
from models.risk_free import rf_yield
from models.defaultable import def_yield

# ── Paths ────────────────────────────────────────────────────────────────────
PANEL_PATH     = os.path.join(BASE_DIR, "processed_data", "panel.csv")
PARAMS_PATH    = os.path.join(BASE_DIR, "results", "parameters.csv")
MODEL_OBS_PATH = os.path.join(BASE_DIR, "results", "model_vs_observed.csv")
SIM_5Y_PATH    = os.path.join(BASE_DIR, "results", "simulation", "simulated_5y_all.csv")

TABLES_DIR     = os.path.join(BASE_DIR, "results", "tables")
LATEX_DIR      = os.path.join(TABLES_DIR, "latex")

# ── Country groupings ────────────────────────────────────────────────────────
DEVELOPED = ["US", "DE", "UK", "JP", "FR", "NL", "SE"]
EMERGING  = ["BR", "CL", "ES", "ID", "MX", "MY", "TH", "TR"]
ALL_15    = DEVELOPED + EMERGING
CORE      = ["US", "DE", "UK", "CL", "ID", "TR"]
GROUP_MAP = {c: "Developed" for c in DEVELOPED}
GROUP_MAP.update({c: "Emerging" for c in EMERGING})


# ── Formatting helpers ───────────────────────────────────────────────────────

def _fmtf(x, dec=4, na="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{dec}f}"


def _fmtpct(x, dec=1, na="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{dec}f}"


def _fmtsci(x, na="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    if abs(x) > 0 and abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:.6f}"


def _save_csv(df: pd.DataFrame, name: str) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name)
    df.to_csv(path, index=False)
    print(f"  CSV  → {os.path.relpath(path, PARENT_DIR)}")


def _save_latex(lines: list[str], name: str) -> None:
    os.makedirs(LATEX_DIR, exist_ok=True)
    path = os.path.join(LATEX_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX→ {os.path.relpath(path, PARENT_DIR)}")


# ── Model helpers ────────────────────────────────────────────────────────────

def _build_base_params() -> DisasterModelParams:
    p = DisasterModelParams()
    p.compute_b_sdf()
    return p


def _build_country_params(
    base: DisasterModelParams,
    row: pd.Series,
    lam_bar_f: float,
    lam_bar_g: float,
) -> DisasterModelParams:
    return dataclasses.replace(
        base,
        h0_star   = float(row["h0"]),
        eta1      = float(row["eta_i"]),
        eta2      = float(row["eta_g"]),
        R         = float(row["R"]),
        lam_bar_f = float(lam_bar_f),
        lam_bar_g = float(lam_bar_g),
        b_sdf     = base.b_sdf,
    )


def _country_params_dict(
    params_df: pd.DataFrame,
    panel: pd.DataFrame,
    base: DisasterModelParams,
) -> dict[str, DisasterModelParams]:
    lam_bar_g = float(panel["lambda_global"].mean())
    out: dict[str, DisasterModelParams] = {}
    for _, row in params_df.iterrows():
        c = row["country"]
        grp = panel[panel["country"] == c]
        lam_bar_f = float(grp["lambda_country"].mean()) if not grp.empty else 1e-4
        out[c] = _build_country_params(base, row, lam_bar_f, lam_bar_g)
    return out


# ── LaTeX table builder ──────────────────────────────────────────────────────

def _latex_table(
    caption: str,
    label:   str,
    col_fmt: str,
    header:  list[str],
    groups:  list[tuple[str, list[list[str]]]],   # [(group_label, [row_cells, ...]), ...]
    note:    str | None = None,
    bold_labels: set[str] | None = None,
) -> list[str]:
    """Build booktabs LaTeX table lines.

    groups: list of (group_title, [[cell, cell, ...], ...])
    bold_labels: set of country codes to render in \\textbf{}
    """
    ncols = len(header)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        " & ".join(header) + r" \\",
    ]

    for g_idx, (group_label, rows) in enumerate(groups):
        lines.append(r"\midrule")
        if group_label:
            span = f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{group_label}}}}}"
            lines.append(span + r" \\")
            lines.append(r"\midrule")
        for row_cells in rows:
            if bold_labels and row_cells[0].strip() in bold_labels:
                row_cells = [f"\\textbf{{{c}}}" for c in row_cells]
            lines.append(" & ".join(row_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines += [
            r"\begin{tablenotes}[flushleft]\footnotesize",
            f"\\item {note}",
            r"\end{tablenotes}",
        ]
    lines.append(r"\end{table}")
    return lines


# ── Table 7.6.1 — Calibration main ──────────────────────────────────────────

def make_calibration_main(
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
) -> pd.DataFrame:
    """Table 7.6.1: parameters + fit, all 15 countries."""

    # Correlation between model and observed spreads per country
    corr_map: dict[str, float] = {}
    rmse_map: dict[str, float] = {}
    for c, grp in model_vs_obs.groupby("country"):
        sub = grp.dropna(subset=["spread_obs", "spread_model"])
        if len(sub) > 2:
            corr_map[c] = float(sub["spread_obs"].corr(sub["spread_model"]))
            rmse_map[c] = float(np.sqrt(np.mean(sub["residual_bps"] ** 2)))
        else:
            corr_map[c] = np.nan
            rmse_map[c] = np.nan

    records = []
    for c in ALL_15:
        row = params_df[params_df["country"] == c]
        if row.empty:
            continue
        row = row.iloc[0]
        eta_i = float(row["eta_i"])
        eta_g = float(row["eta_g"])
        denom = eta_i + eta_g
        global_share = eta_g / denom if denom > 1e-12 else np.nan
        records.append({
            "Country":        c,
            "Group":          GROUP_MAP.get(c, ""),
            "h0_star":        float(row["h0"]),
            "eta_i":          eta_i,
            "eta_g":          eta_g,
            "global_share":   global_share,
            "Recovery":       float(row["R"]),
            "RMSE_bps":       rmse_map.get(c, float(row["rmse"]) * 1e4),
            "Corr_model_obs": corr_map.get(c, np.nan),
            "N":              int(row["num_obs"]),
        })

    df = pd.DataFrame(records)
    _save_csv(df, "calibration_main.csv")

    # LaTeX
    header = [
        "Country", "Group",
        r"$h_0^*$", r"$\eta_i$", r"$\eta_{g,i}$",
        r"Global Share", r"$R$",
        r"RMSE (bps)", r"Corr", r"$N$",
    ]
    dev_rows, em_rows = [], []
    for _, r in df.iterrows():
        gs = _fmtf(r["global_share"], 3) if np.isfinite(r["global_share"]) else "—"
        row_cells = [
            r["Country"],
            r["Group"][:3],
            _fmtsci(r["h0_star"]),
            _fmtf(r["eta_i"], 4),
            _fmtf(r["eta_g"], 4),
            gs,
            _fmtf(r["Recovery"], 2),
            _fmtf(r["RMSE_bps"], 2),
            _fmtf(r["Corr_model_obs"], 3),
            str(r["N"]),
        ]
        if r["Group"] == "Developed":
            dev_rows.append(row_cells)
        else:
            em_rows.append(row_cells)

    lines = _latex_table(
        caption  = "Estimated sovereign hazard parameters and calibration fit.",
        label    = "tab:calibration_main",
        col_fmt  = "llrrrrrrrr",
        header   = header,
        groups   = [("Developed", dev_rows), ("Emerging", em_rows)],
        note     = (
            r"$h_0^*$ is the baseline hazard; $\eta_i$ and $\eta_{g,i}$ are loadings on "
            r"country-specific and global disaster intensity. "
            r"Global Share $= \eta_{g,i}/(\eta_i+\eta_{g,i})$. "
            r"RMSE in basis points across all tenors."
        ),
    )
    _save_latex(lines, "calibration_main.tex")
    return df


# ── Table 7.6.2 — CDS term-structure fit ────────────────────────────────────

def make_cds_term_structure_fit(
    panel:        pd.DataFrame,
    model_vs_obs: pd.DataFrame,
) -> pd.DataFrame:
    """Table 7.6.2: slope diagnostics for core 6 countries."""
    records = []
    for c in CORE:
        grp_panel = panel[panel["country"] == c].sort_values("date")
        grp_mv    = model_vs_obs[model_vs_obs["country"] == c]

        # Observed 10Y-1Y slope from panel
        if "cds_spread_10y" in grp_panel.columns and "cds_spread_1y" in grp_panel.columns:
            slope_obs = (grp_panel["cds_spread_10y"] - grp_panel["cds_spread_1y"]).dropna() * 1e4
            mean_obs_slope = float(slope_obs.mean()) if len(slope_obs) > 0 else np.nan
        else:
            mean_obs_slope = np.nan
            warnings.warn(f"{c}: 10Y or 1Y CDS column missing; skipping slope computation.")

        # Model 10Y-1Y slope from model_vs_observed
        mv_10 = grp_mv[grp_mv["tenor"] == 10.0][["date", "spread_model"]].rename(
            columns={"spread_model": "m10"}
        )
        mv_1  = grp_mv[grp_mv["tenor"] == 1.0][["date", "spread_model"]].rename(
            columns={"spread_model": "m1"}
        )
        if not mv_10.empty and not mv_1.empty:
            merged = mv_10.merge(mv_1, on="date", how="inner")
            mean_model_slope = float((merged["m10"] - merged["m1"]).mean()) if len(merged) > 0 else np.nan
        else:
            mean_model_slope = np.nan
            warnings.warn(f"{c}: model 10Y or 1Y spread missing; skipping slope computation.")

        slope_err = (mean_model_slope - mean_obs_slope) if (
            np.isfinite(mean_model_slope) and np.isfinite(mean_obs_slope)
        ) else np.nan

        # 5Y fit statistics
        sub5 = grp_mv[grp_mv["tenor"] == 5.0].dropna(subset=["spread_obs", "spread_model"])
        corr_5y  = float(sub5["spread_obs"].corr(sub5["spread_model"])) if len(sub5) > 2 else np.nan
        rmse_5y  = float(np.sqrt(np.mean(sub5["residual_bps"] ** 2)))   if len(sub5) > 0 else np.nan

        # All-tenor RMSE
        sub_all  = grp_mv.dropna(subset=["residual_bps"])
        rmse_all = float(np.sqrt(np.mean(sub_all["residual_bps"] ** 2))) if len(sub_all) > 0 else np.nan

        records.append({
            "Country":                    c,
            "Group":                      GROUP_MAP.get(c, ""),
            "Mean_obs_10Y_minus_1Y_bps":  mean_obs_slope,
            "Mean_model_10Y_minus_1Y_bps": mean_model_slope,
            "Slope_error_bps":            slope_err,
            "Corr_5Y":                    corr_5y,
            "RMSE_5Y_bps":                rmse_5y,
            "RMSE_all_bps":               rmse_all,
        })

    df = pd.DataFrame(records)
    _save_csv(df, "cds_term_structure_fit.csv")

    header = [
        "Country", "Group",
        r"Mean obs slope (bps)", r"Mean model slope (bps)",
        r"Slope error (bps)",
        r"Corr$_{5Y}$", r"RMSE$_{5Y}$ (bps)", r"RMSE$_\text{all}$ (bps)",
    ]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["Country"], r["Group"][:3],
            _fmtf(r["Mean_obs_10Y_minus_1Y_bps"], 1),
            _fmtf(r["Mean_model_10Y_minus_1Y_bps"], 1),
            _fmtf(r["Slope_error_bps"], 1),
            _fmtf(r["Corr_5Y"], 3),
            _fmtf(r["RMSE_5Y_bps"], 2),
            _fmtf(r["RMSE_all_bps"], 2),
        ])
    lines = _latex_table(
        caption  = "Observed and model-implied CDS term-structure slope diagnostics.",
        label    = "tab:cds_term_structure_fit",
        col_fmt  = "llrrrrrr",
        header   = header,
        groups   = [("", rows)],
        note     = (
            "Slope = 10Y minus 1Y CDS spread; positive value means upward-sloping. "
            "Slope error = model slope minus observed slope. "
            r"Corr$_{5Y}$: Pearson correlation of model vs.\ observed 5-year CDS spreads."
        ),
    )
    _save_latex(lines, "cds_term_structure_fit.tex")
    return df


# ── Table 8.2.1 — Simulated moments ──────────────────────────────────────────

def _lag1_autocorr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    return float(pd.Series(x).autocorr(lag=1))


def make_simulated_moments_5y(
    panel:        pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    sim_5y_path:  str = SIM_5Y_PATH,
) -> pd.DataFrame:
    """Table 8.2.1: empirical and model-implied moments of 5Y CDS spreads."""
    sim_df = None
    if os.path.exists(sim_5y_path):
        sim_df = pd.read_csv(sim_5y_path)
    else:
        warnings.warn(
            f"Simulation file not found: {sim_5y_path}\n"
            "  Run  python -m calibration.simulate_calibrated_model  first.\n"
            "  Model moments will be filled from model_vs_observed.csv instead."
        )

    # Empirical moments from panel.csv (observed 5Y CDS, in bps)
    panel_5y = panel[["country", "date", "lambda_global", "lambda_country",
                       "cds_spread_5y"]].copy()
    panel_5y = panel_5y.dropna(subset=["cds_spread_5y"])
    panel_5y["cds_5y_bps"] = panel_5y["cds_spread_5y"] * 1e4

    # Model 5Y CDS from model_vs_observed.csv (for corr/autocorr at observed states)
    mv5 = model_vs_obs[model_vs_obs["tenor"] == 5.0].copy()
    mv5 = mv5.dropna(subset=["spread_model", "spread_obs"])
    mv5 = mv5.merge(
        panel[["country", "date", "lambda_global", "lambda_country"]],
        on=["country", "date"], how="left",
    )

    country_list = ["POOLED"] + CORE
    records = []

    for c in country_list:
        if c == "POOLED":
            p_sub  = panel_5y.copy()
            mv_sub = mv5.copy()
            sim_sub = sim_df.copy() if sim_df is not None else None
        else:
            p_sub  = panel_5y[panel_5y["country"] == c]
            mv_sub = mv5[mv5["country"] == c]
            sim_sub = sim_df[sim_df["country"] == c] if sim_df is not None else None

        # Empirical moments
        obs = p_sub["cds_5y_bps"].dropna().to_numpy()
        data_mean = float(np.mean(obs))  if len(obs) > 0 else np.nan
        data_std  = float(np.std(obs, ddof=1)) if len(obs) > 1 else np.nan
        data_ac   = _lag1_autocorr(obs)

        # Empirical correlations with intensities
        merged_emp = p_sub.dropna(subset=["cds_5y_bps", "lambda_global", "lambda_country"])
        data_corr_g = float(merged_emp["cds_5y_bps"].corr(merged_emp["lambda_global"])) if len(merged_emp) > 2 else np.nan
        data_corr_i = float(merged_emp["cds_5y_bps"].corr(merged_emp["lambda_country"])) if len(merged_emp) > 2 else np.nan

        # Model moments: prefer simulation, fall back to model_vs_observed
        if sim_sub is not None and len(sim_sub) > 10:
            model_s = sim_sub["cds_5y_bps"].to_numpy()
            model_mean = float(np.mean(model_s))
            model_std  = float(np.std(model_s, ddof=1))
            model_ac   = _lag1_autocorr(model_s)
            model_corr_g = float(pd.Series(model_s).corr(pd.Series(sim_sub["lambda_global"].to_numpy())))
            model_corr_i = float(pd.Series(model_s).corr(pd.Series(sim_sub["lambda_country"].to_numpy())))
        else:
            # Fall back: use model_vs_observed at observed states
            mv_clean = mv_sub.dropna(subset=["spread_model", "lambda_global", "lambda_country"])
            ms = mv_clean["spread_model"].to_numpy()
            model_mean = float(np.mean(ms))  if len(ms) > 0 else np.nan
            model_std  = float(np.std(ms, ddof=1)) if len(ms) > 1 else np.nan
            model_ac   = _lag1_autocorr(ms)
            model_corr_g = float(mv_clean["spread_model"].corr(mv_clean["lambda_global"])) if len(mv_clean) > 2 else np.nan
            model_corr_i = float(mv_clean["spread_model"].corr(mv_clean["lambda_country"])) if len(mv_clean) > 2 else np.nan

        records.append({
            "Country":            c,
            "Group":              GROUP_MAP.get(c, "All") if c != "POOLED" else "All",
            "Data_mean_5Y_bps":   data_mean,
            "Model_mean_5Y_bps":  model_mean,
            "Data_std_5Y_bps":    data_std,
            "Model_std_5Y_bps":   model_std,
            "Data_autocorr_5Y":   data_ac,
            "Model_autocorr_5Y":  model_ac,
            "Data_corr_5Y_lambda_g":  data_corr_g,
            "Model_corr_5Y_lambda_g": model_corr_g,
            "Data_corr_5Y_lambda_i":  data_corr_i,
            "Model_corr_5Y_lambda_i": model_corr_i,
        })

    df = pd.DataFrame(records)
    _save_csv(df, "simulated_moments_5y.csv")

    header = [
        "Country", "Group",
        r"$\bar{s}^{\text{data}}$", r"$\bar{s}^{\text{model}}$",
        r"$\sigma^{\text{data}}$", r"$\sigma^{\text{model}}$",
        r"$\rho^{\text{data}}_1$", r"$\rho^{\text{model}}_1$",
        r"$r^d(\lambda^g)$", r"$r^m(\lambda^g)$",
        r"$r^d(\lambda^i)$", r"$r^m(\lambda^i)$",
    ]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["Country"], r["Group"][:3] if r["Country"] != "POOLED" else "All",
            _fmtf(r["Data_mean_5Y_bps"],   1),
            _fmtf(r["Model_mean_5Y_bps"],  1),
            _fmtf(r["Data_std_5Y_bps"],    1),
            _fmtf(r["Model_std_5Y_bps"],   1),
            _fmtf(r["Data_autocorr_5Y"],   3),
            _fmtf(r["Model_autocorr_5Y"],  3),
            _fmtf(r["Data_corr_5Y_lambda_g"],  3),
            _fmtf(r["Model_corr_5Y_lambda_g"], 3),
            _fmtf(r["Data_corr_5Y_lambda_i"],  3),
            _fmtf(r["Model_corr_5Y_lambda_i"], 3),
        ])
    lines = _latex_table(
        caption  = "Empirical and model-implied moments of 5-year CDS spreads.",
        label    = "tab:simulated_moments_5y",
        col_fmt  = "llrrrrrrrrrr",
        header   = header,
        groups   = [("", rows)],
        note     = (
            r"All spreads in basis points. $\bar{s}$: mean; $\sigma$: standard deviation; "
            r"$\rho_1$: lag-1 autocorrelation; "
            r"$r(\lambda^g)$, $r(\lambda^i)$: Pearson correlation with global and "
            r"country-specific disaster intensity. "
            r"Model moments from calibrated simulation for individual countries; "
            r"from model-vs-observed at observed states for POOLED."
        ),
    )
    _save_latex(lines, "simulated_moments_5y.tex")
    return df


# ── Table 8.4.1 — Bond term-structure diagnostics ────────────────────────────

def make_bond_term_structure_diagnostics(
    params_df: pd.DataFrame,
    panel:     pd.DataFrame,
    base:      DisasterModelParams,
) -> pd.DataFrame:
    """Table 8.4.1: discriminants and slope diagnostics, all 15 countries."""
    country_params = _country_params_dict(params_df, panel, base)

    phi, sig2 = _phi_sig2(base)
    K         = K_const(base)
    delta_K   = discriminant(K, phi, sig2)   # same for all countries

    records = []
    for c in ALL_15:
        p = country_params.get(c)
        if p is None:
            continue

        A0, Af, Ag = _defaultable_Ai(p)
        x_f = -Af   # Riccati forcing for b_{D,f}
        x_g = -Ag   # Riccati forcing for b_{D,g}

        d_minus_Ai = discriminant(x_f, phi, sig2)
        d_minus_Ag = discriminant(x_g, phi, sig2)

        rf_conv  = bool(delta_K > 0)            # False for all
        def_i_cv = bool(d_minus_Ai > 0)
        def_g_cv = bool(d_minus_Ag > 0)

        # Yield slopes at observed states
        grp = panel[panel["country"] == c].dropna(
            subset=["lambda_global", "lambda_country"]
        )
        if grp.empty:
            rf_slope_mean   = np.nan
            def_slope_mean  = np.nan
            pct_up_rf       = np.nan
            pct_up_def      = np.nan
        else:
            lf = grp["lambda_country"].to_numpy()
            lg = grp["lambda_global"].to_numpy()

            # Compute yields vectorised: formula is linear in state
            from core.closed_form import b_star_cf, a_star_cf, defaultable_coeffs_cf

            b1   = b_star_cf(p, 1.0);   b10  = b_star_cf(p, 10.0)
            a1   = a_star_cf(p, 1.0);   a10  = a_star_cf(p, 10.0)

            rf1  = -(a1  + b1  * (lf + lg)) / 1.0
            rf10 = -(a10 + b10 * (lf + lg)) / 10.0
            rf_slopes = rf10 - rf1

            aD1, bDf1, bDg1   = defaultable_coeffs_cf(p, 1.0)
            aD10, bDf10, bDg10 = defaultable_coeffs_cf(p, 10.0)

            def1  = -(aD1  + bDf1  * lf + bDg1  * lg) / 1.0
            def10 = -(aD10 + bDf10 * lf + bDg10 * lg) / 10.0
            def_slopes = def10 - def1

            finite_rf  = np.isfinite(rf_slopes)
            finite_def = np.isfinite(def_slopes)

            rf_slope_mean  = float(np.mean(rf_slopes[finite_rf]))  if finite_rf.any()  else np.nan
            def_slope_mean = float(np.mean(def_slopes[finite_def])) if finite_def.any() else np.nan
            pct_up_rf  = float(np.mean(rf_slopes[finite_rf]  > 0) * 100) if finite_rf.any()  else np.nan
            pct_up_def = float(np.mean(def_slopes[finite_def] > 0) * 100) if finite_def.any() else np.nan

        records.append({
            "Country":                      c,
            "Group":                        GROUP_MAP.get(c, ""),
            "delta_K":                      delta_K,
            "delta_minus_A_i":              d_minus_Ai,
            "delta_minus_A_g":              d_minus_Ag,
            "RF_converges":                 rf_conv,
            "Defaultable_i_converges":      def_i_cv,
            "Defaultable_g_converges":      def_g_cv,
            "Mean_RF_slope_10Y_1Y_bps":     rf_slope_mean  * 1e4 if np.isfinite(rf_slope_mean)  else np.nan,
            "Mean_Defaultable_slope_10Y_1Y_bps": def_slope_mean * 1e4 if np.isfinite(def_slope_mean) else np.nan,
            "Pct_upward_RF":                pct_up_rf,
            "Pct_upward_Defaultable":       pct_up_def,
        })

    df = pd.DataFrame(records)
    _save_csv(df, "bond_term_structure_diagnostics.csv")

    header = [
        "Country", "Group",
        r"$\delta(K)$", r"$\delta(-A_i)$", r"$\delta(-A_g)$",
        r"RF cvg.", r"$D_i$ cvg.", r"$D_g$ cvg.",
        r"Mean RF slope (bps)", r"Mean $D$ slope (bps)",
        r"\%\ up RF", r"\%\ up $D$",
    ]
    dev_rows, em_rows = [], []
    for _, r in df.iterrows():
        row_cells = [
            r["Country"],
            r["Group"][:3],
            _fmtsci(r["delta_K"]),
            _fmtsci(r["delta_minus_A_i"]),
            _fmtsci(r["delta_minus_A_g"]),
            "Yes" if r["RF_converges"] else "No",
            "Yes" if r["Defaultable_i_converges"] else "No",
            "Yes" if r["Defaultable_g_converges"] else "No",
            _fmtf(r["Mean_RF_slope_10Y_1Y_bps"], 2),
            _fmtf(r["Mean_Defaultable_slope_10Y_1Y_bps"], 2),
            _fmtpct(r["Pct_upward_RF"], 1),
            _fmtpct(r["Pct_upward_Defaultable"], 1),
        ]
        if r["Group"] == "Developed":
            dev_rows.append(row_cells)
        else:
            em_rows.append(row_cells)

    lines = _latex_table(
        caption     = r"Discriminant and slope diagnostics for calibrated bond term structures.",
        label       = "tab:bond_term_structure_diagnostics",
        col_fmt     = "llrrrcccrrrr",
        header      = header,
        groups      = [("Developed", dev_rows), ("Emerging", em_rows)],
        bold_labels = {"TR"},
        note        = (
            r"$\delta(x) = \varphi^2 - 2\sigma_\lambda^2 x$ is the Riccati discriminant. "
            r"``cvg.'': convergent ($\delta > 0$, exponential branch). "
            r"Slope = model-implied $y(10\mathrm{Y}) - y(1\mathrm{Y})$ in basis points, "
            r"averaged over observed intensity states. "
            r"\textbf{TR} is the only country satisfying both "
            r"$\delta(-A_i) > 0$ and $\delta(-A_g) > 0$."
        ),
    )
    _save_latex(lines, "bond_term_structure_diagnostics.tex")
    return df


# ── Appendix tables ───────────────────────────────────────────────────────────

def make_appendix_tables(
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    panel:        pd.DataFrame,
    base:         DisasterModelParams,
    sim_5y_path:  str = SIM_5Y_PATH,
) -> None:
    """Save full versions of all tables as appendix LaTeX."""

    # Full parameters table (all columns)
    records_full = []
    for _, row in params_df.iterrows():
        c = row["country"]
        sub = model_vs_obs[model_vs_obs["country"] == c].dropna(subset=["spread_obs", "spread_model"])
        corr = float(sub["spread_obs"].corr(sub["spread_model"])) if len(sub) > 2 else np.nan
        records_full.append({
            "Country":   c,
            "Group":     GROUP_MAP.get(c, ""),
            "h0_star":   float(row["h0"]),
            "eta_i":     float(row["eta_i"]),
            "eta_g":     float(row["eta_g"]),
            "R":         float(row["R"]),
            "RMSE_bps":  float(row["rmse"]) * 1e4,
            "Corr":      corr,
            "N":         int(row["num_obs"]),
            "Converged": bool(row.get("converged", True)),
        })
    header = ["Country", "Group",
              r"$h_0^*$", r"$\eta_i$", r"$\eta_{g,i}$", r"$R$",
              "RMSE (bps)", "Corr", r"$N$", "Conv."]
    col_fmt = "llrrrrrrrc"
    rows_all = []
    for r in records_full:
        rows_all.append([
            r["Country"], r["Group"][:3],
            _fmtsci(r["h0_star"]), _fmtf(r["eta_i"], 4), _fmtf(r["eta_g"], 4),
            _fmtf(r["R"], 2), _fmtf(r["RMSE_bps"], 2), _fmtf(r["Corr"], 3),
            str(r["N"]), "Y" if r["Converged"] else "N",
        ])
    lines = _latex_table(
        caption = "Calibrated hazard parameters, all countries (appendix).",
        label   = "tab:appendix_full_parameters",
        col_fmt = col_fmt,
        header  = header,
        groups  = [("", rows_all)],
    )
    _save_latex(lines, "appendix_full_parameters.tex")

    # Full intensity diagnostics table from panel
    records_int = []
    for c in ALL_15:
        grp = panel[panel["country"] == c]
        lg  = grp["lambda_global"].dropna().to_numpy() * 100
        li  = grp["lambda_country"].dropna().to_numpy() * 100
        records_int.append({
            "Country":      c,
            "N_obs":        len(grp),
            "LG_mean":      float(np.mean(lg))   if len(lg) > 0 else np.nan,
            "LG_std":       float(np.std(lg))    if len(lg) > 1 else np.nan,
            "LI_mean":      float(np.mean(li))   if len(li) > 0 else np.nan,
            "LI_std":       float(np.std(li))    if len(li) > 1 else np.nan,
        })
    df_int = pd.DataFrame(records_int)
    int_rows = [[r["Country"], str(r["N_obs"]),
                 _fmtf(r["LG_mean"], 3), _fmtf(r["LG_std"], 3),
                 _fmtf(r["LI_mean"], 3), _fmtf(r["LI_std"], 3)]
                for _, r in df_int.iterrows()]
    lines_int = _latex_table(
        caption = r"Disaster intensity summary statistics, all countries (in \%, appendix).",
        label   = "tab:appendix_full_intensity_diagnostics",
        col_fmt = "lrrrrrr",
        header  = ["Country", r"$N$",
                   r"$\bar\lambda^g$ (\%)", r"$\sigma(\lambda^g)$ (\%)",
                   r"$\bar\lambda^i$ (\%)", r"$\sigma(\lambda^i)$ (\%)"],
        groups  = [("", int_rows)],
    )
    _save_latex(lines_int, "appendix_full_intensity_diagnostics.tex")

    # Full simulation moments (re-use make_simulated_moments_5y with all countries)
    sim_df = pd.read_csv(sim_5y_path) if os.path.exists(sim_5y_path) else None
    panel_5y = panel[["country", "date", "lambda_global", "lambda_country",
                       "cds_spread_5y"]].copy().dropna(subset=["cds_spread_5y"])
    panel_5y["cds_5y_bps"] = panel_5y["cds_spread_5y"] * 1e4
    mv5 = model_vs_obs[model_vs_obs["tenor"] == 5.0].dropna(subset=["spread_model"])
    mv5 = mv5.merge(
        panel[["country", "date", "lambda_global", "lambda_country"]],
        on=["country", "date"], how="left",
    )
    sim_records = []
    for c in ["POOLED"] + ALL_15:
        if c == "POOLED":
            p_sub  = panel_5y.copy()
            mv_sub = mv5.copy()
            s_sub  = sim_df.copy() if sim_df is not None else None
        else:
            p_sub  = panel_5y[panel_5y["country"] == c]
            mv_sub = mv5[mv5["country"] == c]
            s_sub  = sim_df[sim_df["country"] == c] if sim_df is not None else None
        obs = p_sub["cds_5y_bps"].dropna().to_numpy()
        d_mean = float(np.mean(obs)) if len(obs) > 0 else np.nan
        d_std  = float(np.std(obs, ddof=1)) if len(obs) > 1 else np.nan
        d_ac   = _lag1_autocorr(obs)
        if s_sub is not None and len(s_sub) > 10:
            ms = s_sub["cds_5y_bps"].to_numpy()
            m_mean = float(np.mean(ms))
            m_std  = float(np.std(ms, ddof=1))
            m_ac   = _lag1_autocorr(ms)
        else:
            mv_clean = mv_sub.dropna(subset=["spread_model"])
            ms = mv_clean["spread_model"].to_numpy()
            m_mean = float(np.mean(ms)) if len(ms) > 0 else np.nan
            m_std  = float(np.std(ms, ddof=1)) if len(ms) > 1 else np.nan
            m_ac   = _lag1_autocorr(ms)
        sim_records.append([
            c, GROUP_MAP.get(c, "All") if c != "POOLED" else "All",
            _fmtf(d_mean, 1), _fmtf(m_mean, 1),
            _fmtf(d_std, 1), _fmtf(m_std, 1),
            _fmtf(d_ac, 3), _fmtf(m_ac, 3),
        ])
    lines_sim = _latex_table(
        caption = "Empirical and model-implied moments of 5-year CDS spreads, all countries (appendix).",
        label   = "tab:appendix_full_simulation_moments",
        col_fmt = "llrrrrrr",
        header  = ["Country", "Group",
                   r"$\bar{s}^d$ (bps)", r"$\bar{s}^m$ (bps)",
                   r"$\sigma^d$ (bps)", r"$\sigma^m$ (bps)",
                   r"$\rho^d_1$", r"$\rho^m_1$"],
        groups  = [("", sim_records)],
    )
    _save_latex(lines_sim, "appendix_full_simulation_moments.tex")

    # Full term-structure diagnostics table (all 15)
    diag_df = make_bond_term_structure_diagnostics(params_df, panel, base)
    diag_rows = []
    for _, r in diag_df.iterrows():
        diag_rows.append([
            r["Country"], r["Group"][:3],
            _fmtsci(r["delta_K"]),
            _fmtsci(r["delta_minus_A_i"]),
            _fmtsci(r["delta_minus_A_g"]),
            "Y" if r["RF_converges"] else "N",
            "Y" if r["Defaultable_i_converges"] else "N",
            "Y" if r["Defaultable_g_converges"] else "N",
            _fmtf(r["Mean_RF_slope_10Y_1Y_bps"], 2),
            _fmtf(r["Mean_Defaultable_slope_10Y_1Y_bps"], 2),
            _fmtpct(r["Pct_upward_RF"], 1),
            _fmtpct(r["Pct_upward_Defaultable"], 1),
        ])
    lines_diag = _latex_table(
        caption = "Discriminant and slope diagnostics, all countries (appendix).",
        label   = "tab:appendix_full_term_structure_diagnostics",
        col_fmt = "llrrrcccrrrr",
        header  = ["Country", "Group",
                   r"$\delta(K)$", r"$\delta(-A_i)$", r"$\delta(-A_g)$",
                   "RF", r"$D_i$", r"$D_g$",
                   "RF slope (bps)", "D slope (bps)",
                   r"\%\ up RF", r"\%\ up D"],
        groups  = [("", diag_rows)],
    )
    _save_latex(lines_diag, "appendix_full_term_structure_diagnostics.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for path in [PANEL_PATH, PARAMS_PATH, MODEL_OBS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    print("Loading data ...")
    panel        = pd.read_csv(PANEL_PATH,     parse_dates=["date"])
    params_df    = pd.read_csv(PARAMS_PATH)
    model_vs_obs = pd.read_csv(MODEL_OBS_PATH, parse_dates=["date"])

    base = _build_base_params()
    print(f"  b_sdf = {base.b_sdf:.6f}")

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(LATEX_DIR,  exist_ok=True)

    print("\nTable 7.6.1 — calibration_main ...")
    make_calibration_main(params_df, model_vs_obs)

    print("\nTable 7.6.2 — cds_term_structure_fit ...")
    make_cds_term_structure_fit(panel, model_vs_obs)

    print("\nTable 8.2.1 — simulated_moments_5y ...")
    make_simulated_moments_5y(panel, model_vs_obs)

    print("\nTable 8.4.1 — bond_term_structure_diagnostics ...")
    make_bond_term_structure_diagnostics(params_df, panel, base)

    print("\nAppendix tables ...")
    make_appendix_tables(params_df, model_vs_obs, panel, base)

    print("\nAll tables generated.")
    for sub in [TABLES_DIR, LATEX_DIR]:
        files = [f for f in os.listdir(sub) if f.endswith((".csv", ".tex"))]
        print(f"\n  {sub}/")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(sub, f))
            print(f"    {f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
