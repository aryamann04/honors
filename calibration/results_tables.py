"""
results_tables.py
=================
Thesis-ready calibration result tables.

All functions accept clean DataFrames loaded from the pipeline outputs and return
pandas DataFrames ready to save as CSV (and optionally LaTeX).

Table inventory
---------------
1.  calibration_sample_summary      — sample coverage and descriptive stats per country
2.  intensity_summary               — λ distribution moments per country × variable
3.  estimated_parameters_summary    — h0, η_f, η_g, global share, R, RMSE  (+.tex)
4.  fit_by_country                  — RMSE/MAE/correlation per country
5.  fit_by_tenor                    — RMSE/MAE/correlation per tenor
6.  hazard_proxy_comparison         — diagnostic: CDS-implied vs model hazard rate
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(s: pd.Series) -> tuple:
    arr = s.dropna().to_numpy()
    if len(arr) == 0:
        return (np.nan,) * 7
    return (
        np.mean(arr), np.median(arr), np.std(arr, ddof=1),
        np.min(arr),  np.max(arr),
        np.percentile(arr, 10), np.percentile(arr, 90),
    )


def _save(df: pd.DataFrame, name: str, out_dir: str = TABLES_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"  → {os.path.relpath(path, PARENT_DIR)}")


# ---------------------------------------------------------------------------
# 1. Calibration sample summary
# ---------------------------------------------------------------------------

def build_calibration_sample_summary(
    panel:     pd.DataFrame,
    params_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per country with sample coverage and mean statistics."""
    TENOR_COLS = ["cds_spread_1y", "cds_spread_2y", "cds_spread_3y",
                  "cds_spread_5y", "cds_spread_7y", "cds_spread_10y"]
    records = []
    for country, grp in panel.sort_values("date").groupby("country"):
        cds_cols   = [c for c in TENOR_COLS if c in grp.columns]
        valid_mask = grp[cds_cols].notna().any(axis=1) if cds_cols else pd.Series(False, index=grp.index)
        tenors_ok  = sorted([
            c.replace("cds_spread_", "").upper()
            for c in cds_cols if grp[c].notna().any()
        ])

        rec: dict = {
            "country":          country,
            "start_date":       str(grp["date"].min().date()),
            "end_date":         str(grp["date"].max().date()),
            "n_months":         int(len(grp)),
            "n_cds_obs":        int(valid_mask.sum()),
            "tenors_available": " ".join(tenors_ok),
        }

        s5y = grp["cds_spread_5y"] if "cds_spread_5y" in grp.columns else pd.Series(dtype=float)
        rec["mean_5y_cds_bps"] = round(s5y.mean() * 1e4, 1) if s5y.notna().any() else np.nan

        for col, key in [
            ("lambda_total",   "mean_lambda_total"),
            ("lambda_country", "mean_lambda_country"),
            ("lambda_global",  "mean_lambda_global"),
        ]:
            if col in grp.columns:
                rec[key] = round(float(grp[col].mean()), 5)

        if "recovery_rate" in grp.columns:
            rv = grp["recovery_rate"].dropna()
            rec["median_R"] = round(float(rv.median()), 2) if len(rv) > 0 else np.nan

        if "cape" in grp.columns:
            rec["mean_cape"] = round(float(grp["cape"].mean()), 2) if grp["cape"].notna().any() else np.nan

        records.append(rec)

    return pd.DataFrame(records).sort_values("country").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Intensity summary
# ---------------------------------------------------------------------------

def build_intensity_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-country distribution moments for λ_total, λ_country, λ_global."""
    variables = [
        ("lambda_total",   "λ_total"),
        ("lambda_country", "λ_country"),
        ("lambda_global",  "λ_global"),
    ]
    records = []
    for country, grp in panel.groupby("country"):
        for col, label in variables:
            if col not in grp.columns:
                continue
            mean, med, std, mn, mx, p10, p90 = _pct(grp[col])
            records.append({
                "country":  country,
                "variable": label,
                "mean":     round(mean, 5) if np.isfinite(mean) else np.nan,
                "median":   round(med,  5) if np.isfinite(med)  else np.nan,
                "std":      round(std,  5) if np.isfinite(std)  else np.nan,
                "min":      round(mn,   5) if np.isfinite(mn)   else np.nan,
                "max":      round(mx,   5) if np.isfinite(mx)   else np.nan,
                "p10":      round(p10,  5) if np.isfinite(p10)  else np.nan,
                "p90":      round(p90,  5) if np.isfinite(p90)  else np.nan,
            })

    return pd.DataFrame(records).sort_values(["country", "variable"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Estimated parameters summary
# ---------------------------------------------------------------------------

def build_estimated_parameters_summary(params_df: pd.DataFrame) -> pd.DataFrame:
    """Country-level hazard parameters, global loading share, fit stats."""
    df = params_df.copy()
    df["eta_f"] = df["eta_i"]    # rename for clarity

    with np.errstate(invalid="ignore"):
        denom            = df["eta_f"] + df["eta_g"]
        df["global_share"] = np.where(denom > 0, df["eta_g"] / denom, np.nan)

    df["rmse_bps"] = df["rmse"] * 1e4 if "rmse" in df.columns else np.nan

    keep = ["country", "h0", "eta_f", "eta_g", "global_share"]
    if "R" in df.columns:
        keep.append("R")
    keep += ["rmse_bps", "num_obs", "converged"]
    keep  = [c for c in keep if c in df.columns]

    return df[keep].sort_values("country").reset_index(drop=True)


def params_to_latex(params_summary: pd.DataFrame) -> str:
    """Return a LaTeX tabular string for the estimated parameters table."""
    df = params_summary.copy()
    rename = {
        "country":      "Country",
        "h0":           r"$h_0^*$",
        "eta_f":        r"$\eta_f$",
        "eta_g":        r"$\eta_g$",
        "global_share": r"$\eta_g/(\eta_f+\eta_g)$",
        "R":            r"$R_i$",
        "rmse_bps":     "RMSE (bps)",
        "num_obs":      "$N$",
    }
    df = df.rename(columns=rename)
    # Format floats
    float_cols = [r"$h_0^*$", r"$\eta_f$", r"$\eta_g$",
                  r"$\eta_g/(\eta_f+\eta_g)$", r"$R_i$", "RMSE (bps)"]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    if "$N$" in df.columns:
        df["$N$"] = df["$N$"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    if "converged" in df.columns:
        df = df.drop(columns=["converged"])

    tex = df.to_latex(index=False, escape=False,
                      caption="Estimated sovereign hazard parameters.",
                      label="tab:estimated_params",
                      column_format="l" + "r" * (len(df.columns) - 1))
    return textwrap.dedent(tex)


# ---------------------------------------------------------------------------
# 4. Fit by country
# ---------------------------------------------------------------------------

def build_fit_by_country(model_vs_obs: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE, mean obs/model spread, correlation — per country."""
    records = []
    for country, grp in model_vs_obs.groupby("country"):
        valid = grp.dropna(subset=["spread_obs", "spread_model"])
        if valid.empty:
            continue
        obs   = valid["spread_obs"].to_numpy()
        model = valid["spread_model"].to_numpy()
        resid = model - obs
        corr  = float(np.corrcoef(obs, model)[0, 1]) if len(obs) > 2 else np.nan
        records.append({
            "country":        country,
            "rmse_bps":       round(float(np.sqrt(np.mean(resid ** 2))), 2),
            "mae_bps":        round(float(np.mean(np.abs(resid))), 2),
            "mean_obs_bps":   round(float(np.mean(obs)), 2),
            "mean_model_bps": round(float(np.mean(model)), 2),
            "correlation":    round(corr, 4),
            "n_obs":          int(len(obs)),
        })
    return pd.DataFrame(records).sort_values("country").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Fit by tenor
# ---------------------------------------------------------------------------

def build_fit_by_tenor(model_vs_obs: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE, mean obs/model spread, correlation — per tenor."""
    records = []
    for tenor, grp in model_vs_obs.groupby("tenor"):
        valid = grp.dropna(subset=["spread_obs", "spread_model"])
        if valid.empty:
            continue
        obs   = valid["spread_obs"].to_numpy()
        model = valid["spread_model"].to_numpy()
        resid = model - obs
        corr  = float(np.corrcoef(obs, model)[0, 1]) if len(obs) > 2 else np.nan
        records.append({
            "tenor_y":        float(tenor),
            "rmse_bps":       round(float(np.sqrt(np.mean(resid ** 2))), 2),
            "mae_bps":        round(float(np.mean(np.abs(resid))), 2),
            "mean_obs_bps":   round(float(np.mean(obs)), 2),
            "mean_model_bps": round(float(np.mean(model)), 2),
            "correlation":    round(corr, 4),
            "n_obs":          int(len(obs)),
        })
    return pd.DataFrame(records).sort_values("tenor_y").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Hazard proxy comparison (diagnostic)
# ---------------------------------------------------------------------------

def build_hazard_proxy_comparison(
    panel:     pd.DataFrame,
    params_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare CDS-implied hazard proxy with model hazard rate.

    Proxy (diagnostic approximation):
        ĥ_CDS(5Y) = s_obs(5Y) / (1 − R_i)

    Model hazard:
        h_model = h0 + η_f · λ^i + η_g · λ^g

    This is a diagnostic, not the formal pricing equation.
    """
    if "cds_spread_5y" not in panel.columns:
        print("  WARNING: cds_spread_5y not in panel — skipping hazard proxy table.")
        return pd.DataFrame()

    p_idx  = params_df.set_index("country")
    records = []

    for country, grp in panel.groupby("country"):
        if country not in p_idx.index:
            continue
        row   = p_idx.loc[country]
        if not row.get("converged", True):
            continue

        h0    = float(row["h0"])
        eta_f = float(row["eta_i"])
        eta_g = float(row["eta_g"])
        R_i   = float(row["R"]) if "R" in row.index and pd.notna(row.get("R")) else 0.4

        sub = grp.dropna(subset=["cds_spread_5y", "lambda_country", "lambda_global"]).copy()
        if sub.empty:
            continue

        # CDS-implied hazard proxy (decimal)
        sub["h_cds"] = sub["cds_spread_5y"] / (1.0 - R_i)
        # Model hazard (decimal)
        sub["h_model"] = h0 + eta_f * sub["lambda_country"] + eta_g * sub["lambda_global"]

        h_cds   = sub["h_cds"].to_numpy()
        h_model = sub["h_model"].to_numpy()
        resid   = h_model - h_cds

        corr = float(np.corrcoef(h_cds, h_model)[0, 1]) if len(h_cds) > 2 else np.nan
        records.append({
            "country":              country,
            "corr_model_vs_proxy":  round(corr, 4),
            "mean_model_hazard":    round(float(np.mean(h_model)), 5),
            "mean_cds_hazard_proxy": round(float(np.mean(h_cds)), 5),
            "std_model_hazard":     round(float(np.std(h_model, ddof=1)), 5),
            "std_cds_hazard_proxy": round(float(np.std(h_cds, ddof=1)), 5),
            "mae":                  round(float(np.mean(np.abs(resid))), 5),
            "rmse":                 round(float(np.sqrt(np.mean(resid ** 2))), 5),
            "n_obs":                int(len(sub)),
            "R_used":               round(R_i, 2),
        })

    return pd.DataFrame(records).sort_values("country").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def save_all_tables(
    panel:        pd.DataFrame,
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    out_dir:      str = TABLES_DIR,
) -> None:
    """Build and save all calibration tables to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    print("Building calibration tables …")

    t1 = build_calibration_sample_summary(panel, params_df)
    _save(t1, "calibration_sample_summary.csv", out_dir)

    t2 = build_intensity_summary(panel)
    _save(t2, "intensity_summary.csv", out_dir)

    t3 = build_estimated_parameters_summary(params_df)
    _save(t3, "estimated_parameters_summary.csv", out_dir)
    # LaTeX export
    try:
        tex = params_to_latex(t3)
        tex_path = os.path.join(out_dir, "estimated_parameters_summary.tex")
        with open(tex_path, "w") as fh:
            fh.write(tex)
        print(f"  → {os.path.relpath(tex_path, PARENT_DIR)}")
    except Exception as exc:
        print(f"  WARNING: LaTeX export failed — {exc}")

    t4 = build_fit_by_country(model_vs_obs)
    _save(t4, "fit_by_country.csv", out_dir)

    t5 = build_fit_by_tenor(model_vs_obs)
    _save(t5, "fit_by_tenor.csv", out_dir)

    t6 = build_hazard_proxy_comparison(panel, params_df)
    if not t6.empty:
        _save(t6, "hazard_proxy_comparison.csv", out_dir)

    print(f"Tables saved to {os.path.relpath(out_dir, PARENT_DIR)}/")
