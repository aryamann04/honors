import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

DATA_DIR = "/Users/aryaman/honors/calibration/data"
LAMBDA_PATH = os.path.join(DATA_DIR, "lambda_t_series.csv")
CSV_PATH = os.path.join(DATA_DIR, "CDS spreads.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "CDS_USD_5Y.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import calibration.extract_intensities as ei
import calibration.CDS_data as cdsdata

USECOLS = [
    "date",
    "ticker",
    "shortname",
    "region",
    "country",
    "avrating",
    "tier",
    "currency",
    "docclause",
    "runningcoupon",
    "tenor",
    "parspread",
    "cdsassumedrecovery",
    "dp",
    "jtd",
]

def load_cds_data():
    if os.path.exists(PARQUET_PATH):
        cds = pd.read_parquet(PARQUET_PATH)
    else:
        cds = cdsdata.build_filtered_parquet_from_csv()

    cds["date"] = pd.to_datetime(cds["date"], errors="coerce")
    cds["parspread"] = pd.to_numeric(cds["parspread"], errors="coerce")
    cds["cdsassumedrecovery"] = pd.to_numeric(cds["cdsassumedrecovery"], errors="coerce")
    cds["dp"] = pd.to_numeric(cds["dp"], errors="coerce")
    cds["jtd"] = pd.to_numeric(cds["jtd"], errors="coerce")

    cds = cds.dropna(subset=["date", "country", "parspread"]).copy()
    cds = cds.sort_values(["country", "date"]).reset_index(drop=True)
    return cds


def normalize_recovery(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").copy()
    mask_pct = x > 1.0
    x.loc[mask_pct] = x.loc[mask_pct] / 100.0
    return x.clip(lower=0.0, upper=0.99)

def prepare_country_month_panel(cds: pd.DataFrame, lambda_monthly: pd.DataFrame):
    cds = cds.copy()
    cds["date"] = pd.to_datetime(cds["date"], errors="coerce")
    cds["recovery"] = normalize_recovery(cds["cdsassumedrecovery"])
    cds = cds[cds["recovery"].notna()].copy()
    cds = cds[cds["recovery"] < 1.0].copy()

    cds["hazard_proxy"] = cds["parspread"] / (1.0 - cds["recovery"])
    cds["month"] = cds["date"].dt.to_period("M")

    country_month = (
        cds.groupby(["country", "month"], as_index=False)
        .agg(
            parspread=("parspread", "mean"),
            recovery=("recovery", "mean"),
            hazard_proxy=("hazard_proxy", "mean"),
            dp=("dp", "mean"),
            jtd=("jtd", "mean"),
            n_daily_obs=("parspread", "size"),
        )
        .sort_values(["country", "month"])
        .reset_index(drop=True)
    )

    lambda_monthly = lambda_monthly.copy()

    if "Date" in lambda_monthly.columns:
        lambda_monthly["Date"] = pd.to_datetime(lambda_monthly["Date"], errors="coerce")
        lambda_monthly["month"] = lambda_monthly["Date"].dt.to_period("M")
    elif "date" in lambda_monthly.columns:
        lambda_monthly["date"] = pd.to_datetime(lambda_monthly["date"], errors="coerce")
        lambda_monthly["month"] = lambda_monthly["date"].dt.to_period("M")
    elif "month" not in lambda_monthly.columns:
        raise KeyError("lambda_monthly must contain either 'Date', 'date', or 'month'.")

    lambda_monthly = (
        lambda_monthly.groupby("month", as_index=False)["lambda_t"]
        .mean()
        .sort_values("month")
        .reset_index(drop=True)
    )

    lambda_monthly["d_lambda_t"] = lambda_monthly["lambda_t"].diff()

    panel = country_month.merge(lambda_monthly, on="month", how="inner")
    panel["month_end"] = panel["month"].dt.to_timestamp("M")

    panel = panel.sort_values(["country", "month"]).reset_index(drop=True)
    panel["d_hazard_proxy"] = panel.groupby("country")["hazard_proxy"].diff()
    panel["d_parspread"] = panel.groupby("country")["parspread"].diff()

    return panel


def fit_ols_with_hc1(y: np.ndarray, x: np.ndarray):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC1")
    return model, robust


def run_country_level_regression(df_country: pd.DataFrame):
    df = df_country.dropna(subset=["hazard_proxy", "lambda_t"]).copy()
    if len(df) < 24:
        return None

    x = df["lambda_t"].astype(float).to_numpy()
    y = df["hazard_proxy"].astype(float).to_numpy()

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None

    model, robust = fit_ols_with_hc1(y, x)

    return {
        "country": df["country"].iloc[0],
        "specification": "levels",
        "lhs": "hazard_proxy",
        "rhs": "lambda_t",
        "nobs": int(model.nobs),
        "intercept": float(model.params[0]),
        "slope": float(model.params[1]),
        "intercept_se": float(robust.bse[0]),
        "slope_se": float(robust.bse[1]),
        "intercept_t": float(robust.tvalues[0]),
        "slope_t": float(robust.tvalues[1]),
        "intercept_p": float(robust.pvalues[0]),
        "slope_p": float(robust.pvalues[1]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(model.resid ** 2))),
        "mean_lhs": float(np.mean(y)),
        "std_lhs": float(np.std(y, ddof=1)),
        "mean_rhs": float(np.mean(x)),
        "std_rhs": float(np.std(x, ddof=1)),
        "mean_spread": float(df["parspread"].mean()),
        "mean_recovery": float(df["recovery"].mean()),
    }


def run_country_diff_regression(df_country: pd.DataFrame):
    df = df_country.dropna(subset=["d_hazard_proxy", "d_lambda_t"]).copy()
    if len(df) < 24:
        return None

    x = df["d_lambda_t"].astype(float).to_numpy()
    y = df["d_hazard_proxy"].astype(float).to_numpy()

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None

    model, robust = fit_ols_with_hc1(y, x)

    return {
        "country": df["country"].iloc[0],
        "specification": "first_differences",
        "lhs": "d_hazard_proxy",
        "rhs": "d_lambda_t",
        "nobs": int(model.nobs),
        "intercept": float(model.params[0]),
        "slope": float(model.params[1]),
        "intercept_se": float(robust.bse[0]),
        "slope_se": float(robust.bse[1]),
        "intercept_t": float(robust.tvalues[0]),
        "slope_t": float(robust.tvalues[1]),
        "intercept_p": float(robust.pvalues[0]),
        "slope_p": float(robust.pvalues[1]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(model.resid ** 2))),
        "mean_lhs": float(np.mean(y)),
        "std_lhs": float(np.std(y, ddof=1)),
        "mean_rhs": float(np.mean(x)),
        "std_rhs": float(np.std(x, ddof=1)),
        "mean_spread": float(df["parspread"].mean()),
        "mean_recovery": float(df["recovery"].mean()),
    }


def run_all_regressions(panel: pd.DataFrame):
    level_results = []
    diff_results = []

    for country, df_country in panel.groupby("country"):
        level_out = run_country_level_regression(df_country)
        if level_out is not None:
            level_results.append(level_out)

        diff_out = run_country_diff_regression(df_country)
        if diff_out is not None:
            diff_results.append(diff_out)

    level_df = pd.DataFrame(level_results)
    diff_df = pd.DataFrame(diff_results)

    if level_df.empty:
        raise ValueError("No valid level regressions were produced.")
    if diff_df.empty:
        raise ValueError("No valid first-difference regressions were produced.")

    level_df = level_df.sort_values(["r2", "nobs"], ascending=[False, False]).reset_index(drop=True)
    diff_df = diff_df.sort_values(["r2", "nobs"], ascending=[False, False]).reset_index(drop=True)

    return level_df, diff_df


def add_readable_columns(df: pd.DataFrame):
    out = df.copy()
    out["eta_i"] = out["slope"]
    out["eta_se"] = out["slope_se"]
    out["h0_i"] = out["intercept"]
    out["h0_se"] = out["intercept_se"]
    return out


def print_results_table(results_df: pd.DataFrame, title: str):
    display_cols = [
        "country",
        "nobs",
        "h0_i",
        "h0_se",
        "eta_i",
        "eta_se",
        "slope_t",
        "slope_p",
        "r2",
        "adj_r2",
        "rmse",
        "mean_spread",
        "mean_recovery",
    ]

    pretty = results_df[display_cols].copy()

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print(f"\n{title}\n")
    print(pretty.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))


def print_top_lists(level_df: pd.DataFrame, diff_df: pd.DataFrame):
    print("\nTop 15 countries by level-regression R^2:\n")
    print(
        level_df[["country", "eta_i", "r2", "adj_r2", "nobs"]]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x:,.6f}")
    )

    print("\nTop 15 countries by first-difference R^2:\n")
    print(
        diff_df[["country", "eta_i", "r2", "adj_r2", "nobs"]]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x:,.6f}")
    )

    print("\nTop 15 countries by absolute first-difference eta_i:\n")
    diff_abs = diff_df.assign(abs_eta=lambda d: d["eta_i"].abs()).sort_values("abs_eta", ascending=False)
    print(
        diff_abs[["country", "eta_i", "abs_eta", "r2", "nobs"]]
        .head(15)
        .to_string(index=False, float_format=lambda x: f"{x:,.6f}")
    )


def main():

    if os.path.exists(LAMBDA_PATH):
        lambda_monthly = pd.read_csv(LAMBDA_PATH, parse_dates=["Date"])
    else:
        print("no lambda series exists; building...")
        lambda_monthly = ei.build_and_save_lambda_series(LAMBDA_PATH)
        print("built lambda series")

    cds = load_cds_data()
    panel = prepare_country_month_panel(cds, lambda_monthly)

    level_df, diff_df = run_all_regressions(panel)
    level_df = add_readable_columns(level_df)
    diff_df = add_readable_columns(diff_df)

    print_results_table(
        level_df,
        "Country-by-country level regressions: hazard_pctpts on lambda_pctpts",
    )

    print_results_table(
        diff_df,
        "Country-by-country robustness regressions: d_hazard_pctpts on d_lambda_pctpts",
    )

    print_top_lists(level_df, diff_df)

    panel_out = os.path.join(RESULTS_DIR, "country_lambda_hazard_panel.csv")
    level_out = os.path.join(RESULTS_DIR, "country_eta_regressions_levels.csv")
    diff_out = os.path.join(RESULTS_DIR, "country_eta_regressions_first_differences.csv")
    combined_out = os.path.join(RESULTS_DIR, "country_eta_regressions_combined.csv")

    panel.to_csv(panel_out, index=False)
    level_df.to_csv(level_out, index=False)
    diff_df.to_csv(diff_out, index=False)
    pd.concat([level_df, diff_df], ignore_index=True).to_csv(combined_out, index=False)

    print(f"\nSaved merged panel to: {panel_out}")
    print(f"Saved level regressions to: {level_out}")
    print(f"Saved first-difference regressions to: {diff_out}")
    print(f"Saved combined regression results to: {combined_out}")


if __name__ == "__main__":
    main()