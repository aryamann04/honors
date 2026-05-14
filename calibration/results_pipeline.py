"""
results_pipeline.py
===================
Orchestrator for the full results and simulation pipeline.

Assumes the calibration pipeline has already been run:
    python -m calibration.run_calibration [--skip-lseg]

This script loads the calibration outputs and produces all thesis-ready
tables and figures.

Usage
-----
    python -m calibration.results_pipeline                  # tables + calibration figures
    python -m calibration.results_pipeline --run-simulation # also run CIR simulation
    python -m calibration.results_pipeline --countries US DE JP MX --run-simulation
    python -m calibration.results_pipeline --skip-build     # skip, just re-run tables/plots

Output layout
-------------
    calibration/results/tables/
        calibration_sample_summary.csv
        intensity_summary.csv
        estimated_parameters_summary.csv   (+.tex)
        fit_by_country.csv
        fit_by_tenor.csv
        hazard_proxy_comparison.csv
        moment_comparison.csv              (simulation only)
        simulated_spread_moments_by_tenor.csv
        variance_decomposition.csv

    calibration/results/figures/
        valuation_vs_intensity.png
        global_intensity_timeseries.png
        country_intensity_panels.png
        estimated_parameters.png
        model_vs_observed_scatter.png
        model_vs_observed_5y_timeseries.png
        residuals_by_tenor.png
        residuals_over_time.png
        cds_data_coverage_heatmap.png
        rmse_by_country_tenor.png
        latest_cds_term_structure_fit.png
        hazard_proxy_scatter.png
        hazard_proxy_timeseries.png
        simulated_paths_representative_country.png  (simulation only)
        flight_to_quality_mechanism.png
        cds_term_structure_by_disaster_state.png
        yield_curves_by_disaster_state.png
        spread_variance_decomposition.png
        hazard_contribution_decomposition.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import pandas as pd

# Calibration result paths
PANEL_PATH    = os.path.join(BASE_DIR, "processed_data", "panel.csv")
PARAMS_PATH   = os.path.join(BASE_DIR, "results", "parameters.csv")
MODEL_OBS_PATH = os.path.join(BASE_DIR, "results", "model_vs_observed.csv")

# Output directories
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
FIGS_DIR   = os.path.join(BASE_DIR, "results", "figures")

DIVIDER = "=" * 64


def _header(msg: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {msg}")
    print(DIVIDER)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load panel, parameters, and model_vs_observed. Raises if missing."""
    missing = [p for p in [PANEL_PATH, PARAMS_PATH, MODEL_OBS_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Calibration outputs not found:\n" +
            "\n".join(f"  {p}" for p in missing) +
            "\n\nRun  python -m calibration.run_calibration  first."
        )

    panel        = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    params_df    = pd.read_csv(PARAMS_PATH)
    model_vs_obs = pd.read_csv(MODEL_OBS_PATH, parse_dates=["date"])

    print(f"  Panel:          {len(panel):,} rows, "
          f"{panel['country'].nunique()} countries")
    print(f"  Parameters:     {len(params_df)} countries "
          f"({params_df.get('converged', pd.Series(True)).fillna(False).sum()} converged)")
    print(f"  Model-vs-obs:   {len(model_vs_obs):,} (date × country × tenor) pairs")
    return panel, params_df, model_vs_obs


def run_tables(
    panel:        pd.DataFrame,
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
) -> None:
    from calibration.results_tables import save_all_tables
    _header("STEP: Calibration tables")
    save_all_tables(panel, params_df, model_vs_obs, TABLES_DIR)


def run_calibration_plots(
    panel:        pd.DataFrame,
    params_df:    pd.DataFrame,
    model_vs_obs: pd.DataFrame,
    countries:    list[str] | None,
) -> None:
    from calibration.results_plots import save_all_plots
    _header("STEP: Calibration figures")
    save_all_plots(panel, params_df, model_vs_obs, FIGS_DIR, countries)


def run_simulation(
    panel:     pd.DataFrame,
    params_df: pd.DataFrame,
    countries: list[str] | None,
    n_months:  int,
) -> dict:
    from core.parameters import DisasterModelParams
    from calibration.simulation import run_all_countries

    _header(f"STEP: CIR simulation  (n_months={n_months})")
    base_params = DisasterModelParams()
    base_params.compute_b_sdf()
    print(f"  b_sdf = {base_params.b_sdf:.6f}")

    sim_results = run_all_countries(
        params_df   = params_df,
        panel       = panel,
        base_params = base_params,
        n_months    = n_months,
        countries   = countries,
    )
    print(f"  Simulated {len(sim_results)} countries.")
    return sim_results, base_params


def run_simulation_tables(
    sim_results: dict,
    panel:       pd.DataFrame,
    base_params,
) -> None:
    from calibration.simulation_tables import save_all_simulation_tables
    _header("STEP: Simulation tables")
    save_all_simulation_tables(sim_results, panel, base_params, TABLES_DIR)


def run_simulation_plots(
    sim_results: dict,
    base_params,
    params_df:   pd.DataFrame,
    panel:       pd.DataFrame,
    countries:   list[str] | None,
) -> None:
    from calibration.simulation_plots import save_all_simulation_plots
    _header("STEP: Simulation figures")
    save_all_simulation_plots(
        sim_results = sim_results,
        base_params = base_params,
        params_df   = params_df,
        panel       = panel,
        out_dir     = FIGS_DIR,
        countries   = countries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis results tables and figures from calibration outputs."
    )
    parser.add_argument("--skip-build",    action="store_true",
                        help="Skip loading check (assume inputs exist).")
    parser.add_argument("--run-simulation", action="store_true",
                        help="Also run CIR simulation and produce simulation outputs.")
    parser.add_argument("--run-diagnostics", action="store_true",
                        help="Run yield-curve slope and long-maturity convergence diagnostics.")
    parser.add_argument("--countries",     nargs="+", default=None,
                        metavar="ISO2",
                        help="Restrict to these countries (ISO-2 codes). Default: all.")
    parser.add_argument("--n-months",      type=int, default=20_000,
                        help="Simulation length in months (default: 20000).")
    args = parser.parse_args()

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR,   exist_ok=True)

    t0 = time.time()

    _header("Loading calibration outputs")
    panel, params_df, model_vs_obs = _load_inputs()

    if args.countries:
        panel        = panel[panel["country"].isin(args.countries)].copy()
        params_df    = params_df[params_df["country"].isin(args.countries)].copy()
        model_vs_obs = model_vs_obs[model_vs_obs["country"].isin(args.countries)].copy()
        print(f"  Restricted to: {args.countries}")

    run_tables(panel, params_df, model_vs_obs)
    run_calibration_plots(panel, params_df, model_vs_obs, args.countries)

    if args.run_diagnostics:
        from core.parameters import DisasterModelParams
        from calibration.yield_curve_diagnostics import run_yield_diagnostics
        _header("STEP: Yield-curve diagnostics")
        base_params = DisasterModelParams()
        base_params.compute_b_sdf()
        run_yield_diagnostics(panel, params_df, base_params,
                              tables_dir=TABLES_DIR, figs_dir=FIGS_DIR)

    if args.run_simulation:
        sim_results, base_params = run_simulation(panel, params_df,
                                                  args.countries, args.n_months)
        run_simulation_tables(sim_results, panel, base_params)
        run_simulation_plots(sim_results, base_params, params_df, panel, args.countries)
    else:
        # Still produce flight-to-quality and term-structure plots without full simulation
        try:
            from core.parameters import DisasterModelParams
            from calibration.simulation_plots import (
                plot_flight_to_quality, plot_cds_term_structure_by_state,
                plot_yield_curves_by_state, plot_hazard_contribution_decomposition,
            )
            _header("STEP: Static model figures (no simulation)")
            base_params = DisasterModelParams()
            base_params.compute_b_sdf()
            plot_flight_to_quality(base_params, params_df, panel, FIGS_DIR)
            plot_hazard_contribution_decomposition({}, panel, params_df, FIGS_DIR)
        except Exception as exc:
            print(f"  WARNING: static model figures skipped — {exc}")

    elapsed = time.time() - t0
    _header(f"RESULTS PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"  Tables:  {os.path.relpath(TABLES_DIR, PARENT_DIR)}/")
    print(f"  Figures: {os.path.relpath(FIGS_DIR,   PARENT_DIR)}/\n")


if __name__ == "__main__":
    main()
