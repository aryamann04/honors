"""
run_calibration.py
==================
Full calibration pipeline runner.

Executes in order:
  1. calibration.lseg_cape          — fetch CAPE data from LSEG
  2. calibration.extract_intensities — CAPE → λ intensities
  3. calibration.build_panel         — merge intensities + CDS
  4. calibration.calibrate_cds       — NLS calibration per country
  5. calibration.plot_results        — diagnostic plots

Each step is run as a subprocess so failures are isolated and clearly reported.
The runner stops immediately if any step fails.

Usage
-----
    python -m calibration.run_calibration [--skip-lseg]

Options
-------
  --skip-lseg   Skip the LSEG data fetch (use existing raw_data/country_cape_panel.csv).
                Useful when LSEG credentials are unavailable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

BASE_DIR   = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)

# Expected output files produced by each stage
STAGE_OUTPUTS: dict[str, list[str]] = {
    "lseg_cape":           [os.path.join(BASE_DIR, "raw_data", "country_cape_panel.csv")],
    "extract_intensities": [os.path.join(BASE_DIR, "processed_data", "intensities.csv")],
    "build_panel":         [os.path.join(BASE_DIR, "processed_data", "panel.csv")],
    "calibrate_cds":       [
        os.path.join(BASE_DIR, "results", "parameters.csv"),
        os.path.join(BASE_DIR, "results", "model_vs_observed.csv"),
    ],
    "plot_results":        [],   # verified by checking *.png existence
}

DIVIDER = "=" * 68


def _print_header(msg: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {msg}")
    print(DIVIDER)


def _run_step(module: str, extra_args: list[str] | None = None) -> None:
    """Run  python -m calibration.<module>  as a subprocess.  Raises on failure."""
    cmd = [sys.executable, "-m", f"calibration.{module}"] + (extra_args or [])
    _print_header(f"STEP: python -m calibration.{module}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PARENT_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{module}' failed with exit code {result.returncode} "
            f"(elapsed {elapsed:.1f}s)."
        )
    print(f"\n  ✓ {module} completed in {elapsed:.1f}s")


def _verify_outputs(stage: str) -> None:
    """Check that expected output files exist after a step completes."""
    expected = STAGE_OUTPUTS.get(stage, [])
    missing  = [p for p in expected if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Step '{stage}' finished but expected outputs are missing:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    for p in expected:
        size_kb = os.path.getsize(p) / 1024
        print(f"  Output: {os.path.relpath(p, PARENT_DIR)}  ({size_kb:.1f} KB)")


def _print_cape_diagnostics() -> None:
    import pandas as pd
    path = os.path.join(BASE_DIR, "raw_data", "country_cape_panel.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["cape"].notna()]
    print("\nCAPE panel summary:")
    print(f"  Countries: {sorted(df['country'].unique())}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    for country, grp in df.groupby("country"):
        latest = grp.sort_values("date").iloc[-1]
        print(f"  {country}: {len(grp)} obs, latest CAPE = {latest['cape']:.2f} "
              f"({latest['date'].date()})")


def _print_intensity_diagnostics() -> None:
    import pandas as pd, numpy as np
    path = os.path.join(BASE_DIR, "processed_data", "intensities.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, parse_dates=["date"])
    print("\nIntensity summary (annualised %):")
    for col in ["lambda_total", "lambda_global", "lambda_country"]:
        arr = df[col].dropna().to_numpy()
        if len(arr) == 0:
            continue
        print(f"  {col}: min={arr.min()*100:.3f}  p25={np.percentile(arr,25)*100:.3f}  "
              f"med={np.median(arr)*100:.3f}  p75={np.percentile(arr,75)*100:.3f}  "
              f"max={arr.max()*100:.3f}")
    # Check global vs total scale
    g_mean = df.groupby("date")["lambda_global"].first().mean()
    t_mean = df.groupby("date")["lambda_total"].mean().mean()
    ratio  = g_mean / t_mean if t_mean > 0 else float("nan")
    print(f"  λ^g mean / λ^total mean = {ratio:.3f}")


def _print_panel_diagnostics() -> None:
    import pandas as pd
    path = os.path.join(BASE_DIR, "processed_data", "panel.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    cds_cols = [c for c in df.columns if c.startswith("cds_spread_")]
    print("\nPanel summary:")
    print(f"  Countries: {sorted(df['country'].unique())}")
    print(f"  Total rows: {len(df):,}")
    print(f"  CDS tenors available: {cds_cols}")
    print("\n  Missing CDS share by country:")
    for country, grp in df.groupby("country"):
        miss = {c.replace("cds_spread_", ""): f"{grp[c].isna().mean()*100:.0f}%"
                for c in cds_cols}
        print(f"    {country:4s}: {miss}")


def _print_calibration_diagnostics() -> None:
    import pandas as pd
    path = os.path.join(BASE_DIR, "results", "parameters.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    print("\nCalibration results:")
    failed = df[~df["converged"]] if "converged" in df.columns else pd.DataFrame()
    if not failed.empty:
        print(f"  WARNING: {len(failed)} countries did not converge: {failed['country'].tolist()}")
    fitted = df[df["converged"]] if "converged" in df.columns else df
    if not fitted.empty:
        print(f"  Converged: {fitted['country'].tolist()}")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(fitted[["country", "h0", "eta_i", "eta_g", "rmse", "num_obs"]]
              .to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def _print_plot_diagnostics() -> None:
    import glob
    pngs = glob.glob(os.path.join(BASE_DIR, "results", "*.png"))
    print(f"\nPlots saved ({len(pngs)}):")
    for p in sorted(pngs):
        size_kb = os.path.getsize(p) / 1024
        print(f"  {os.path.basename(p)}  ({size_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full calibration pipeline.")
    parser.add_argument(
        "--skip-lseg", action="store_true",
        help="Skip LSEG data fetch (requires existing raw_data/country_cape_panel.csv).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE_DIR, "raw_data"),       exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "processed_data"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"),        exist_ok=True)

    total_start = time.time()

    try:
        # ── Step 1: LSEG CAPE fetch ──────────────────────────────────────
        if args.skip_lseg:
            _print_header("STEP: lseg_cape  [SKIPPED — using existing country_cape_panel.csv]")
            cape_out = os.path.join(BASE_DIR, "raw_data", "country_cape_panel.csv")
            # Fall back to root-level file if needed
            if not os.path.exists(cape_out):
                fallback = os.path.join(PARENT_DIR, "country_cape_panel.csv")
                if os.path.exists(fallback):
                    import shutil
                    shutil.copy(fallback, cape_out)
                    print(f"  Copied {fallback} → {cape_out}")
                else:
                    raise FileNotFoundError(
                        f"--skip-lseg set but no cape_panel.csv found at "
                        f"{cape_out} or {fallback}"
                    )
        else:
            _run_step("lseg_cape")
            _verify_outputs("lseg_cape")

        _print_cape_diagnostics()

        # ── Step 2: Intensity extraction ─────────────────────────────────
        _run_step("extract_intensities")
        _verify_outputs("extract_intensities")
        _print_intensity_diagnostics()

        # ── Step 3: Panel build ──────────────────────────────────────────
        _run_step("build_panel")
        _verify_outputs("build_panel")
        _print_panel_diagnostics()

        # ── Step 4: CDS calibration ──────────────────────────────────────
        _run_step("calibrate_cds")
        _verify_outputs("calibrate_cds")
        _print_calibration_diagnostics()

        # ── Step 5: Plots ────────────────────────────────────────────────
        _run_step("plot_results")
        _verify_outputs("plot_results")
        _print_plot_diagnostics()

    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\n{'!' * 68}")
        print(f"  PIPELINE FAILED: {exc}")
        print(f"{'!' * 68}\n")
        sys.exit(1)

    total_elapsed = time.time() - total_start
    _print_header(f"PIPELINE COMPLETE  (total: {total_elapsed/60:.1f} min)")
    print(f"  Processed data: {os.path.join(BASE_DIR, 'processed_data')}")
    print(f"  Results:        {os.path.join(BASE_DIR, 'results')}\n")


if __name__ == "__main__":
    main()
