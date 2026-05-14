"""Generate all 10 thesis figures and save them to final/.

Usage:
    cd /Users/aryaman/honors
    python -m plots.generate_all_figures

All figures are written to final/ with clean filenames.

Figure inventory
----------------
Fig 1  fig1_wachter_blowup.png            — blowup of b*(τ), regularisation by default
Fig 2  fig2_yield_term_structures.png     — y*(τ) & y_D*(τ) for 3 λ^f states
Fig 3  fig3_credit_spread_term_structure.png — s*(τ) for 3 λ^f states
Fig 4  fig4_spread_sensitivities.png      — ∂s*/∂λ^f and ∂s*/∂λ^g
Fig 5  fig5_loading_functions.png         — b*(τ), b_{D,f}*(τ) with limits
Fig 6  fig6_yields_with_limits.png        — y*(τ), y_D*(τ) with y*_∞ lines
Fig 7  fig7_longrun_yields_vs_eta_f.png   — y*_∞ (flat) and y_{D,∞}* vs η_f
Fig 8  fig8_cds_vs_bond_spread.png        — s_CDS(τ) vs s*(τ)
Fig 9  fig9_cds_vs_lambda_f.png           — s_CDS(5) vs λ^f
Fig 10 fig10_G_and_K.png                  — G(t,u) and K(t,u) vs u
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution

# ensure repo root is on the path
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.parameters import DisasterModelParams
from core.utils import tau_grid, set_plot_style
from plots.loading_plots import plot_wachter_blowup, plot_loading_functions
from plots.yield_plots import plot_yield_term_structures_3states, plot_yields_with_limits
from plots.spread_plots import (
    plot_credit_spread_term_structure,
    plot_spread_sensitivities,
    plot_long_run_limits_vs_eta_f,
)
from plots.cds_plots import plot_cds_vs_bond_spread, plot_cds_vs_lambda_f, plot_G_and_K


def _make_baseline() -> DisasterModelParams:
    p = DisasterModelParams()
    p.v     = 0.0
    p.b_sdf = None
    p.compute_b_sdf()
    return p


def main():
    set_plot_style()

    params   = _make_baseline()
    lam_f    = params.lam_bar_f
    lam_g    = params.lam_bar_g
    tg       = tau_grid(tau_max=30.0, n=300)

    print("=" * 60)
    print("Generating all thesis figures → final/")
    print("=" * 60)

    # ── Figure 1: Wachter blowup ───────────────────────────────────────────
    # The baseline calibration sits on the trigonometric branch for the
    # risk-free loading (δ(K) < 0), so no separate illustrative params needed.
    print("\n[Fig 1] Wachter blowup ...")
    plot_wachter_blowup(params)

    # ── Figure 2: Yield term structures, 3 states ──────────────────────────
    print("[Fig 2] Yield term structures ...")
    plot_yield_term_structures_3states(params, tg, lam_g)

    # ── Figure 3: Credit spread term structure, 3 states ──────────────────
    print("[Fig 3] Credit spread term structure ...")
    plot_credit_spread_term_structure(params, tg, lam_g)

    # ── Figure 4: Spread sensitivities ────────────────────────────────────
    print("[Fig 4] Spread sensitivities ...")
    plot_spread_sensitivities(params, tg, lam_f, lam_g)

    # ── Figure 5: Loading functions with limits ────────────────────────────
    print("[Fig 5] Loading functions with limits ...")
    plot_loading_functions(params, tg)

    # ── Figure 6: Yields with long-run limits ─────────────────────────────
    print("[Fig 6] Yields with long-run limits ...")
    plot_yields_with_limits(params, tg, lam_f, lam_g)

    # ── Figure 7: Long-run limits vs η_f ─────────────────────────────────
    print("[Fig 7] Long-run yield limits vs η_f ...")
    eta_f_grid = np.linspace(0.5, 4.0, 80)
    plot_long_run_limits_vs_eta_f(params, eta_f_grid)

    # ── Figure 8: CDS vs bond spread term structure ────────────────────────
    print("[Fig 8] CDS vs bond spread (term structure) ...")
    # Use a shorter grid for CDS to keep runtime manageable
    tg_cds = tau_grid(tau_max=20.0, n=80)
    plot_cds_vs_bond_spread(params, tg_cds, lam_f, lam_g)

    # ── Figure 9: CDS at τ₀=5 vs λ^f ─────────────────────────────────────
    print("[Fig 9] CDS spread at τ₀=5 vs λ^f ...")
    lam_f_grid = np.linspace(max(1e-5, 0.2 * lam_f), 3.0 * lam_f, 60)
    plot_cds_vs_lambda_f(params, lam_f_grid, lam_g, tau0=5.0)

    # ── Figure 10: G and K objects ────────────────────────────────────────
    print("[Fig 10] G and K objects ...")
    u_grid = np.linspace(0.01, 20.0, 200)
    plot_G_and_K(params, u_grid, lam_f, lam_g)

    print("\n" + "=" * 60)
    print("All figures saved to final/")
    print("=" * 60)

    # ── Quick sanity numbers ──────────────────────────────────────────────
    from models.defaultable import credit_spread
    from models.cds import fair_cds_spread
    s_bond = credit_spread(params, 5.0, lam_f, lam_g)
    s_cds  = fair_cds_spread(params, 5.0, lam_f, lam_g)
    print(f"\n5Y bond spread: {1e4*s_bond:.2f} bp")
    print(f"5Y CDS spread:  {1e4*s_cds:.2f} bp")


if __name__ == "__main__":
    main()
