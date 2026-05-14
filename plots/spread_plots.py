"""Credit spread plots (Figures 3, 4, 7).

Figure 3 — Credit spread term structure
    s*(τ) for three λ^f states (low / baseline / high)

Figure 4 — Credit spread sensitivities
    ∂s*/∂λ^f  and  ∂s*/∂λ^g  as functions of τ
    (central finite differences, step 1e-5)

Figure 7 — Long-run yield limits vs η_f
    y*_∞ (flat: independent of η_f because b*_∞ depends only on K)
    y_{D,∞}* (increasing in η_f)
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

from core.parameters import DisasterModelParams
from core.closed_form import y_star_inf, y_D_star_inf
from models.defaultable import credit_spread_curve, spread_sensitivity
from core.utils import savefig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Credit spread term structure
# ──────────────────────────────────────────────────────────────────────────────

def plot_credit_spread_term_structure(params: DisasterModelParams,
                                      tau_grid: np.ndarray,
                                      lam_g: float):
    """Figure 3: s*(τ) for three λ^f states."""
    lam_f_vals   = [0.5 * params.lam_bar_f,
                    params.lam_bar_f,
                    2.0 * params.lam_bar_f]
    state_labels = [r"$\lambda^f$ low",
                    r"$\lambda^f$ baseline",
                    r"$\lambda^f$ high"]
    colors       = ["steelblue", "darkorange", "crimson"]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    for lf, col, lab in zip(lam_f_vals, colors, state_labels):
        s = credit_spread_curve(params, tau_grid, lf, lam_g)
        ax.plot(tau_grid, 1e4 * s, lw=2.2, color=col, label=lab)

    ax.set_xlabel(r"Maturity $\tau$ (years)", fontsize=12)
    ax.set_ylabel(r"Credit spread $s^*(\tau)$ (bp)", fontsize=12)
    ax.set_title(r"Credit spread term structure for three $\lambda^f$ states", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)

    savefig("fig3_credit_spread_term_structure.png")
    print("Saved: fig3_credit_spread_term_structure.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Spread sensitivities
# ──────────────────────────────────────────────────────────────────────────────

def plot_spread_sensitivities(params: DisasterModelParams,
                              tau_grid: np.ndarray,
                              lam_f: float,
                              lam_g: float):
    """Figure 4: ∂s*/∂λ^f and ∂s*/∂λ^g vs τ (central finite differences)."""
    dsf, dsg = spread_sensitivity(params, tau_grid, lam_f, lam_g)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)

    axes[0].plot(tau_grid, 1e4 * dsf, lw=2.2, color="steelblue")
    axes[0].set_title(r"$\partial s^*(\tau) / \partial \lambda^f$", fontsize=12)
    axes[0].set_xlabel(r"Maturity $\tau$ (years)", fontsize=11)
    axes[0].set_ylabel(r"bp per unit of $\lambda^f$", fontsize=11)
    axes[0].grid(alpha=0.28)

    axes[1].plot(tau_grid, 1e4 * dsg, lw=2.2, color="darkorange")
    axes[1].set_title(r"$\partial s^*(\tau) / \partial \lambda^g$", fontsize=12)
    axes[1].set_xlabel(r"Maturity $\tau$ (years)", fontsize=11)
    axes[1].set_ylabel(r"bp per unit of $\lambda^g$", fontsize=11)
    axes[1].grid(alpha=0.28)

    fig.suptitle("Credit spread sensitivities to disaster intensities", fontsize=12)
    savefig("fig4_spread_sensitivities.png")
    print("Saved: fig4_spread_sensitivities.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 — Long-run yield limits vs η_f
# ──────────────────────────────────────────────────────────────────────────────

def plot_long_run_limits_vs_eta_f(params: DisasterModelParams,
                                   eta_f_grid: np.ndarray):
    """Figure 7: y*_∞ (flat) and y_{D,∞}* (increasing) vs η_f.

    y*_∞ is independent of η_f because b*_∞ = (−φ − δ(K))/σ² depends only on K,
    and K = exp(−γZ)(1 − exp Z) does not contain η_f.
    """
    y_rf_inf_vals = []
    y_D_inf_vals  = []

    for eta_f in eta_f_grid:
        p       = copy.copy(params)
        p.eta1  = float(eta_f)
        p.b_sdf = None          # force recomputation for changed params
        p.compute_b_sdf()
        y_rf_inf_vals.append(y_star_inf(p))
        y_D_inf_vals.append(y_D_star_inf(p))

    # y_star_inf may be None if on trig branch; replace with nan for plotting
    y_rf_arr = np.array([v if v is not None else np.nan for v in y_rf_inf_vals])
    y_D_arr  = np.array([v if v is not None else np.nan for v in y_D_inf_vals])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(eta_f_grid, 100.0 * y_rf_arr, lw=2.2, color="steelblue",
            label=r"$y^*_\infty$ (risk-free, flat)")
    ax.plot(eta_f_grid, 100.0 * y_D_arr,  lw=2.2, color="darkorange", ls="--",
            label=r"$y^*_{D,\infty}$ (defaultable, increasing)")

    ax.set_xlabel(r"Hazard loading $\eta_f$", fontsize=12)
    ax.set_ylabel(r"Long-run yield (%)", fontsize=12)
    ax.set_title(r"Long-run yield limits $y^*_\infty$ and $y^*_{D,\infty}$ vs $\eta_f$",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)

    savefig("fig7_longrun_yields_vs_eta_f.png")
    if np.all(np.isnan(y_rf_arr)):
        rf_range_str = "all NaN (trig branch — no finite limit)"
    else:
        rf_range = np.nanmax(y_rf_arr) - np.nanmin(y_rf_arr)
        rf_range_str = f"{1e4*rf_range:.3f} bp — should be ~0 if exp branch"
    print(f"Saved: fig7_longrun_yields_vs_eta_f.png  (y*_∞ range across η_f: {rf_range_str})")
