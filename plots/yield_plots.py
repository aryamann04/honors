"""Yield term-structure plots (Figures 2 and 6).

Figure 2 — Risk-free and defaultable yield term structures
    y*(τ) and y_D*(τ) for three λ^f states (low / baseline / high)
    Solid lines = risk-free;  dashed = defaultable
    Vertical gap is the credit spread s*(τ)

Figure 6 — Yields with long-run limits
    y*(τ) and y_D*(τ) for the baseline state
    Horizontal dashed lines at y*_∞ and y_{D,∞}*
"""
import numpy as np
import matplotlib.pyplot as plt

from core.parameters import DisasterModelParams
from core.closed_form import y_star_inf, y_D_star_inf, _phi_sig2, K_const
from core.riccati import blowup_time, discriminant
from models.risk_free import rf_yield_curve
from models.defaultable import def_yield_curve
from core.utils import savefig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Three λ^f states
# ──────────────────────────────────────────────────────────────────────────────

def plot_yield_term_structures_3states(params: DisasterModelParams,
                                       tau_grid: np.ndarray,
                                       lam_g: float):
    """Figure 2: y*(τ) and y_D*(τ) for λ^f ∈ {low, baseline, high}."""
    lam_f_vals  = [0.5 * params.lam_bar_f,
                   params.lam_bar_f,
                   2.0 * params.lam_bar_f]
    colors      = ["steelblue", "darkorange", "crimson"]
    state_labels = [r"$\lambda^f$ low",
                    r"$\lambda^f$ baseline",
                    r"$\lambda^f$ high"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for lf, col, lab in zip(lam_f_vals, colors, state_labels):
        y_rf  = rf_yield_curve(params, tau_grid, lf, lam_g)
        y_def = def_yield_curve(params, tau_grid, lf, lam_g)
        ax.plot(tau_grid, 100.0 * y_rf,  color=col, lw=2.0,
                label=rf"$y^*(\tau)$ — {lab}")
        ax.plot(tau_grid, 100.0 * y_def, color=col, lw=2.0, ls="--",
                label=rf"$y^*_D(\tau)$ — {lab}")

    # legend: solid = RF, dashed = defaultable
    ax.plot([], [], "k-",  lw=1.8, label="Solid = risk-free")
    ax.plot([], [], "k--", lw=1.8, label="Dashed = defaultable")

    ax.set_xlabel(r"Maturity $\tau$ (years)", fontsize=12)
    ax.set_ylabel(r"Yield (%)", fontsize=12)
    ax.set_title(r"Yield term structures for three $\lambda^f$ states", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.28)

    savefig("fig2_yield_term_structures.png")
    print("Saved: fig2_yield_term_structures.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Yields with long-run limits
# ──────────────────────────────────────────────────────────────────────────────

def plot_yields_with_limits(params: DisasterModelParams,
                            tau_grid: np.ndarray,
                            lam_f: float,
                            lam_g: float):
    """Figure 6: y*(τ) and y_D*(τ) for the baseline state, with asymptotes."""
    y_rf  = rf_yield_curve(params, tau_grid, lam_f, lam_g)
    y_def = def_yield_curve(params, tau_grid, lam_f, lam_g)

    y_inf    = y_star_inf(params)
    y_D_inf  = y_D_star_inf(params)

    # blowup time for RF (used if y_inf is None — trig branch)
    phi, sig2 = _phi_sig2(params)
    K         = K_const(params)
    blowup_tau = blowup_time(phi, sig2, K) if discriminant(K, phi, sig2) < 0 else None

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(tau_grid, 100.0 * y_rf,  lw=2.2, color="steelblue",
            label=r"$y^*(\tau)$ risk-free")
    ax.plot(tau_grid, 100.0 * y_def, lw=2.2, color="darkorange", ls="--",
            label=r"$y^*_D(\tau)$ defaultable")

    if y_inf is not None:
        ax.axhline(100.0 * y_inf, ls=":", lw=1.6, color="steelblue",
                   label=rf"$y^*_{{\infty}} = {100*y_inf:.3f}\%$")
    else:
        ax.axvline(100.0 * blowup_tau, ls=":", lw=1.6, color="steelblue",
                   label=rf"RF blowup $\tau^* \approx {blowup_tau:.1f}$ yr")

    if y_D_inf is not None:
        ax.axhline(100.0 * y_D_inf, ls=":", lw=1.6, color="darkorange",
                   label=rf"$y^*_{{D,\infty}} = {100*y_D_inf:.3f}\%$")

    ax.set_xlabel(r"Maturity $\tau$ (years)", fontsize=12)
    ax.set_ylabel(r"Yield (%)", fontsize=12)
    ax.set_title(r"Yield term structures with long-run limits", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)

    savefig("fig6_yields_with_limits.png")
    y_inf_str   = f"{100*y_inf:.3f}%"   if y_inf   is not None else "no finite limit (trig)"
    y_D_inf_str = f"{100*y_D_inf:.3f}%" if y_D_inf is not None else "no finite limit"
    print(f"Saved: fig6_yields_with_limits.png  "
          f"(y*_∞={y_inf_str}, y_D*_∞={y_D_inf_str})")
