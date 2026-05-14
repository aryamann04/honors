"""Loading-function plots (Figures 1 and 5).

Figure 1 — Wachter blowup & default regularisation
    b*(τ) under δ(K) < 0 (trigonometric branch, finite blowup at τ*)
    b_{D,f}*(τ) under δ(−A_f) > 0 (exponential branch, converges)
    Illustrates that introducing default regularises the long end.

Figure 5 — Loading functions with long-run limits
    b*(τ) and b_{D,f}*(τ) for the baseline calibration
    Horizontal dashed lines at b*_∞ and b_{D,f,∞}*
"""
import numpy as np
import matplotlib.pyplot as plt

from core.parameters import DisasterModelParams
from core.riccati import psi, psi_vec, discriminant, blowup_time
from core.closed_form import (
    _phi_sig2, K_const, _defaultable_Ai,
    b_star_inf, b_Df_inf,
)
from core.utils import savefig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Wachter blowup
# ──────────────────────────────────────────────────────────────────────────────

def plot_wachter_blowup(params: DisasterModelParams,
                        tau_max_def: float = 55.0):
    """Figure 1: illustrate finite blowup for the risk-free bond loading under
    the trigonometric branch (δ(K) < 0) and convergence of b_{D,f}*(τ).

    The baseline calibration itself sits on the trigonometric branch for the
    risk-free loading, so no separate 'illustrative' params are needed.
    The defaultable loading is on the exponential branch and converges.
    """
    phi, sig2 = _phi_sig2(params)
    K         = K_const(params)
    disc_K    = discriminant(K, phi, sig2)
    if disc_K >= 0.0:
        raise ValueError(
            f"δ(K) = {disc_K:.6f} ≥ 0; the trigonometric branch is not active "
            "for these parameters.  The Wachter blowup requires δ(K) < 0."
        )
    tau_star = blowup_time(phi, sig2, K)

    # RF loading: plot up to 98.5% of blowup time
    tau_rf = np.linspace(0.25, 0.985 * tau_star, 600)
    b_rf   = psi_vec(tau_rf, K, phi, sig2)

    # Defaultable loading: exponential branch, plot past the RF blowup horizon
    _, Af, _ = _defaultable_Ai(params)
    tau_def  = np.linspace(0.25, tau_max_def, 600)
    b_Df     = psi_vec(tau_def, -Af, phi, sig2)

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    ax.plot(tau_rf, b_rf, lw=2.2, color="steelblue",
            label=r"$b^*(\tau)$ — risk-free (trigonometric branch)")
    ax.axvline(tau_star, ls=":", color="gray", lw=1.8,
               label=rf"Blowup $\tau^* \approx {tau_star:.1f}$ yr")
    ax.plot(tau_def, b_Df, lw=2.2, color="darkorange", ls="--",
            label=r"$b^*_{D,f}(\tau)$ — defaultable (exponential branch)")

    # show long-run limit of defaultable
    from core.closed_form import b_Df_inf as _bDf_inf
    bDf_lim = _bDf_inf(params)
    if bDf_lim is not None:
        ax.axhline(bDf_lim, ls=":", lw=1.4, color="darkorange",
                   label=rf"$b^*_{{D,f,\infty}} = {bDf_lim:.3f}$")

    ax.set_xlabel(r"Maturity $\tau$ (years)", fontsize=12)
    ax.set_ylabel(r"Loading $b(\tau)$", fontsize=12)
    ax.set_title(
        r"Wachter blowup: $b^*(\tau) \to \infty$ at $\tau^*$;"
        "\ndefault regularises the long end",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)
    ax.set_xlim(left=0.0)

    savefig("fig1_wachter_blowup.png")
    print(f"Saved: fig1_wachter_blowup.png  (τ* ≈ {tau_star:.1f} yr)")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Loading functions with long-run limits
# ──────────────────────────────────────────────────────────────────────────────

def plot_loading_functions(params: DisasterModelParams, tau_grid: np.ndarray):
    """Figure 5: b*(τ) and b_{D,f}*(τ) with horizontal asymptotes."""
    phi, sig2   = _phi_sig2(params)
    K           = K_const(params)
    _, Af, _    = _defaultable_Ai(params)

    b_rf  = psi_vec(tau_grid, K,   phi, sig2)
    b_Df_ = psi_vec(tau_grid, -Af, phi, sig2)

    b_inf  = b_star_inf(params)
    bDf_inf = b_Df_inf(params)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(tau_grid, b_rf,  lw=2.2, color="steelblue",
            label=r"$b^*(\tau)$")
    ax.plot(tau_grid, b_Df_, lw=2.2, color="darkorange", ls="--",
            label=r"$b^*_{D,f}(\tau)$")

    if b_inf is not None:
        ax.axhline(b_inf,  ls=":", lw=1.6, color="steelblue",
                   label=rf"$b^*_{{\infty}} = {b_inf:.4f}$")
    if bDf_inf is not None:
        ax.axhline(bDf_inf, ls=":", lw=1.6, color="darkorange",
                   label=rf"$b^*_{{D,f,\infty}} = {bDf_inf:.4f}$")

    ax.set_xlabel(r"Maturity $\tau$ (years)", fontsize=12)
    ax.set_ylabel(r"Loading", fontsize=12)
    ax.set_title(r"Bond-price loadings $b^*(\tau)$ and $b^*_{D,f}(\tau)$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)

    savefig("fig5_loading_functions.png")
    b_inf_str   = f"{b_inf:.4f}"   if b_inf   is not None else "∞ (trig branch)"
    bDf_inf_str = f"{bDf_inf:.4f}" if bDf_inf is not None else "∞ (trig branch)"
    print(f"Saved: fig5_loading_functions.png  "
          f"(b*_∞={b_inf_str}, b_Df_∞={bDf_inf_str})")
