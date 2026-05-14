"""CDS pricing plots (Figures 8, 9, 10).

Figure 8  — CDS spread vs bond credit spread (term structure)
    s_CDS(τ) and s*(τ) on the same axes for the baseline state

Figure 9  — CDS spread at fixed tenor τ₀ = 5 vs λ^f
    s_CDS(5) as a function of λ^f — expected shape: increasing and convex

Figure 10 — G(t, t+u) and K(t, t+u) as functions of u
    For the baseline state, u ∈ [0, 20]
    Survival-weighted discount factor G and hazard-weighted object K
"""
import numpy as np
import matplotlib.pyplot as plt

from core.parameters import DisasterModelParams
from models.defaultable import credit_spread_curve
from models.cds import G_value, K_value, fair_cds_spread, cds_spread_curve
from core.utils import savefig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8 — CDS vs bond spread term structure
# ──────────────────────────────────────────────────────────────────────────────

def plot_cds_vs_bond_spread(params: DisasterModelParams,
                             tau_grid: np.ndarray,
                             lam_f: float,
                             lam_g: float,
                             payment_interval: float = 0.25):
    """Figure 8: bond credit spread s*(τ) and fair CDS spread s_CDS(τ)."""
    bond_spread = credit_spread_curve(params, tau_grid, lam_f, lam_g)
    cds_spread  = cds_spread_curve(params, tau_grid, lam_f, lam_g, payment_interval)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    ax.plot(tau_grid, 1e4 * bond_spread, lw=2.2, color="steelblue",
            label=r"Bond credit spread $s^*(\tau)$")
    ax.plot(tau_grid, 1e4 * cds_spread,  lw=2.2, color="darkorange", ls="--",
            label=r"Fair CDS spread $s_{\mathrm{CDS}}(\tau)$")

    ax.set_xlabel(r"Maturity $\tau$ (years)", fontsize=12)
    ax.set_ylabel(r"Spread (bp)", fontsize=12)
    ax.set_title(r"Bond spread vs fair CDS spread", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)

    savefig("fig8_cds_vs_bond_spread.png")
    print(f"Saved: fig8_cds_vs_bond_spread.png  "
          f"(5Y bond={1e4*credit_spread_curve(params,np.array([5.0]),lam_f,lam_g)[0]:.1f} bp, "
          f"5Y CDS={1e4*fair_cds_spread(params,5.0,lam_f,lam_g):.1f} bp)")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9 — CDS spread at fixed tenor vs λ^f
# ──────────────────────────────────────────────────────────────────────────────

def plot_cds_vs_lambda_f(params: DisasterModelParams,
                          lam_f_grid: np.ndarray,
                          lam_g: float,
                          tau0: float = 5.0,
                          payment_interval: float = 0.25):
    """Figure 9: s_CDS(τ₀) and s*(τ₀) as functions of λ^f."""
    s_cds  = np.array([fair_cds_spread(params, tau0, lf, lam_g, payment_interval)
                        for lf in lam_f_grid])
    s_bond = np.array([credit_spread_curve(params, np.array([tau0]), lf, lam_g)[0]
                        for lf in lam_f_grid])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(lam_f_grid, 1e4 * s_cds,  lw=2.2, color="steelblue",
            label=rf"$s_{{\mathrm{{CDS}}}}(\tau_0)$, $\tau_0 = {tau0}$ yr")
    ax.plot(lam_f_grid, 1e4 * s_bond, lw=2.2, color="darkorange", ls="--",
            label=rf"$s^*(\tau_0)$, $\tau_0 = {tau0}$ yr")

    ax.set_xlabel(r"Foreign disaster intensity $\lambda^f_t$", fontsize=12)
    ax.set_ylabel(r"Spread (bp)", fontsize=12)
    ax.set_title(rf"CDS and bond spreads at $\tau_0 = {tau0}$ yr vs $\lambda^f$",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.28)

    savefig("fig9_cds_vs_lambda_f.png")
    print(f"Saved: fig9_cds_vs_lambda_f.png  "
          f"(CDS range: {1e4*s_cds.min():.1f}–{1e4*s_cds.max():.1f} bp)")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10 — G and K as functions of u
# ──────────────────────────────────────────────────────────────────────────────

def plot_G_and_K(params: DisasterModelParams,
                  u_grid: np.ndarray,
                  lam_f: float,
                  lam_g: float):
    """Figure 10: G(t, t+u) and K(t, t+u) vs u for the baseline state."""
    G_vals = np.array([G_value(params, float(u), lam_f, lam_g) for u in u_grid])
    K_vals = np.array([K_value(params, float(u), lam_f, lam_g) for u in u_grid])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(u_grid, G_vals, lw=2.2, color="steelblue")
    axes[0].set_xlabel(r"Horizon $u$ (years)", fontsize=11)
    axes[0].set_ylabel(r"$G(t,\, t+u)$", fontsize=11)
    axes[0].set_title(r"Discounted survival probability $G(t,\, t+u)$", fontsize=11)
    axes[0].grid(alpha=0.28)

    axes[1].plot(u_grid, K_vals, lw=2.2, color="darkorange")
    axes[1].set_xlabel(r"Horizon $u$ (years)", fontsize=11)
    axes[1].set_ylabel(r"$K(t,\, t+u)$", fontsize=11)
    axes[1].set_title(r"Hazard-weighted object $K(t,\, t+u)$", fontsize=11)
    axes[1].grid(alpha=0.28)

    fig.suptitle(
        rf"$G$ and $K$ objects  "
        rf"($\lambda^f = {lam_f:.4f}$, $\lambda^g = {lam_g:.4f}$)",
        fontsize=12,
    )
    savefig("fig10_G_and_K.png")
    print(f"Saved: fig10_G_and_K.png  "
          f"(G(20)={G_vals[-1]:.4f}, K(20)={K_vals[-1]:.6f})")
