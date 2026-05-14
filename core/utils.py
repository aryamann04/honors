"""Shared utilities: figure saving, maturity grids."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Output directory for all thesis figures
FINAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final")
os.makedirs(FINAL_DIR, exist_ok=True)


def savefig(name: str, dpi: int = 220):
    """Save the current matplotlib figure to final/ and close it."""
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_DIR, name), dpi=dpi)
    plt.close()


def tau_grid(tau_max: float = 30.0, n: int = 300, tau_min: float = 0.25) -> np.ndarray:
    """Standard maturity grid τ ∈ [tau_min, tau_max] with n points."""
    return np.linspace(tau_min, tau_max, n)


def set_plot_style():
    """Apply consistent matplotlib style for publication figures."""
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "lines.linewidth": 2.2,
    })
