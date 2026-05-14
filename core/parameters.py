from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq


@dataclass
class DisasterModelParams:
    # Preferences / consumption
    beta: float = 0.012       # subjective discount rate β
    gamma: float = 3.0        # risk aversion γ
    mu: float = 0.0252        # drift of log consumption μ
    sigma_c: float = 0.02     # consumption volatility σ
    Z: float = np.log(0.8)    # disaster jump in log consumption (negative, from Du)
    rho_C: float = 0.3        # correlation between foreign and domestic consumption shocks
    phi: float = 2.6          # leverage ratio
    mu_D: float = None        # drift of log dividends (set in __post_init__)

    # Disaster intensity dynamics (foreign/global/home)
    kappa: float = 0.08               # mean reversion of λ
    lam_bar_total: float = 0.0355
    global_share: float = 0.98
    lam_bar_f: float = None           # long-run mean of foreign intensity λ̄^f
    lam_bar_g: float = None           # long-run mean of global intensity λ̄^g
    lam_bar_h: float = None           # long-run mean of home intensity λ̄^h
    sigma_lambda: float = 0.067       # volatility of intensity σ_λ
    v: float = 0.0                    # intensity jump size when disaster hits

    # Default / hazard structure
    R: float = 0.4           # recovery of market value R
    h0_star: float = 0.015   # baseline hazard h*_0
    eta1: float = 2.0        # loading on λ^f in hazard rate η_f
    eta2: float = 1.0        # loading on λ^g in hazard rate η_g

    # SDF loading (computed via compute_b_sdf)
    b_sdf: float = None

    def __post_init__(self):
        if self.mu_D is None:
            object.__setattr__(
                self, "mu_D",
                self.phi * self.mu + 0.5 * self.phi * (self.phi - 1) * self.sigma_c**2,
            )
        if self.lam_bar_g is None:
            object.__setattr__(self, "lam_bar_g", self.global_share * self.lam_bar_total)
        if self.lam_bar_f is None:
            object.__setattr__(self, "lam_bar_f", (1.0 - self.global_share) * self.lam_bar_total)
        if self.lam_bar_h is None:
            object.__setattr__(self, "lam_bar_h", (1.0 - self.global_share) * self.lam_bar_total)

    def compute_b_sdf(self):
        """Solve for the SDF intensity loading b.

        b is the unique stable root of:
            -(β + κ + v)b + (σ_λ²/2)b² + exp(b·v + (1-γ)Z) - 1 = 0
        satisfying  b·σ_λ² - κ < 0  (stability / global bond existence).
        """
        if self.b_sdf is not None:
            return self.b_sdf

        beta = self.beta
        kappa = self.kappa
        v = self.v
        sigma_l = self.sigma_lambda
        gamma = self.gamma
        Z = self.Z
        A = beta + kappa + v

        if v == 0.0:
            disc = (beta + kappa) ** 2 - 2.0 * sigma_l**2 * (np.exp((1.0 - gamma) * Z) - 1.0)
            if disc <= 0.0:
                raise RuntimeError("No real solution for b_sdf (v=0).")
            s = np.sqrt(disc)
            b_plus  = ((beta + kappa) + s) / sigma_l**2
            b_minus = ((beta + kappa) - s) / sigma_l**2
            if b_minus * sigma_l**2 - kappa < 0.0:
                self.b_sdf = float(b_minus)
                return self.b_sdf
            if b_plus * sigma_l**2 - kappa < 0.0:
                self.b_sdf = float(b_plus)
                return self.b_sdf
            raise RuntimeError("Both roots give b·σ_λ² - κ ≥ 0; no stable root.")

        def f(b):
            return -A * b + 0.5 * sigma_l**2 * b**2 + np.exp(b * v + (1.0 - gamma) * Z) - 1.0

        grid = np.linspace(-200.0, 200.0, 40001)
        vals  = f(grid)
        roots = []
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                try:
                    roots.append(brentq(f, grid[i], grid[i + 1]))
                except ValueError:
                    pass

        if not roots:
            raise RuntimeError("No real solution for b_sdf.")

        stability_threshold = (kappa + v) / sigma_l**2
        stable = [r for r in roots if r < stability_threshold]
        if not stable:
            raise RuntimeError("Only explosive roots found for b_sdf.")

        self.b_sdf = float(min(stable, key=lambda x: abs(x)))
        return self.b_sdf
