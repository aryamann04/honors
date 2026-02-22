from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import brentq


@dataclass
class DisasterModelParams:
    # Preferences / consumption
    beta: float = 0.02      # subjective discount rate β
    gamma: float = 4.0      # risk aversion γ
    mu: float = 0.0218      # drift of log consumption μ
    sigma_c: float = 0.009  # consumption volatility σ
    Z: float = -0.20        # disaster jump in log consumption (negative)
    rho_C: float = 0.3      # correlation between foreign and domestic consumption shocks
    phi: float = 2.5        # leverage ratio
    mu_D: float = phi * mu + 0.5 * phi * (phi - 1) * (sigma_c ** 2)   # drift of log dividends

    # Disaster intensity dynamics (foreign/global)
    kappa: float = 0.146            # mean reversion of λ
    lam_bar_f: float = 0.017 * 0.06 # long-run mean of foreign intensity λ̄^f
    lam_bar_g: float = 0.017 * 0.94 # long-run mean of global intensity λ̄^g
    lam_bar_h: float = 0.017 * 0.06 # long-run mean of hazard intensity λ̄^h
    sigma_lambda: float = 0.09      # volatility of intensity σ_λ
    v: float = 0            # jump size in intensity when a disaster hits

    # Default/hazard structure
    R: float = 0.4          # recovery of market value (RMV) R
    h0_star: float = 0.02   # baseline hazard h*_0
    eta1: float = 2         # loading on λ_f in hazard
    eta2: float = 1         # loading on λ_g in hazard

    # Constant b in the SDF 
    b_sdf: float = None    

    def compute_b_sdf(self):
        if self.b_sdf is not None:
            return self.b_sdf

        beta = self.beta
        kappa = self.kappa
        v = self.v
        sigma_l = self.sigma_lambda
        gamma = self.gamma
        Z = self.Z

        A = beta + kappa + v

        if v == 0:
            self.b_sdf = ( (beta + kappa) + np.sqrt((beta + kappa) ** 2 - 2 * sigma_l ** 2 * (np.exp((1 - gamma) * Z) - 1)) ) / (sigma_l ** 2)
            return self.b_sdf

        def f(b):
            return (
                -(A) * b
                + 0.5 * sigma_l**2 * b**2
                + np.exp(b * v + (1.0 - gamma) * Z)
                - 1.0
            )

        grid = np.linspace(-200.0, 200.0, 40001)
        vals = f(grid)

        roots = []

        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i+1]):
                if vals[i] * vals[i+1] < 0:
                    try:
                        root = brentq(f, grid[i], grid[i+1])
                        roots.append(root)
                    except ValueError:
                        pass

        if len(roots) == 0:
            raise RuntimeError("No real solution for b_sdf.")

        stable_roots = []
        stability_threshold = (kappa + v) / (sigma_l**2)

        for r in roots:
            if r < stability_threshold:
                stable_roots.append(r)

        if len(stable_roots) == 0:
            raise RuntimeError("Only explosive roots found for b_sdf.")

        b_star = min(stable_roots, key=lambda x: abs(x))
        self.b_sdf = float(b_star)

        return self.b_sdf