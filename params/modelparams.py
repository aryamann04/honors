from dataclasses import dataclass, replace
import numpy as np

@dataclass
class DisasterModelParams:
    # Preferences / consumption
    beta: float = 0.02      # subjective discount rate β
    gamma: float = 4.0      # risk aversion γ
    mu: float = 0.0218      # drift of log consumption μ
    sigma_c: float = 0.009  # consumption volatility σ
    Z: float = -0.31        # disaster jump in log consumption (negative)
    rho_C: float = 0.3      # correlation between foreign and domestic consumption shocks

    # Disaster intensity dynamics (foreign/global)
    kappa: float = 0.146       # mean reversion of λ
    lam_bar_f: float = 0.017 * 0.12 * 7/8 # long-run mean of foreign intensity λ̄^f
    lam_bar_g: float = 0.017 * 0.94 # long-run mean of global intensity λ̄^g
    lam_bar_h: float = 0.017 * 0.12 * 1/2 # long-run mean of hazard intensity λ̄^h
    sigma_lambda: float = 0.09 # volatility of intensity σ_λ
    v: float = 0.0057          # jump size in intensity when a disaster hits

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

        def f(b):
            return (-(beta + kappa + v)*b
                    + 0.5 * sigma_l**2 * b**2
                    + np.exp(b*v + (1-gamma)*Z)
                    - 1)

        def fprime(b):
            return (-(beta + kappa + v)
                    + sigma_l**2 * b
                    + v * np.exp(b*v + (1-gamma)*Z))

        b = 0.01 
        for _ in range(50):
            try:
                fb = f(b)
                fpb = fprime(b)
                if not np.isfinite(fb) or not np.isfinite(fpb) or abs(fpb) < 1e-10:
                    b = 0.5 * b  
                    continue
                step = fb / fpb
                b_new = b - step
                if abs(b_new - b) > 1.0:
                    b_new = b - np.sign(step) * 1.0
                b = b_new
            except FloatingPointError:
                b *= 0.5

        self.b_sdf = float(b)
        return self.b_sdf