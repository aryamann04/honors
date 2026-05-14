"""Quanto (domestic-currency defaultable) bond pricing — QUARANTINED.

This module preserves the quanto functionality from the old reducedmodel/plots.py
for reference.  It is NOT used in any thesis figure in the current draft.

The quanto bond B̃_D*(t, T) is priced in domestic currency.  Its loading on λ^f
is NOT tilted under Q* (foreign risk-neutral measure) because the FX-default
covariance is priced separately.  This leads to a distinct ODE for b̃_f.

Do not import this module into core/ or models/ without explicit thesis need.
"""
import numpy as np
from core.parameters import DisasterModelParams
from core.riccati import psi, phi_integral


def quanto_coeffs(params: DisasterModelParams, tau: float):
    """(a_q(τ), b_{q,h}(τ), b_{q,g}(τ), b_{q,f}(τ)) for the quanto bond (v = 0).

    Home and global loadings use the standard φ-based Riccati.
    Foreign loading uses an untilted Riccati with φ̃ = −κ (no SDF tilt on λ^f).

    Constants:
        A_h = exp(−γZ)(exp Z − 1)         (home: no default loading)
        A_g = A_h + (1−R) η_g
        A_f = (1−R) η_f                   (foreign: untilted, no A_h term)
    """
    params.compute_b_sdf()
    b    = float(params.b_sdf)
    sig2 = params.sigma_lambda ** 2
    phi  = b * sig2 - params.kappa

    K  = np.exp(-params.gamma * params.Z) * (1.0 - np.exp(params.Z))
    Ah = -K                                   # = exp(−γZ)(exp Z − 1)
    Ag = Ah + (1.0 - params.R) * params.eta2
    Af = (1.0 - params.R) * params.eta1       # untilted loading

    A0 = (params.beta + params.mu - params.gamma * params.sigma_c**2
          + (1.0 - params.R) * params.h0_star)

    # h and g loadings: standard tilted Riccati
    bh  = psi(tau, -Ah, phi, sig2)
    bg  = psi(tau, -Ag, phi, sig2)
    Phh = phi_integral(tau, -Ah, phi, sig2)
    Phg = phi_integral(tau, -Ag, phi, sig2)

    # f loading: untilted Riccati with φ̃ = −κ
    phi_f = -params.kappa
    delta_f = np.sqrt(params.kappa**2 + 2.0 * sig2 * Af)
    bf      = psi(tau, Af, phi_f, sig2)      # x = Af > 0, phi_f = -kappa
    Phf     = phi_integral(tau, Af, phi_f, sig2)

    a_q = (-A0 * tau
           + params.kappa * params.lam_bar_h / sig2 * sig2 * Phh
           + params.kappa * params.lam_bar_g / sig2 * sig2 * Phg
           + params.kappa * params.lam_bar_f / sig2 * sig2 * Phf)

    return a_q, bh, bg, bf
