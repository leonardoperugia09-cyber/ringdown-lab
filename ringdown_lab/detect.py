"""Can you hear the overtone? Three ways to answer, from the literature.

1. Overtone significance  Z = A1 / sigma_A1
   How many standard deviations the overtone amplitude sits away from zero
   (Fisher estimate). Used for GW150914 (Isi et al. 2019, arXiv:1905.00869)
   and GW250114 (LVK 2025, arXiv:2509.08054). We call Z >= 3 "heard".

2. Residual SNR  rho_res = ||h - best 220-only fit|| / sigma
   The part of the signal a fundamental-only model cannot explain. If it is
   below 1, the two models cannot be told apart at all (Lindblom, Owen &
   Brown 2008, arXiv:0809.3844).

3. Rayleigh resolvability  |x220 - x221| > max(sigma_x220, sigma_x221)
   for x = f or tau (Berti, Cardoso & Will 2006, gr-qc/0512160): can we
   tell the two modes apart in frequency, or in damping time?
"""

import numpy as np

from .fit import fisher_errors, fit_one_mode
from .qnm import qnm
from .signal import damped_sinusoid, two_mode


def true_params(mass, spin, ratio=1.0, phase=0.0):
    """Two-mode parameters at the peak (t = 0).

    The fundamental is cos(2 pi f0 t); the overtone is
    ratio * cos(2 pi f1 t + phase), i.e. a1 = ratio*cos(phase),
    b1 = -ratio*sin(phase).
    """
    f0, tau0 = qnm("220", mass, spin)
    f1, tau1 = qnm("221", mass, spin)
    return np.array([1.0, 0.0, f0, tau0,
                     ratio * np.cos(phase), -ratio * np.sin(phase), f1, tau1])


def shift_to_start(params, t0):
    """Rewrite two-mode params so that the model of (t - t0) is unchanged.

    Starting the clock at t0, each mode has decayed by exp(-t0/tau) and its
    phase has turned by 2 pi f t0, so (a, b) change but f and tau do not.
    """
    out = np.array(params, dtype=float)
    for i in (0, 4):
        a, b, f, tau = out[i:i + 4]
        env, phi = np.exp(-t0 / tau), 2 * np.pi * f * t0
        out[i] = env * (a * np.cos(phi) + b * np.sin(phi))
        out[i + 1] = env * (b * np.cos(phi) - a * np.sin(phi))
    return out


def covariance(t, params, sigma=1.0, model=two_mode):
    """Inverse Fisher matrix (the expected covariance of the fit)."""
    errors_sq = fisher_errors(t, params, sigma, model) ** 2   # checks invertibility
    params = np.asarray(params, dtype=float)
    D = np.zeros((len(params), len(t)))
    for i, p in enumerate(params):
        step = 1e-6 * max(abs(p), 1e-3)
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        D[i] = (model(t, *up) - model(t, *down)) / (2 * step)
    cov = np.linalg.inv(D @ D.T / sigma ** 2)
    assert np.allclose(np.diag(cov), errors_sq, rtol=1e-6)
    return cov


def overtone_significance(t, params, sigma=1.0):
    """Z = A1 / sigma_A1 for the overtone amplitude A1 = sqrt(a1^2 + b1^2).

    sigma_A1 comes from the Fisher covariance of (a1, b1):
    sigma_A^2 = (a^2 C_aa + 2 a b C_ab + b^2 C_bb) / A^2
    """
    cov = covariance(t, params, sigma)
    a, b = params[4], params[5]
    A = np.hypot(a, b)
    var = (a * a * cov[4, 4] + 2 * a * b * cov[4, 5] + b * b * cov[5, 5]) / A ** 2
    return A / np.sqrt(var)


def kerr_model(t, a0, b0, a1, b1, mass, spin):
    """Two modes whose f and tau are BOTH set by one mass and spin (as in GR)."""
    f0, tau0 = qnm("220", mass, spin)
    f1, tau1 = qnm("221", mass, spin)
    return two_mode(t, a0, b0, f0, tau0, a1, b1, f1, tau1)


def overtone_significance_kerr(t, params, mass, spin, sigma=1.0):
    """Like overtone_significance, but trusting GR: 6 parameters
    (a0, b0, a1, b1, mass, spin) instead of 8 free ones. Fewer free numbers
    means less blurring between the two modes, so Z is usually higher.
    """
    q = np.array([params[0], params[1], params[4], params[5], mass, spin])
    cov = covariance(t, q, sigma, model=kerr_model)
    a, b = q[2], q[3]
    A = np.hypot(a, b)
    var = (a * a * cov[2, 2] + 2 * a * b * cov[2, 3] + b * b * cov[3, 3]) / A ** 2
    return A / np.sqrt(var)


def resolvability(t, params, sigma=1.0):
    """Rayleigh test for f and tau: (resolved_f, resolved_tau, errors)."""
    err = fisher_errors(t, params, sigma)
    res_f = abs(params[2] - params[6]) > max(err[2], err[6])
    res_tau = abs(params[3] - params[7]) > max(err[3], err[7])
    return bool(res_f), bool(res_tau), err


def overtone_snr(t, params, sigma=1.0):
    """SNR of the overtone on its own: ||h221|| / sigma."""
    return np.sqrt(np.sum(damped_sinusoid(t, *params[4:]) ** 2)) / sigma


def residual_snr(t, params, sigma=1.0):
    """What a 220-only model cannot explain: ||h - best one-mode fit|| / sigma.

    Returns (rho_res, one-mode fit params). The one-mode fit starts at the
    true fundamental, on noiseless data.
    """
    h = two_mode(t, *params)
    fit, ok = fit_one_mode(t, h, params[:4])
    if not ok:
        return np.nan, fit
    resid = h - damped_sinusoid(t, *fit)
    return np.sqrt(np.sum(resid ** 2)) / sigma, fit
