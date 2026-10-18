"""Least-squares fits of ringdown models, and Fisher-matrix error estimates."""

import numpy as np
from scipy.optimize import curve_fit

from .signal import damped_sinusoid, two_mode

# Allowed ranges for the fit. Amplitudes a, b are free (no limits).
F_MIN, F_MAX = 100.0, 500.0        # Hz
TAU_MIN, TAU_MAX = 1e-4, 2e-2      # seconds

# Lower/upper bounds for one mode's (a, b, f, tau).
_LO = [-np.inf, -np.inf, F_MIN, TAU_MIN]
_HI = [np.inf, np.inf, F_MAX, TAU_MAX]


def _clip_start(p0, lo, hi):
    """curve_fit needs the starting guess strictly inside the bounds."""
    lo, hi = np.array(lo), np.array(hi)
    eps = 1e-9 * np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    return np.clip(np.asarray(p0, dtype=float), lo + eps, hi - eps)


def _fit(model, t, data, p0, lo, hi):
    """Run curve_fit; return (params, ok). ok is False if the fit failed."""
    try:
        params, _ = curve_fit(model, t, data, p0=_clip_start(p0, lo, hi),
                              bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError):
        return np.full(len(p0), np.nan), False
    return params, bool(np.all(np.isfinite(params)))


def fit_one_mode(t, data, p0):
    """Fit one damped sinusoid. p0 = (a, b, f, tau)."""
    return _fit(damped_sinusoid, t, data, p0, _LO, _HI)


def fit_two_mode(t, data, p0):
    """Fit two damped sinusoids. p0 = (a0, b0, f0, tau0, a1, b1, f1, tau1).

    Both modes have the same bounds, so the fit could swap them. We fix the
    labels afterwards: the mode with the LONGER damping time is the
    fundamental (220), the shorter one is the overtone (221).
    """
    params, ok = _fit(two_mode, t, data, p0, _LO + _LO, _HI + _HI)
    if ok and params[3] < params[7]:
        params = np.concatenate([params[4:], params[:4]])
    return params, ok


def fisher_errors(t, params, sigma=1.0, model=two_mode, rel_step=1e-6):
    """Expected 1-sigma errors on each parameter from the Fisher matrix.

    D[i] = d(model)/d(param_i), by central differences:
           (model(p + step) - model(p - step)) / (2 * step)
    F    = D . D^T / sigma^2
    errors = sqrt(diag(F^-1))
    """
    params = np.asarray(params, dtype=float)
    D = np.zeros((len(params), len(t)))
    for i, p in enumerate(params):
        step = rel_step * max(abs(p), 1e-3)   # small step, never exactly 0
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        D[i] = (model(t, *up) - model(t, *down)) / (2 * step)
    F = D @ D.T / sigma ** 2
    return np.sqrt(np.diag(np.linalg.inv(F)))
