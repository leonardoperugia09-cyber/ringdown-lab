"""Checks for the detection measures in ringdown_lab/detect.py."""

import numpy as np
import pytest

from ringdown_lab.detect import (overtone_significance, overtone_significance_kerr,
                                 overtone_snr, residual_snr,
                                 resolvability, shift_to_start, true_params)
from ringdown_lab.noise import snr
from ringdown_lab.signal import time_axis, two_mode

T = time_axis()
P = true_params(62.7, 0.68)


def scaled(params, target, t=T):
    """Copy of params with amplitudes scaled so the signal on t has SNR target."""
    out = np.array(params, dtype=float)
    out[[0, 1, 4, 5]] *= target / snr(two_mode(t, *params))
    return out


def test_true_params_peak_value():
    # ratio 2, phase 0: h(0) = 1 + 2 = 3. Phase pi: 1 - 2 = -1.
    assert two_mode(np.array([0.0]), *true_params(62.7, 0.68, 2.0, 0.0))[0] == pytest.approx(3)
    assert two_mode(np.array([0.0]), *true_params(62.7, 0.68, 2.0, np.pi))[0] == pytest.approx(-1)


@pytest.mark.parametrize("t0", [0.0, 1e-3, 3e-3])
def test_shift_to_start_keeps_the_waveform(t0):
    keep = T >= t0 - 1e-12
    tt = T[keep] - T[keep][0]
    shifted = shift_to_start(P, T[keep][0])
    assert np.allclose(two_mode(tt, *shifted), two_mode(T, *P)[keep])


def test_significance_grows_linearly_with_snr():
    z50 = overtone_significance(T, scaled(P, 50))
    z100 = overtone_significance(T, scaled(P, 100))
    assert z100 / z50 == pytest.approx(2.0, rel=1e-4)


def test_residual_snr_zero_without_overtone():
    p = scaled(true_params(62.7, 0.68, ratio=0.0), 100)
    p[4] = 0.0
    rho, fit = residual_snr(T, p)
    assert rho < 1e-6
    assert fit[3] == pytest.approx(p[3], rel=1e-8)


def test_later_start_makes_overtone_quieter():
    """The overtone decays ~3x faster, so starting later must lower every measure."""
    values = []
    for t0 in [0.0, 2e-3, 4e-3]:
        keep = T >= t0 - 1e-12
        tt = T[keep] - T[keep][0]
        p = shift_to_start(P, T[keep][0])
        p[[0, 1, 4, 5]] *= 100 / snr(two_mode(T, *P))       # SNR 100 from the peak
        values.append((overtone_significance(tt, p), residual_snr(tt, p)[0],
                       overtone_snr(tt, p)))
    for earlier, later in zip(values, values[1:]):
        assert all(e > l for e, l in zip(earlier, later))


def test_kerr_linked_fit_is_never_less_certain():
    """Fixing f and tau by GR removes freedom, so Z can only go up (or stay)."""
    for ratio, phase in [(1, 0), (4, np.pi), (0.5, 1.0)]:
        p = scaled(true_params(62.7, 0.68, ratio, phase), 50)
        assert (overtone_significance_kerr(T, p, 62.7, 0.68)
                >= overtone_significance(T, p) * (1 - 1e-6))


def test_counter_phase_reproduces_gw250114_significance():
    """Ratio 4 at peak, phase 180 deg, fit from 2 ms with SNR 26 left at the
    start (GW250114 two-mode ringdown SNR, arXiv:2509.08054) gives Z ~ 4,
    close to the 4.1 sigma reported; phase 0 gives only ~1."""
    keep = T >= 2e-3 - 1e-12
    tt = T[keep] - T[keep][0]
    z = {}
    for phase in (0.0, np.pi):
        p = shift_to_start(true_params(68.1, 0.68, 4.0, phase), T[keep][0])
        z[phase] = overtone_significance(tt, scaled(p, 26, tt))
    assert 3.5 < z[np.pi] < 4.7
    assert z[0.0] < 2


def test_resolvability_at_high_snr():
    res_f, res_tau, err = resolvability(T, scaled(P, 5000))
    assert res_f and res_tau
    res_f, res_tau, err = resolvability(T, scaled(P, 10))
    assert not res_f and not res_tau
