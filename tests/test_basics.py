"""Basic checks for Steps 1-4. Run with:  python -m pytest"""

import numpy as np
import pytest

from ringdown_lab.qnm import qnm, REMNANT_MASS, REMNANT_SPIN
from ringdown_lab.signal import time_axis, damped_sinusoid, two_mode, FS, DURATION
from ringdown_lab.noise import white_noise, snr, scale_to_snr
from ringdown_lab.fit import fit_one_mode, fit_two_mode, fisher_errors

F0, TAU0 = qnm("220", REMNANT_MASS, REMNANT_SPIN)
F1, TAU1 = qnm("221", REMNANT_MASS, REMNANT_SPIN)
TRUE = np.array([1.0, 0.0, F0, TAU0, 1.0, 0.0, F1, TAU1])


# ---- Step 1: QNM formulas --------------------------------------------------

@pytest.mark.parametrize("mode, f_expected, tau_ms_expected", [
    ("220", 271.4, 3.76),
    ("221", 265.4, 1.24),
])
def test_qnm_values(mode, f_expected, tau_ms_expected):
    f, tau = qnm(mode, REMNANT_MASS, REMNANT_SPIN)
    assert abs(f - f_expected) < 0.1
    assert abs(tau * 1e3 - tau_ms_expected) < 0.1


def test_overtone_dies_faster():
    assert TAU1 < TAU0


# ---- Step 2: signal --------------------------------------------------------

def test_time_axis():
    t = time_axis()
    assert t[0] == 0.0
    assert len(t) == int(FS * DURATION)
    assert np.allclose(np.diff(t), 1 / FS)


def test_two_mode_starts_at_two():
    t = time_axis()
    assert two_mode(t, *TRUE)[0] == pytest.approx(2.0)


def test_damped_sinusoid_decays_by_e():
    # After one damping time the envelope has fallen by a factor e.
    # Pick f so that t = tau lands on a whole number of cycles (cos = 1).
    tau = 0.01
    f = 3 / tau
    h = damped_sinusoid(np.array([0.0, tau]), 1.0, 0.0, f, tau)
    assert h[1] / h[0] == pytest.approx(np.exp(-1))


# ---- Step 3: noise ---------------------------------------------------------

def test_noise_std_is_one():
    n = white_noise(100_000, 1.0, np.random.default_rng(0))
    assert abs(n.std() - 1.0) < 0.01
    assert abs(n.mean()) < 0.01


@pytest.mark.parametrize("target", [8, 50, 300])
def test_scale_to_snr(target):
    h = two_mode(time_axis(), *TRUE)
    assert snr(scale_to_snr(h, target)) == pytest.approx(target, rel=1e-12)


# ---- Step 4: fitting -------------------------------------------------------

def test_noiseless_two_mode_fit_returns_truth():
    t = time_axis()
    start = TRUE * np.array([1, 1, 1.08, 0.92, 1, 1, 0.93, 1.09])
    params, ok = fit_two_mode(t, two_mode(t, *TRUE), start)
    assert ok
    assert np.allclose(params, TRUE, rtol=1e-6, atol=1e-8)


def test_noiseless_one_mode_fit_returns_truth():
    t = time_axis()
    true = [1.0, 0.0, F0, TAU0]
    params, ok = fit_one_mode(t, damped_sinusoid(t, *true), [1, 0, F0 * 1.05, TAU0 * 0.95])
    assert ok
    assert np.allclose(params, true, rtol=1e-6, atol=1e-8)


def test_fisher_errors_scale_as_one_over_snr():
    t = time_axis()
    h = two_mode(t, *TRUE)
    amp = np.array([1, 1, 0, 0, 1, 1, 0, 0])

    def params_at(target):
        s = target / snr(h)
        return TRUE * np.where(amp == 1, s, 1.0)

    e50 = fisher_errors(t, params_at(50))
    e100 = fisher_errors(t, params_at(100))
    ratio = e50 / e100
    # Doubling the SNR (bigger signal, same noise) halves the f and tau errors.
    assert np.allclose(ratio[amp == 0], 2.0, rtol=1e-4)
    # Amplitude errors are set by the noise level, which did not change,
    # so they stay the same (the RELATIVE amplitude error still halves).
    assert np.allclose(ratio[amp == 1], 1.0, rtol=1e-4)
