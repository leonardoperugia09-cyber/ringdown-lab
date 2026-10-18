"""Ringdown signal models: damped sinusoids."""

import numpy as np

FS = 4096        # sampling rate in Hz (samples per second)
DURATION = 0.1   # length of the data in seconds


def time_axis(fs=FS, duration=DURATION):
    """Times of each sample, starting at t = 0 (the peak)."""
    n = int(fs * duration)          # 4096 * 0.1 -> 409 samples
    return np.arange(n) / fs


def damped_sinusoid(t, a, b, f, tau):
    """One ringing mode: exp(-t/tau) * (a*cos(2 pi f t) + b*sin(2 pi f t)).

    a and b together set the amplitude and phase of the mode.
    """
    phase = 2 * np.pi * f * t
    return np.exp(-t / tau) * (a * np.cos(phase) + b * np.sin(phase))


def two_mode(t, a0, b0, f0, tau0, a1, b1, f1, tau1):
    """Fundamental (220) plus first overtone (221)."""
    return damped_sinusoid(t, a0, b0, f0, tau0) + damped_sinusoid(t, a1, b1, f1, tau1)
