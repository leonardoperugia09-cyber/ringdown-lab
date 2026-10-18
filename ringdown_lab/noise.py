"""White Gaussian noise and signal-to-noise ratio (SNR)."""

import numpy as np


def white_noise(n, sigma, rng):
    """n independent Gaussian samples with mean 0 and standard deviation sigma.

    rng is a numpy random Generator, e.g. np.random.default_rng(42),
    so results can be reproduced exactly.
    """
    return rng.normal(0.0, sigma, n)


def snr(h, sigma=1.0):
    """Optimal SNR of signal h in white noise: sqrt(sum(h^2)) / sigma."""
    return np.sqrt(np.sum(h ** 2)) / sigma


def scale_to_snr(h, target, sigma=1.0):
    """Return h multiplied by a constant so that snr(h, sigma) == target."""
    return h * (target / snr(h, sigma))
