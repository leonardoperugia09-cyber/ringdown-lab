"""Quasi-normal mode (QNM) frequencies and damping times.

Uses the fitting formulas of Berti, Cardoso & Will (2006):
    F = f1 + f2 * (1 - chi)**f3      (dimensionless frequency)
    Q = q1 + q2 * (1 - chi)**q3      (quality factor)
    f   = F / (2 * pi * M)           (frequency in Hz, M in seconds)
    tau = Q / (pi * f)               (damping time in seconds)
"""

import numpy as np

# One solar mass expressed in seconds (G * M_sun / c^3).
MSUN_SECONDS = 4.925491e-6

# Fit coefficients (f1, f2, f3, q1, q2, q3) for each mode.
COEFFS = {
    "220": (1.5251, -1.1568, 0.1292, 0.7000, 1.4187, -0.4990),
    "221": (1.3673, -1.0260, 0.1628, 0.1000, 0.5436, -0.4731),
}

# The remnant black hole we study (GW250114-like).
REMNANT_MASS = 62.7   # solar masses
REMNANT_SPIN = 0.68   # dimensionless spin chi


def qnm(mode, mass, spin):
    """Return (f, tau) for a mode ("220" or "221").

    mass is in solar masses, spin is the dimensionless chi (0 <= chi < 1).
    f is in Hz, tau is in seconds.
    """
    f1, f2, f3, q1, q2, q3 = COEFFS[mode]
    F = f1 + f2 * (1 - spin) ** f3
    Q = q1 + q2 * (1 - spin) ** q3
    M = mass * MSUN_SECONDS          # mass converted to seconds
    f = F / (2 * np.pi * M)
    tau = Q / (np.pi * f)
    return f, tau
