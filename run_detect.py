"""Experiments C and D: can you HEAR the overtone, and when does it matter?

    python run_detect.py           # full run, N = 200
    python run_detect.py --quick   # N = 30
    python run_detect.py --figures-only

Experiment C (simulated detection test). For each SNR, fit the noisy data with
    one mode and with two modes. The improvement dchi2 = chi2_one - chi2_two is
    large when the overtone is really there. To know what "large" means we also
    fit data that contain ONLY the fundamental: the 99th percentile of dchi2 on
    those is the threshold (1% false alarms). Detected = dchi2 > threshold.
    We compare with the Fisher prediction "overtone significance Z >= 3".

Experiment D (amplitude ratio and phase). On a grid of overtone amplitude
    ratio and phase, compute (no noise needed):
      - the crossover SNR where the tau220 bias from a 220-only fit equals the
        noise error (noise error from the one-mode Fisher matrix), and
      - the SNR at which the overtone reaches Z = 3.
    Between those two SNRs lies the "blind spot": the overtone already spoils
    the 220 measurement, but you cannot yet hear it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.ticker import FuncFormatter

from ringdown_lab.detect import (overtone_significance, residual_snr,
                                 shift_to_start, true_params)
from ringdown_lab.fit import fisher_errors, fit_one_mode, fit_two_mode
from ringdown_lab.noise import snr, white_noise
from ringdown_lab.qnm import REMNANT_MASS, REMNANT_SPIN
from ringdown_lab.signal import damped_sinusoid, time_axis, two_mode

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))   # find run_build.py next to us
import run_build                  # shared plot style, colours, helpers
from run_build import BLUE, ORANGE, AQUA, INK_2, PLAIN, jitter

ROOT = Path(__file__).resolve().parent
SNRS = run_build.SNRS
C_START_MS = [0, 2]
D_START_MS = [0, 2]
D_RATIOS = [0.25, 0.5, 1.0, 2.0, 4.0]
D_PHASES_DEG = list(range(0, 360, 30))
B_START_MS = run_build.START_TIMES_MS
FALSE_ALARM = 0.01
Z_HEARD = 3.0
SEED = 2027
ORANGE_DARK = "#a03f15"
ORANGE_SEQ = ListedColormap(["#fde3d6", "#f9c3a7", "#f4a078", "#eb6834",
                             "#c9521f", "#a03f15", "#76300f"])
BLUE_SEQ = ListedColormap(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                           "#256abf", "#184f95", "#0d366b"])


def window(t, t0_ms):
    """Samples from t0 on, with the clock restarted at t0."""
    keep = t >= t0_ms * 1e-3 - 1e-12
    return keep, t[keep] - t[keep][0], t[keep][0]


def chi2(data, model):
    return float(np.sum((data - model) ** 2))


def delta_chi2(tt, data, start2, rng):
    """chi2(one mode) - chi2(two modes), both fits started near the truth.

    The two-mode model contains the one-mode model (overtone amplitude 0),
    so at its best fit chi2_two <= chi2_one. If the optimiser gets stuck
    in a worse spot we clip at 0 instead of reporting a negative number.
    """
    s1 = start2[:4].copy()
    s1[2:] = jitter(rng, s1[2:])
    p1, ok1 = fit_one_mode(tt, data, s1)
    s2 = start2.copy()
    s2[[2, 3, 6, 7]] = jitter(rng, s2[[2, 3, 6, 7]])
    p2, ok2 = fit_two_mode(tt, data, s2)
    if not (ok1 and ok2):
        return np.nan
    return max(chi2(data, damped_sinusoid(tt, *p1)) - chi2(data, two_mode(tt, *p2)), 0.0)


def experiment_c(t, P, n_real, rng):
    h_full = two_mode(t, *P)
    sig, null = {}, {}
    for t0_ms in C_START_MS:
        keep, tt, t0 = window(t, t0_ms)
        sig[t0_ms], null[t0_ms] = [], []
        for target in SNRS:
            p = shift_to_start(P, t0)
            p[[0, 1, 4, 5]] *= target / snr(h_full)
            h_sig = two_mode(tt, *p)
            h_null = damped_sinusoid(tt, *p[:4])
            d_sig = [delta_chi2(tt, h_sig + white_noise(len(tt), 1, rng), p, rng)
                     for _ in range(n_real)]
            d_null = [delta_chi2(tt, h_null + white_noise(len(tt), 1, rng), p, rng)
                      for _ in range(n_real)]
            sig[t0_ms].append(np.array(d_sig))
            null[t0_ms].append(np.array(d_null))
    pooled = np.concatenate([x for v in null.values() for x in v])
    threshold = float(np.nanpercentile(pooled, 100 * (1 - FALSE_ALARM)))
    rows = []
    for t0_ms in C_START_MS:
        keep, tt, t0 = window(t, t0_ms)
        for target, d_sig, d_null in zip(SNRS, sig[t0_ms], null[t0_ms]):
            p = shift_to_start(P, t0)
            p[[0, 1, 4, 5]] *= target / snr(h_full)
            rows.append({
                "start_ms": t0_ms, "snr": target,
                "percent_detected": float(100 * np.nanmean(d_sig > threshold)),
                "percent_false_alarm": float(100 * np.nanmean(d_null > threshold)),
                "median_dchi2": float(np.nanmedian(d_sig)),
                "fisher_Z": float(overtone_significance(tt, p)),
                "residual_snr": float(residual_snr(tt, p)[0]),
                "n_failed": int(np.sum(np.isnan(d_sig)) + np.sum(np.isnan(d_null))),
            })
            r = rows[-1]
            print(f"  C  start {t0_ms} ms SNR {target:>3}: detected {r['percent_detected']:5.1f}%"
                  f"  (false alarm {r['percent_false_alarm']:4.1f}%)  Fisher Z {r['fisher_Z']:6.2f}"
                  f"  rho_res {r['residual_snr']:6.2f}")
    print(f"  C  dchi2 threshold for {100 * FALSE_ALARM:.0f}% false alarms: {threshold:.2f}")
    return rows, threshold


def per_unit_snr(t, P, t0_ms):
    """Bias, noise k and Z for a full-signal SNR of 1 (everything scales from here)."""
    keep, tt, t0 = window(t, t0_ms)
    p = shift_to_start(P, t0)
    p[[0, 1, 4, 5]] /= snr(two_mode(t, *P))
    rho_res, fit = residual_snr(tt, p)
    bias = fit[3] - P[3]
    k = fisher_errors(tt, fit, model=damped_sinusoid)[3]   # tau220 error at SNR 1
    z1 = overtone_significance(tt, p)
    return {"bias_percent": 100 * bias / P[3], "k_ms": 1e3 * k,
            "crossover_snr": k / abs(bias), "snr_heard": Z_HEARD / z1,
            "snr_indistinguishable": 1 / rho_res}


def experiment_d(t):
    grid = {}
    for t0_ms in D_START_MS:
        cells = []
        for ratio in D_RATIOS:
            for ph in D_PHASES_DEG:
                P = true_params(REMNANT_MASS, REMNANT_SPIN, ratio, np.radians(ph))
                cells.append({"ratio": ratio, "phase_deg": ph, **per_unit_snr(t, P, t0_ms)})
        grid[t0_ms] = cells
    # Blind spot vs start time for the default setup (ratio 1, phase 0).
    P = true_params(REMNANT_MASS, REMNANT_SPIN)
    blind = [{"start_ms": s, **per_unit_snr(t, P, s)} for s in B_START_MS]
    for r in blind:
        print(f"  D  default, start {r['start_ms']} ms: bias wins above SNR {r['crossover_snr']:8.1f}"
              f" (Fisher k {r['k_ms']:.3f} ms), heard (Z=3) above SNR {r['snr_heard']:9.1f}")
    for t0_ms in D_START_MS:
        cr = [c["crossover_snr"] for c in grid[t0_ms]]
        hd = [c["snr_heard"] for c in grid[t0_ms]]
        print(f"  D  grid start {t0_ms} ms: crossover SNR {min(cr):.1f}-{max(cr):.1f},"
              f" heard SNR {min(hd):.1f}-{max(hd):.1f}")
    return grid, blind


def make_figures(c_rows, threshold, grid, blind):
    figdir = ROOT / "figures"

    # 5) Detection rate vs SNR, with the Fisher Z=3 SNR marked.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for t0_ms, color, ls in zip(C_START_MS, [ORANGE, ORANGE_DARK], ["-", "--"]):
        rows = [r for r in c_rows if r["start_ms"] == t0_ms]
        x = [r["snr"] for r in rows]
        ax.semilogx(x, [r["percent_detected"] for r in rows], color=color, ls=ls, marker="o",
                    label=f"overtone, fit starts {t0_ms} ms after peak")
        heard = [b for b in blind if b["start_ms"] == t0_ms][0]["snr_heard"]
        if heard <= 300:
            ax.axvline(heard, color=color, ls="--", lw=1.2)
            ax.annotate(f"Fisher Z = 3\nat SNR {heard:.0f}", (heard, 55), xytext=(6, 0),
                        textcoords="offset points", color=INK_2, fontsize=9)
    ax.axhline(100 * FALSE_ALARM, color=INK_2, lw=1, ls=":")
    ax.set_ylim(-2, 102)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("Signal-to-noise ratio (SNR, whole signal from peak)")
    ax.set_ylabel("221 overtone detected (% of noisy fits)")
    ax.set_title(f"221 overtone: Δχ² test, threshold {threshold:.1f} (1% false alarms, dotted)",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "05_detection_vs_snr.png", dpi=150)
    plt.close(fig)

    # 6) The blind spot: bias already wins, overtone not yet heard.
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    s = [b["start_ms"] for b in blind]
    cross = [b["crossover_snr"] for b in blind]
    heard = [b["snr_heard"] for b in blind]
    ax.fill_between(s, cross, heard, color="#ebe9e4", lw=0)
    ax.semilogy(s, cross, color=BLUE, marker="o",
                label="220 fundamental: τ₂₂₀ bias beats noise above this SNR")
    ax.semilogy(s, heard, color=ORANGE, marker="s",
                label="221 overtone: heard (Z ≥ 3) above this SNR")
    ax.text(5.1, 2600, "blind spot: biased,\nbut overtone not heard",
            color=INK_2, fontsize=9, va="center")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("Fit start time after the peak (ms)")
    ax.set_ylabel("Signal-to-noise ratio (SNR)")
    ax.set_title("Amplitude ratio 1, phase 0", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "06_blind_spot_vs_start.png", dpi=150)
    plt.close(fig)

    # 7) Heatmaps over amplitude ratio and phase (fit from the peak).
    cells = grid[0]
    shape = (len(D_RATIOS), len(D_PHASES_DEG))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, key, title, cmap in [
        (axes[0], "crossover_snr", "220 fundamental: SNR where τ₂₂₀ bias beats noise", BLUE_SEQ),
        (axes[1], "snr_heard", "221 overtone: SNR needed to hear it (Z = 3)", ORANGE_SEQ),
    ]:
        z = np.array([c[key] for c in cells]).reshape(shape)
        im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap,
                       norm=LogNorm(vmin=max(z.min(), 0.5), vmax=z.max()))
        for i in range(shape[0]):
            for j in range(shape[1]):
                light = z[i, j] < np.sqrt(z.min() * z.max())
                ax.text(j, i, f"{z[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#0b0b0b" if light else "#ffffff")
        ax.set_xticks(range(shape[1]), [str(p) if p % 60 == 0 else "" for p in D_PHASES_DEG])
        ax.set_yticks(range(shape[0]), [f"{r:g}" for r in D_RATIOS])
        ax.set_xlabel("Overtone phase at peak (degrees)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, format=PLAIN)
        ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000)
                 if z.min() <= v <= z.max()]
        cb.set_ticks(ticks if len(ticks) <= 6 else [v for v in ticks if v in (1, 10, 100, 1000, 10000)])
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes[0].set_ylabel("Amplitude ratio A₂₂₁ / A₂₂₀ at peak")
    fig.suptitle("Fit from the peak (0 ms): ratio and phase change everything",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "07_ratio_phase_maps.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="N = 30 instead of 200")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results" / "detection.json"
    if args.figures_only:
        r = json.loads(out.read_text())
        grid = {int(k): v for k, v in r["experiment_d"]["grid"].items()}
        make_figures(r["experiment_c"]["rows"], r["experiment_c"]["threshold"],
                     grid, r["experiment_d"]["blind_spot"])
        print("Redrew figures 5-7 from", out.relative_to(ROOT))
        return

    n_real = 30 if args.quick else 200
    rng = np.random.default_rng(SEED)
    t = time_axis()
    P = true_params(REMNANT_MASS, REMNANT_SPIN)
    start = time.time()
    print(f"Experiment C (N = {n_real})")
    c_rows, threshold = experiment_c(t, P, n_real, rng)
    print("Experiment D")
    grid, blind = experiment_d(t)
    make_figures(c_rows, threshold, grid, blind)
    out.write_text(json.dumps({
        "settings": {"n_realisations": n_real, "seed": SEED, "false_alarm": FALSE_ALARM,
                     "z_heard": Z_HEARD, "c_start_ms": C_START_MS, "d_start_ms": D_START_MS,
                     "ratios": D_RATIOS, "phases_deg": D_PHASES_DEG},
        "experiment_c": {"rows": c_rows, "threshold": threshold},
        "experiment_d": {"grid": grid, "blind_spot": blind},
    }, indent=2, default=float))
    print(f"Saved {out.relative_to(ROOT)} and figures 5-7 in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
