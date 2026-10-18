"""Run both experiments, save results/results.json and the four figures.

    python run_build.py           # full run, N = 200 noise realisations per point
    python run_build.py --quick   # fast run, N = 30
    python run_build.py --figures-only   # redraw figures from results.json

Experiment A: fit BOTH modes to noisy data. How well do we measure the
              overtone (221) frequency as the SNR grows?
Experiment B: fit ONLY the fundamental (220) to data that contain both modes.
              The missing overtone biases tau220. At what SNR does that bias
              become bigger than the noise error?
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ringdown_lab.qnm import qnm, REMNANT_MASS, REMNANT_SPIN
from ringdown_lab.signal import damped_sinusoid, time_axis, two_mode
from ringdown_lab.noise import white_noise, snr, scale_to_snr
from ringdown_lab.fit import fit_one_mode, fit_two_mode, fisher_errors

ROOT = Path(__file__).resolve().parent
SNRS = [8, 10, 15, 20, 30, 50, 80, 120, 200, 300]
START_TIMES_MS = [0, 1, 2, 3, 4, 6, 8]
SLOPE_MIN_SNR = 50        # fit the log-log slope using SNR >= 50
LOST_THRESHOLD = 0.20     # overtone "lost" if fitted f221 is >20% off
SEED = 2026               # fixed seed -> same numbers every run

# Colours (reference data-viz palette) and a clean plot style.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 7,
    "font.size": 11, "axes.titlesize": 12, "legend.frameon": False,
})


def half_width_68(x):
    """Half of the 16th-84th percentile range: a robust '1-sigma' spread."""
    p16, p84 = np.percentile(x, [16, 84])
    return (p84 - p16) / 2


def jitter(rng, values):
    """Multiply each value by a random factor between 0.9 and 1.1 (+/-10%)."""
    return np.asarray(values) * rng.uniform(0.9, 1.1, len(values))


def experiment_a(t, true_unit, n_real, rng):
    """Two-mode fits at each SNR: error on f221, % overtone lost, Fisher."""
    f221_true = true_unit[6]
    h_unit = two_mode(t, *true_unit)
    rows = []
    for target in SNRS:
        scale = target / snr(h_unit)
        true = true_unit.copy()
        true[[0, 1, 4, 5]] *= scale             # amplitudes grow with SNR
        h = h_unit * scale
        f221_fits, lost = [], 0
        for _ in range(n_real):
            data = h + white_noise(len(t), 1.0, rng)
            start = true.copy()
            start[[2, 3, 6, 7]] = jitter(rng, true[[2, 3, 6, 7]])
            params, ok = fit_two_mode(t, data, start)
            if not ok or abs(params[6] - f221_true) / f221_true > LOST_THRESHOLD:
                lost += 1
            if ok:
                f221_fits.append(params[6])
        rows.append({
            "snr": target,
            "f221_error_hz": half_width_68(f221_fits),
            "f221_fisher_hz": fisher_errors(t, true)[6],
            "percent_lost": 100 * lost / n_real,
            "n_failed": n_real - len(f221_fits),
        })
        print(f"  A  SNR {target:>3}: f221 error {rows[-1]['f221_error_hz']:7.2f} Hz"
              f" (Fisher {rows[-1]['f221_fisher_hz']:7.2f})  lost {rows[-1]['percent_lost']:5.1f}%")
    high = [r for r in rows if r["snr"] >= SLOPE_MIN_SNR]
    slope = np.polyfit(np.log10([r["snr"] for r in high]),
                       np.log10([r["f221_error_hz"] for r in high]), 1)[0]
    return rows, slope


def experiment_b(t, true_unit, n_real, rng):
    """One-mode fits starting at t0 after the peak: bias vs noise error."""
    f0, tau0 = true_unit[2], true_unit[3]
    h_unit = two_mode(t, *true_unit)            # SNR is set on the FULL signal
    out = []
    for t0_ms in START_TIMES_MS:
        keep = t >= t0_ms * 1e-3 - 1e-12
        tt = t[keep] - t[keep][0]                # restart the clock at t0
        t0 = t[keep][0]
        # Starting guess = the true 220 mode as it looks at time t0.
        env, phase = np.exp(-t0 / tau0), 2 * np.pi * f0 * t0
        start_unit = np.array([env * np.cos(phase), -env * np.sin(phase), f0, tau0])

        # Bias: fit the noiseless data (no noise -> any error is model error).
        params, ok = fit_one_mode(tt, h_unit[keep], start_unit)
        bias = params[3] - tau0

        # Noise error: spread of tau220 over many noisy fits, at each SNR.
        noise_err = []
        for target in SNRS:
            scale = target / snr(h_unit)
            data_clean = h_unit[keep] * scale
            fits = []
            for _ in range(n_real):
                data = data_clean + white_noise(len(tt), 1.0, rng)
                start = start_unit.copy()
                start[:2] *= scale
                start[2:] = jitter(rng, start_unit[2:])
                p, ok_n = fit_one_mode(tt, data, start)
                if ok_n:
                    fits.append(p[3])
            noise_err.append(half_width_68(fits))
        noise_err = np.array(noise_err)

        # noise error ~ k / SNR. Estimate k from the high-SNR points.
        hi = np.array(SNRS) >= SLOPE_MIN_SNR
        k = float(np.mean(noise_err[hi] * np.array(SNRS)[hi]))
        crossover = k / abs(bias)                # where k/SNR = |bias|
        out.append({
            "start_ms": t0_ms,
            "fit_ok": ok,
            "tau220_fit_ms": params[3] * 1e3,
            "bias_ms": bias * 1e3,
            "bias_percent": 100 * bias / tau0,
            "noise_error_ms": (noise_err * 1e3).tolist(),
            "k_ms": k * 1e3,
            "crossover_snr": crossover,
        })
        print(f"  B  start {t0_ms} ms: bias {out[-1]['bias_percent']:+7.2f}%"
              f"  k {out[-1]['k_ms']:8.3f} ms  crossover SNR {crossover:10.1f}")
    return out


PLAIN = FuncFormatter(lambda v, _: f"{v:g}")   # 0.01, 1, 100 instead of 10^x


def make_figures(a_rows, slope, b_rows, tau0):
    figdir = ROOT / "figures"
    snrs = np.array([r["snr"] for r in a_rows])
    tau0_ms = tau0 * 1e3

    # Colour rule for every figure: blue = 220 fundamental, orange = 221 overtone.

    # 0) The two modes on their own, with their decay envelopes.
    t = time_axis() * 1e3
    f0, tau0_s = qnm("220", REMNANT_MASS, REMNANT_SPIN)
    f1, tau1_s = qnm("221", REMNANT_MASS, REMNANT_SPIN)
    show = t <= 15
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for f, tau, color, name, lx, sign in [
        (f0, tau0_s, BLUE, "220 fundamental", 1.6 * tau0_s * 1e3, 1),
        (f1, tau1_s, ORANGE, "221 overtone", 1.2 * tau1_s * 1e3, -1),
    ]:
        env = np.exp(-t[show] / (tau * 1e3))
        ax.plot(t[show], env, color=color, lw=1, ls="--")
        ax.plot(t[show], -env, color=color, lw=1, ls="--")
        ax.plot(t[show], damped_sinusoid(t[show] / 1e3, 1, 0, f, tau), color=color, lw=2)
        ax.annotate(f"{name}: {f:.0f} Hz, τ = {tau * 1e3:.2f} ms",
                    (lx, sign * np.exp(-lx / (tau * 1e3))), xytext=(10, 12 * sign),
                    textcoords="offset points", color=INK, fontsize=10, fontweight="bold",
                    va="center")
    ax.set_xlim(0, 15)
    ax.set_xlabel("Time after the peak (ms)")
    ax.set_ylabel("Strain (amplitude 1 at the peak)")
    ax.set_title("Same pitch, different lifetimes. Dashed: how fast each mode fades", loc="left")
    fig.tight_layout()
    fig.savefig(figdir / "00_two_modes.png", dpi=150)
    plt.close(fig)

    # 1) Overtone frequency error vs SNR, with the Fisher prediction.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(snrs, [r["f221_fisher_hz"] for r in a_rows], color=INK_2,
              ls="--", label="Fisher prediction")
    ax.loglog(snrs, [r["f221_error_hz"] for r in a_rows], color=ORANGE,
              marker="o", label=f"Simulated fits (slope above SNR 50: {slope:.2f})")
    ax.set_xlabel("Signal-to-noise ratio (SNR)")
    ax.set_ylabel("Error on overtone frequency $f_{221}$ (Hz)")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_yticks([10, 20, 50, 100, 200, 500])
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title("221 overtone: frequency error falls as 1/SNR at high SNR", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "01_overtone_error_vs_snr.png", dpi=150)
    plt.close(fig)

    # 2) Percentage of fits where the overtone is lost.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(snrs, [r["percent_lost"] for r in a_rows], color=ORANGE, marker="o")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Signal-to-noise ratio (SNR)")
    ax.set_ylabel("Overtone lost (% of fits)")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_title("221 overtone lost: fitted $f_{221}$ more than 20% off", loc="left")
    fig.tight_layout()
    fig.savefig(figdir / "02_overtone_lost_vs_snr.png", dpi=150)
    plt.close(fig)

    # 3) Bias (flat lines) vs noise error (k/SNR lines), crossover dots.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    snr_line = np.logspace(np.log10(2), np.log10(3e4), 200)
    for row, color in zip(b_rows, BLUE_RAMP):
        bias_pct = abs(row["bias_percent"])
        noise_pct = 100 * row["k_ms"] / tau0_ms / snr_line
        ax.loglog(snr_line, noise_pct, color=color, lw=1.5)
        ax.axhline(bias_pct, color=color, lw=1.5, ls=":")
        x = row["crossover_snr"]
        if snr_line[0] <= x <= snr_line[-1]:
            ax.plot(x, bias_pct, "o", color=color, mec="#fcfcfb", mew=2, ms=9)
        ax.plot([], [], color=color, label=f"start {row['start_ms']} ms")
    ax.set_xlim(snr_line[0], snr_line[-1])
    ax.set_ylim(1e-3, 300)
    ax.set_xlabel("Signal-to-noise ratio (SNR)")
    ax.set_ylabel(r"Error on fundamental $\tau_{220}$ (% of true value)")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_title("220 fundamental, τ₂₂₀ — solid: noise k/SNR, dotted: |bias|, dot: crossover",
                 loc="left", fontsize=11)
    ax.legend(ncol=2, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(figdir / "03_bias_vs_noise_crossover.png", dpi=150)
    plt.close(fig)

    # 4) Bias and noise error vs start time (noise error at a few SNRs).
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    starts = [r["start_ms"] for r in b_rows]
    ax.semilogy(starts, [abs(r["bias_percent"]) for r in b_rows], color=BLUE,
                marker="o", lw=2.5, label="|bias| from ignoring the overtone (no noise)")
    for target, color in [(20, BLUE_RAMP[0]), (120, BLUE_RAMP[5])]:
        i = SNRS.index(target)
        ax.semilogy(starts, [100 * r["noise_error_ms"][i] / tau0_ms for r in b_rows],
                    color=color, ls="--", marker="s", label=f"noise error, SNR {target}")
    ax.set_xlabel("Fit start time after the peak (ms)")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_ylabel(r"Error on fundamental $\tau_{220}$ (% of true value)")
    ax.set_title("220 fundamental: later start, smaller bias, bigger noise", loc="left")
    ax.set_ylim(0.08, 400)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "04_bias_and_noise_vs_start.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="N = 30 instead of 200")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures from results/results.json")
    args = parser.parse_args()
    out = ROOT / "results" / "results.json"

    if args.figures_only:
        res = json.loads(out.read_text())
        make_figures(res["experiment_a"]["rows"], res["experiment_a"]["slope_above_snr_50"],
                     res["experiment_b"], res["remnant"]["tau220_ms"] / 1e3)
        print("Redrew 4 figures from", out.relative_to(ROOT))
        return

    n_real = 30 if args.quick else 200

    rng = np.random.default_rng(SEED)
    t = time_axis()
    f0, tau0 = qnm("220", REMNANT_MASS, REMNANT_SPIN)
    f1, tau1 = qnm("221", REMNANT_MASS, REMNANT_SPIN)
    # Amplitude ratio 1, phase 0: a0 = a1 = 1, b0 = b1 = 0.
    true_unit = np.array([1.0, 0.0, f0, tau0, 1.0, 0.0, f1, tau1])

    t_start = time.time()
    print(f"Experiment A (N = {n_real})")
    a_rows, slope = experiment_a(t, true_unit, n_real, rng)
    print(f"  log-log slope above SNR {SLOPE_MIN_SNR}: {slope:.3f}")
    print(f"Experiment B (N = {n_real})")
    b_rows = experiment_b(t, true_unit, n_real, rng)
    make_figures(a_rows, slope, b_rows, tau0)

    results = {
        "settings": {"n_realisations": n_real, "seed": SEED, "snrs": SNRS,
                     "start_times_ms": START_TIMES_MS, "fs_hz": 4096,
                     "duration_s": 0.1, "amplitude_ratio": 1.0, "phase": 0.0},
        "remnant": {"mass_msun": REMNANT_MASS, "spin": REMNANT_SPIN,
                    "f220_hz": f0, "tau220_ms": tau0 * 1e3,
                    "f221_hz": f1, "tau221_ms": tau1 * 1e3},
        "experiment_a": {"rows": a_rows, "slope_above_snr_50": slope},
        "experiment_b": b_rows,
    }
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out.relative_to(ROOT)} and 4 figures in "
          f"{time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
