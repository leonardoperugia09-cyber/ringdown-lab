// Overtone Explorer: physics and numerics (no UI code).
// Berti, Cardoso & Will (2006) QNM fits, two damped sinusoids, white Gaussian noise,
// Fisher-matrix significance and Levenberg-Marquardt least-squares fits.
// Ported line for line from the Ringdown Lab Python package and checked against it.
"use strict";
// ---------------------------------------------------------------- physics
const FS = 4096, N = Math.floor(FS * 0.1);          // 409 samples, like the Python code
const MSUN_S = 4.925491e-6;
const COEFFS = {
  "220": [1.5251, -1.1568, 0.1292, 0.7000, 1.4187, -0.4990],
  "221": [1.3673, -1.0260, 0.1628, 0.1000, 0.5436, -0.4731],
};
const T = Float64Array.from({ length: N }, (_, i) => i / FS);
const Z_HEARD = 3, Z_HINT = 2;
// Delta-chi2 threshold for 1% false alarms, from Experiment C (results/detection.json).
const DCHI2_THRESHOLD = 15.70;
const BOUNDS1 = { lo: [-Infinity, -Infinity, 20, 1e-5], hi: [Infinity, Infinity, 2000, 0.1] };
const BOUNDS2 = { lo: BOUNDS1.lo.concat(BOUNDS1.lo), hi: BOUNDS1.hi.concat(BOUNDS1.hi) };

function qnm(mode, mass, spin) {
  const [f1, f2, f3, q1, q2, q3] = COEFFS[mode];
  const F = f1 + f2 * Math.pow(1 - spin, f3);
  const Q = q1 + q2 * Math.pow(1 - spin, q3);
  const f = F / (2 * Math.PI * mass * MSUN_S);
  return [f, Q / (Math.PI * f)];
}
function ds(t, a, b, f, tau) {
  const out = new Float64Array(t.length), w = 2 * Math.PI * f;
  for (let i = 0; i < t.length; i++) out[i] = Math.exp(-t[i] / tau) * (a * Math.cos(w * t[i]) + b * Math.sin(w * t[i]));
  return out;
}
const oneModel = (t, p) => ds(t, p[0], p[1], p[2], p[3]);
function twoModel(t, p) {
  const h = ds(t, p[0], p[1], p[2], p[3]), g = ds(t, p[4], p[5], p[6], p[7]);
  for (let i = 0; i < h.length; i++) h[i] += g[i];
  return h;
}
function kerrModel(t, q) {      // q = [a0, b0, a1, b1, mass, spin]
  const [f0, t0] = qnm("220", q[4], q[5]), [f1, t1] = qnm("221", q[4], q[5]);
  return twoModel(t, [q[0], q[1], f0, t0, q[2], q[3], f1, t1]);
}
function trueParams(mass, spin, ratio, phaseRad) {
  const [f0, t0] = qnm("220", mass, spin), [f1, t1] = qnm("221", mass, spin);
  return [1, 0, f0, t0, ratio * Math.cos(phaseRad), -ratio * Math.sin(phaseRad), f1, t1];
}
function shiftToStart(p, t0) {    // same waveform, clock restarted at t0
  const out = p.slice();
  for (const i of [0, 4]) {
    const [a, b, f, tau] = p.slice(i, i + 4), env = Math.exp(-t0 / tau), ph = 2 * Math.PI * f * t0;
    out[i] = env * (a * Math.cos(ph) + b * Math.sin(ph));
    out[i + 1] = env * (b * Math.cos(ph) - a * Math.sin(ph));
  }
  return out;
}
const norm = (h) => { let s = 0; for (const x of h) s += x * x; return Math.sqrt(s); };
function windowFrom(t0ms) {
  let i0 = 0;
  while (i0 < N - 8 && T[i0] < t0ms * 1e-3 - 1e-12) i0++;
  return { i0, t0: T[i0], tt: Float64Array.from(T.subarray(i0), (x) => x - T[i0]) };
}

// ------------------------------------------------------------ linear algebra
function jacobian(model, t, p) {
  return p.map((pi, i) => {
    const step = 1e-6 * Math.max(Math.abs(pi), 1e-3), up = p.slice(), dn = p.slice();
    up[i] += step; dn[i] -= step;
    const hu = model(t, up), hd = model(t, dn), row = new Float64Array(t.length);
    for (let k = 0; k < t.length; k++) row[k] = (hu[k] - hd[k]) / (2 * step);
    return row;
  });
}
function gram(J) {
  const n = J.length, A = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    let s = 0; const a = J[i], b = J[j];
    for (let k = 0; k < a.length; k++) s += a[k] * b[k];
    A[i][j] = A[j][i] = s;
  }
  return A;
}
function invert(A) {              // Gauss-Jordan with partial pivoting
  const n = A.length, M = A.map((r) => r.slice()), I = A.map((_, i) => A.map((_, j) => (i === j ? 1 : 0)));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (!(Math.abs(M[piv][c]) > 0)) return null;
    [M[c], M[piv]] = [M[piv], M[c]]; [I[c], I[piv]] = [I[piv], I[c]];
    const d = M[c][c];
    for (let j = 0; j < n; j++) { M[c][j] /= d; I[c][j] /= d; }
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = M[r][c];
      if (f) for (let j = 0; j < n; j++) { M[r][j] -= f * M[c][j]; I[r][j] -= f * I[c][j]; }
    }
  }
  return I;
}
const covariance = (model, t, p) => invert(gram(jacobian(model, t, p)));   // sigma = 1
function ampSig(C, a, b, ia, ib) {
  const A = Math.hypot(a, b);
  return A / Math.sqrt((a * a * C[ia][ia] + 2 * a * b * C[ia][ib] + b * b * C[ib][ib]) / (A * A));
}
function zFree(t, p) { const C = covariance(twoModel, t, p); return C ? ampSig(C, p[4], p[5], 4, 5) : NaN; }
function zKerr(t, p, mass, spin) {
  const q = [p[0], p[1], p[4], p[5], mass, spin], C = covariance(kerrModel, t, q);
  return C ? ampSig(C, q[2], q[3], 2, 3) : NaN;
}

// Levenberg-Marquardt least squares with simple box bounds.
function lmFit(model, t, y, p0, bounds, maxIter = 300) {
  const clip = (p) => p.map((v, i) => Math.min(bounds.hi[i], Math.max(bounds.lo[i], v)));
  const chi2Of = (p) => { const m = model(t, p); let s = 0; for (let k = 0; k < y.length; k++) { const r = y[k] - m[k]; s += r * r; } return s; };
  let p = clip(p0), chi = chi2Of(p), lam = 1e-3;
  for (let it = 0; it < maxIter; it++) {
    const J = jacobian(model, t, p), A = gram(J), m = model(t, p);
    const g = J.map((row) => { let s = 0; for (let k = 0; k < y.length; k++) s += row[k] * (y[k] - m[k]); return s; });
    let accepted = false;
    for (let tries = 0; tries < 12; tries++) {
      const Aug = A.map((r, i) => r.map((v, j) => (i === j ? v * (1 + lam) + 1e-300 : v)));
      const Inv = invert(Aug);
      if (!Inv) { lam *= 10; continue; }
      const step = Inv.map((r) => r.reduce((s, v, j) => s + v * g[j], 0));
      const pn = clip(p.map((v, i) => v + step[i])), cn = chi2Of(pn);
      if (cn < chi) {
        const gain = chi - cn;
        p = pn; chi = cn; lam = Math.max(lam / 10, 1e-12); accepted = true;
        if (gain <= 1e-13 * chi + 1e-300) return { p, chi2: chi };
        break;
      }
      lam *= 10;
    }
    if (!accepted) break;
  }
  return { p, chi2: chi };
}

// ------------------------------------------------------------ the analysis
function analyse(s, opts = {}) {
  const P = trueParams(s.mass, s.spin, s.ratio, (s.phase * Math.PI) / 180);
  const hFull = twoModel(T, P);
  const { i0, t0, tt } = windowFrom(s.t0);
  const p = shiftToStart(P, t0);
  const ref = s.snrFrom === "start" ? norm(twoModel(tt, p)) : norm(hFull);
  const k = s.snr / ref;                          // amplitude scale for the chosen SNR
  const ps = p.map((v, i) => ([0, 1, 4, 5].includes(i) ? v * k : v));
  const hs = twoModel(tt, ps);
  const out = { P, ps, i0, t0, tt, scale: k };
  out.snrPeak = norm(hFull) * k;
  out.snrStart = norm(hs);
  out.rho220 = norm(oneModel(tt, ps.slice(0, 4)));
  out.rho221 = norm(oneModel(tt, ps.slice(4)));
  out.Z = zFree(tt, ps);
  out.ZK = zKerr(tt, ps, s.mass, s.spin);
  const C = covariance(twoModel, tt, ps);
  const err = C ? C.map((r, i) => Math.sqrt(r[i])) : new Array(8).fill(NaN);
  out.err = err;
  out.resolvedF = Math.abs(ps[2] - ps[6]) > Math.max(err[2], err[6]);
  out.resolvedTau = Math.abs(ps[3] - ps[7]) > Math.max(err[3], err[7]);
  const fit = lmFit(oneModel, tt, hs, ps.slice(0, 4), BOUNDS1).p;
  out.fit1 = fit;
  const resid = oneModel(tt, fit); for (let i = 0; i < resid.length; i++) resid[i] = hs[i] - resid[i];
  out.rhoRes = norm(resid);
  out.biasTau = (fit[3] - P[3]) / P[3];
  out.biasF = (fit[2] - P[2]) / P[2];
  const C1 = covariance(oneModel, tt, fit);
  out.noiseTau = C1 ? Math.sqrt(C1[3][3]) / P[3] : NaN;
  out.crossover = s.snr * out.noiseTau / Math.abs(out.biasTau);
  out.snrHeard = s.snr * Z_HEARD / out.Z;
  out.snrHeardK = s.snr * Z_HEARD / out.ZK;
  out.indistinguishableBelow = s.snr / out.rhoRes;
  if (opts.phaseScan) {
    out.phaseZ = [];
    for (let ph = 0; ph < 360; ph += 30) {
      const q = shiftToStart(trueParams(s.mass, s.spin, s.ratio, (ph * Math.PI) / 180), t0);
      const r = s.snrFrom === "start" ? norm(twoModel(tt, q)) : norm(twoModel(T, trueParams(s.mass, s.spin, s.ratio, (ph * Math.PI) / 180)));
      const qs = q.map((v, i) => ([0, 1, 4, 5].includes(i) ? (v * s.snr) / r : v));
      out.phaseZ.push([ph, zFree(tt, qs)]);
    }
  }
  return out;
}

function scanStart(s) {
  const pts = [];
  for (let t0 = 0; t0 <= 8.0001; t0 += 0.25) {
    const a = analyse({ ...s, t0 });
    pts.push({ t0, Z: a.Z, ZK: a.ZK, bias: Math.abs(a.biasTau) * 100, noise: a.noiseTau * 100 });
  }
  return pts;
}

// ---------------------------------------------------------------- noise
function rng(seed) {                       // mulberry32 + Box-Muller
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { u, normal() { let x = 0; while (x === 0) x = u(); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * u()); } };
}
function noiseFor(seed, n) { const r = rng(seed); return Float64Array.from({ length: n }, () => r.normal()); }

function noisyTrial(a, seed) {
  const r = rng(seed * 7919 + 13);
  const n = noiseFor(seed, a.tt.length), y = twoModel(a.tt, a.ps);
  for (let i = 0; i < y.length; i++) y[i] += n[i];
  const jit = (v) => v * (0.9 + 0.2 * r.u());
  const s1 = a.ps.slice(0, 4); s1[2] = jit(s1[2]); s1[3] = jit(s1[3]);
  const s2 = a.ps.slice(); for (const i of [2, 3, 6, 7]) s2[i] = jit(s2[i]);
  const f1 = lmFit(oneModel, a.tt, y, s1, BOUNDS1), f2 = lmFit(twoModel, a.tt, y, s2, BOUNDS2, 400);
  let p2 = f2.p;
  if (p2[3] < p2[7]) p2 = p2.slice(4).concat(p2.slice(0, 4));   // longer-lived = 220
  return { y, d: Math.max(f1.chi2 - f2.chi2, 0), fit1: f1.p, fit2: p2 };
}
