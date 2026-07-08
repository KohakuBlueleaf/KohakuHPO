/* Shared utilities for the S3-TuRBO demos: seeded RNG, Beta/Gaussian samplers, small dense linear
 * algebra (Cholesky), a miniature exact GP with joint posterior sampling, color helpers, and canvas
 * setup with devicePixelRatio handling. No dependencies. */
"use strict";

/* ---------------- seeded RNG ---------------- */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class RNG {
  constructor(seed) { this.u = mulberry32(seed); this._spare = null; }
  random() { return this.u(); }
  int(n) { return Math.floor(this.u() * n); }
  normal() {
    if (this._spare !== null) { const v = this._spare; this._spare = null; return v; }
    let u1 = 0, u2 = 0;
    do { u1 = this.u(); } while (u1 <= 1e-12);
    u2 = this.u();
    const r = Math.sqrt(-2 * Math.log(u1));
    this._spare = r * Math.sin(2 * Math.PI * u2);
    return r * Math.cos(2 * Math.PI * u2);
  }
  uniform(lo, hi) { return lo + (hi - lo) * this.u(); }
  /* Marsaglia-Tsang gamma sampler; boosted for shape < 1. */
  gamma(shape) {
    if (shape < 1) {
      const g = this.gamma(shape + 1);
      return g * Math.pow(Math.max(this.u(), 1e-16), 1 / shape);
    }
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = this.normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = this.u();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  beta(a, b) {
    const g1 = this.gamma(a), g2 = this.gamma(b);
    const s = g1 + g2;
    if (s <= 0 || !isFinite(s)) return this.u() < a / (a + b) ? 1 : 0; // deep-polarized underflow
    return g1 / s;
  }
}

/* ---------------- small linear algebra ---------------- */
/* In-place Cholesky of a dense symmetric PD matrix stored row-major (n x n). Adds growing jitter on
 * failure. Returns the lower factor L (same buffer). */
function cholesky(A, n) {
  for (let jit = 1e-10; jit < 1e-2; jit *= 10) {
    const L = A.slice();
    let ok = true;
    for (let i = 0; i < n; i++) L[i * n + i] += jit;
    for (let j = 0; j < n && ok; j++) {
      let d = L[j * n + j];
      for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
      if (d <= 0) { ok = false; break; }
      const dj = Math.sqrt(d);
      L[j * n + j] = dj;
      for (let i = j + 1; i < n; i++) {
        let s = L[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        L[i * n + j] = s / dj;
      }
    }
    if (ok) {
      for (let j = 0; j < n; j++) for (let i = 0; i < j; i++) L[i * n + j] = 0;
      return L;
    }
  }
  throw new Error("cholesky failed");
}

function solveLower(L, n, b) { // L y = b
  const y = b.slice();
  for (let i = 0; i < n; i++) {
    let s = y[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  return y;
}
function solveUpperT(L, n, y) { // L^T x = y
  const x = y.slice();
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/* ---------------- miniature exact GP (RBF kernel) ---------------- */
/* X: array of points (each an array of length dim), y: array of values. Fixed lengthscale + signal
 * variance from data, small noise. Exposes joint posterior sampling on a test set, which is exactly
 * what discretized batch Thompson sampling needs. */
class MiniGP {
  constructor(X, y, lengthscale, noise = 1e-6) {
    this.X = X; this.n = X.length; this.dim = X[0].length;
    this.ls = lengthscale;
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    let v = 0;
    for (const yi of y) v += (yi - mean) * (yi - mean);
    this.sig2 = Math.max(v / Math.max(y.length - 1, 1), 1e-10);
    this.mean = mean;
    this.noise = noise * this.sig2 + 1e-12;
    const n = this.n;
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = i; j < n; j++) {
        const k = this.k(X[i], X[j]);
        K[i * n + j] = k; K[j * n + i] = k;
      }
    for (let i = 0; i < n; i++) K[i * n + i] += this.noise;
    this.L = cholesky(K, n);
    const yc = y.map((yi) => yi - mean);
    this.alpha = solveUpperT(this.L, n, solveLower(this.L, n, yc));
  }
  k(a, b) {
    let d2 = 0;
    for (let j = 0; j < a.length; j++) { const d = a[j] - b[j]; d2 += d * d; }
    return this.sig2 * Math.exp(-0.5 * d2 / (this.ls * this.ls));
  }
  /* Posterior mean and covariance on test points T (m x dim). Returns {mu, cov}. */
  posterior(T) {
    const n = this.n, m = T.length;
    const Ks = new Float64Array(m * n); // k(T, X)
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) Ks[i * n + j] = this.k(T[i], this.X[j]);
    const mu = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = this.mean;
      for (let j = 0; j < n; j++) s += Ks[i * n + j] * this.alpha[j];
      mu[i] = s;
    }
    // V = L^{-1} Ks^T  (n x m), cov = K** - V^T V
    const V = new Float64Array(n * m);
    for (let c = 0; c < m; c++) {
      const col = new Float64Array(n);
      for (let j = 0; j < n; j++) col[j] = Ks[c * n + j];
      const v = solveLower(this.L, n, col);
      for (let j = 0; j < n; j++) V[j * m + c] = v[j];
    }
    const cov = new Float64Array(m * m);
    for (let i = 0; i < m; i++)
      for (let j = i; j < m; j++) {
        let s = this.k(T[i], T[j]);
        for (let r = 0; r < n; r++) s -= V[r * m + i] * V[r * m + j];
        if (i === j) s = Math.max(s, 1e-12);
        cov[i * m + j] = s; cov[j * m + i] = s;
      }
    return { mu, cov };
  }
  /* Draw nDraws joint posterior samples on T. Returns array of Float64Array(m). */
  sample(T, nDraws, rng) {
    const m = T.length;
    const { mu, cov } = this.posterior(T);
    const Lc = cholesky(cov, m);
    const draws = [];
    for (let d = 0; d < nDraws; d++) {
      const z = new Float64Array(m);
      for (let i = 0; i < m; i++) z[i] = rng.normal();
      const s = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        let acc = mu[i];
        for (let k = 0; k <= i; k++) acc += Lc[i * m + k] * z[k];
        s[i] = acc;
      }
      draws.push(s);
    }
    return { draws, mu, sd: Array.from({ length: m }, (_, i) => Math.sqrt(cov[i * m + i])) };
  }
}

/* ---------------- color helpers ---------------- */
function hexToRgb(h) {
  const x = parseInt(h.slice(1), 16);
  return [(x >> 16) & 255, (x >> 8) & 255, x & 255];
}
function lerpColorStops(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
      const a = hexToRgb(c0), b = hexToRgb(c1);
      return [
        Math.round(a[0] + (b[0] - a[0]) * u),
        Math.round(a[1] + (b[1] - a[1]) * u),
        Math.round(a[2] + (b[2] - a[2]) * u),
      ];
    }
  }
  return hexToRgb(stops[stops.length - 1][1]);
}
/* Landscape colormap: low values (good) glow bright cyan, high values fade into the slate background. */
const LANDSCAPE_STOPS = [
  [0.0, "#f0fdff"], [0.06, "#7ff3ff"], [0.18, "#22d3ee"], [0.38, "#2563eb"],
  [0.62, "#1e3a6e"], [0.85, "#16233f"], [1.0, "#0d1526"],
];

/* ---------------- canvas helpers ---------------- */
/* Size a canvas for crisp rendering: CSS width from layout, backing store scaled by dpr.
 * Returns ctx with a transform such that drawing coordinates are CSS pixels. */
function setupCanvas(canvas, cssW, cssH) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
function quantileSorted(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function quantile(arr, q) { return quantileSorted(Array.from(arr).sort((a, b) => a - b), q); }

/* Robust value scale S_y = max(IQR, 1.4826 MAD, small floor). Mirrors the optimizer. */
function robustScale(y) {
  if (y.length < 4) return 1.0;
  const s = Array.from(y).sort((a, b) => a - b);
  const iqr = quantileSorted(s, 0.75) - quantileSorted(s, 0.25);
  const med = quantileSorted(s, 0.5);
  const mad = quantile(y.map((v) => Math.abs(v - med)), 0.5) * 1.4826;
  const floor = 0.02 * Math.max(1.0, Math.abs(s[0]));
  return Math.max(iqr, mad, floor, 1e-12);
}

/* Debounced-on-resize observer that re-runs a draw callback when a demo becomes visible/resizes. */
function onResize(el, cb) {
  let raf = 0;
  const ro = new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(cb);
  });
  ro.observe(el);
  return ro;
}
