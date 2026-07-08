/* Adaptive soft mask demo.
 *
 * Runs a faithful miniature of the adaptive-mask credit loop on a synthetic anisotropic objective
 * where only a few of d coordinates actually matter. As improving moves land, per-coordinate credit
 * accrues (attributed by squared realized displacement), confidence rises, and the concentration c0
 * sharpens the mask from soft toward hard on the active set -- as in src:
 *
 *   s_j    <- lambda s_j + sum_i  max(0, y*-y_i)/S_y * (delta_ij^2 / sum_k delta_ik^2)
 *   p_j     = s_j / sum s
 *   C       = 1 - H(p)/log d
 *   c0_t    = exp((1-C) log c_max + C log c_min)      (rho = 1/sqrt(d), derived, not learned)
 *
 * Left  : per-coordinate credit bars, active set highlighted; a live mask draw overlaid.
 * Right : confidence C, effective dimension k_eff, derived rho, learned c0, best-so-far.
 */
"use strict";

(function () {
  const root = document.getElementById("demo-adaptive");
  if (!root) return;

  const cv = root.querySelector(".ad-canvas");
  const statBest = root.querySelector(".ad-best");
  const statConf = root.querySelector(".ad-conf");
  const statKeff = root.querySelector(".ad-keff");
  const statRho = root.querySelector(".ad-rho");
  const statC0 = root.querySelector(".ad-c0");
  const statEval = root.querySelector(".ad-eval");
  const dSlider = root.querySelector(".ad-d");
  const dVal = root.querySelector(".ad-d-val");
  const kSlider = root.querySelector(".ad-k");
  const kVal = root.querySelector(".ad-k-val");
  const btnPlay = root.querySelector(".ad-play");
  const btnStep = root.querySelector(".ad-step");
  const btnReset = root.querySelector(".ad-reset");

  // adaptive knobs. LAMBDA is longer than the src default (0.92) here purely so the accumulated
  // credit stays visible after the objective is solved; the credit -> concentration math matches src.
  const LAMBDA = 0.985, C_MIN = 0.03, C_MAX = 1.2, Q = 4;
  const state = { d: 24, k: 3, playing: false, timer: null };
  // `credit` is the src decayed credit that drives c0; `creditShown` is a non-decayed accumulator
  // used ONLY for the display bars, so the concentration stays visible after descent.
  let rng, active, credit, creditShown, best, bestU, evals, rhoT, c0T, conf, kEff, curU, lastAlpha;

  // rho = 1/sqrt(d), the derived active fraction; not learned. Only c0 is learned.
  function rhoValue() {
    return 1 / Math.sqrt(state.d);
  }

  function reset() {
    rng = new RNG(12345 + state.d * 7 + state.k);
    // active set = the first k coordinates (shuffled positions for realism)
    active = new Set();
    const perm = Array.from({ length: state.d }, (_, i) => i);
    for (let i = perm.length - 1; i > 0; i--) { const j = rng.int(i + 1);[perm[i], perm[j]] = [perm[j], perm[i]]; }
    for (let i = 0; i < state.k; i++) active.add(perm[i]);
    credit = new Float64Array(state.d);
    creditShown = new Float64Array(state.d);
    curU = Array.from({ length: state.d }, () => 0.5);
    best = objective(curU); bestU = curU.slice();
    evals = 1;
    rhoT = rhoValue(); c0T = C_MAX; conf = 0; kEff = state.d;
    lastAlpha = null;
    draw();
    refreshStats();
  }

  // anisotropic + mildly multimodal on the active coords, so descent takes many improving steps
  // (a pure quadratic solves in a handful of steps and credit never accrues enough to watch)
  function objective(u) {
    let s = 0, i = 0;
    for (const j of active) {
      const t = 0.2 + 0.6 * ((j * 2654435761) % 1000) / 1000;
      const d = u[j] - t;
      s += (i + 1) * (d * d + 0.06 * (1 - Math.cos(9 * Math.PI * d)));  // ripples slow the descent
      i++;
    }
    return s;
  }

  function sampleAlpha() {
    // soft mask at (rhoT, c0T) over d coords
    const a = new Float64Array(state.d);
    const al = Math.max(rhoT * c0T, 1e-3), be = Math.max((1 - rhoT) * c0T, 1e-3);
    let sum = 0;
    for (let j = 0; j < state.d; j++) { a[j] = rng.beta(al, be); sum += a[j]; }
    const target = rhoT * state.d;
    for (let j = 0; j < state.d; j++) a[j] = clamp((a[j] * target) / Math.max(sum, 1e-12), 0, 1);
    return a;
  }

  function refreshMaskShape() {
    rhoT = rhoValue();
    const total = credit.reduce((x, y) => x + y, 0);
    if (total <= 1e-12) { c0T = C_MAX; conf = 0; kEff = state.d; return; }
    let ent = 0, sq = 0;
    for (let j = 0; j < state.d; j++) { const p = (credit[j] + 1e-12) / (total + 1e-12 * state.d); ent += -p * Math.log(p); sq += p * p; }
    conf = clamp(1 - ent / Math.log(Math.max(state.d, 2)), 0, 1);
    kEff = clamp(1 / sq, 1, state.d);
    c0T = Math.exp((1 - conf) * Math.log(C_MAX) + conf * Math.log(C_MIN));  // c0: learned from confidence
  }

  function step() {
    // one batch: q masked local moves around the incumbent, keep the best, accrue credit
    const before = best;
    const Sy = 0.5 * Math.max(1e-6, Math.abs(best)) + 1e-3;
    for (let j = 0; j < state.d; j++) credit[j] *= LAMBDA;
    let batchAlpha = null;
    for (let s = 0; s < Q; s++) {
      const alpha = sampleAlpha();
      const u = bestU.slice();
      const disp = new Float64Array(state.d);
      const stepScale = 0.06 + 0.14 * (1 - conf);  // shrink the step as the mask sharpens
      let dsq = 0;
      for (let j = 0; j < state.d; j++) {
        u[j] = clamp(bestU[j] + alpha[j] * rng.uniform(-1, 1) * stepScale, 0, 1);
        disp[j] = u[j] - bestU[j];
        dsq += disp[j] * disp[j];
      }
      const y = objective(u);
      const improvement = Math.max(0, before - y) / Sy;
      // credit attributed by squared realized displacement (as in src), not by the mask weight
      if (improvement > 0 && dsq > 1e-18) for (let j = 0; j < state.d; j++) {
        const share = improvement * (disp[j] * disp[j]) / dsq;
        credit[j] += share;
        creditShown[j] += share;  // non-decayed, for the display bars only
      }
      if (y < best) { best = y; bestU = u.slice(); batchAlpha = alpha; }
      evals++;
    }
    if (batchAlpha) lastAlpha = batchAlpha;
    refreshMaskShape();
    draw();
    refreshStats();
    return best < before - 1e-9;
  }

  function refreshStats() {
    statBest.textContent = best.toExponential(2);
    statConf.textContent = conf.toFixed(2);
    statKeff.textContent = kEff.toFixed(1) + " / " + state.d;
    statRho.textContent = rhoT.toFixed(3);
    statC0.textContent = c0T.toFixed(3);
    statEval.textContent = evals;
  }

  function draw() {
    const w = cv.parentElement.clientWidth - 2, h = 240;
    const ctx = setupCanvas(cv, w, h);
    ctx.clearRect(0, 0, w, h);
    const padL = 8, padT = 14, padB = 28;
    const plotW = w - padL - 8, plotH = h - padT - padB;
    const bw = plotW / state.d;
    const maxC = Math.max(1e-9, ...creditShown);
    for (let j = 0; j < state.d; j++) {
      const x = padL + j * bw;
      // credit bar (cumulative, for legibility)
      const ch = (creditShown[j] / maxC) * plotH;
      const isActive = active.has(j);
      ctx.fillStyle = isActive ? "rgba(52,211,153,0.85)" : "rgba(100,116,139,0.45)";
      ctx.fillRect(x + 1, padT + plotH - ch, bw - 2, ch);
      // live mask draw overlaid as a thin cyan cap
      if (lastAlpha) {
        const mh = lastAlpha[j] * plotH;
        ctx.fillStyle = "rgba(34,211,238,0.9)";
        ctx.fillRect(x + bw * 0.35, padT + plotH - mh, Math.max(1.5, bw * 0.3), Math.max(1.5, mh));
      }
    }
    ctx.fillStyle = "#64748b";
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--mono");
    ctx.textAlign = "left";
    ctx.fillText("coordinate credit s_j (green = truly active)  ·  cyan = last mask draw", padL, h - 8);
  }

  function play() { if (state.playing) return; state.playing = true; btnPlay.textContent = "❚❚ pause"; state.timer = setInterval(() => { step(); if (evals > 600) pause(); }, 90); }
  function pause() { state.playing = false; btnPlay.textContent = "▶ play"; if (state.timer) clearInterval(state.timer); state.timer = null; }

  btnPlay.addEventListener("click", () => (state.playing ? pause() : play()));
  btnStep.addEventListener("click", () => { pause(); step(); });
  btnReset.addEventListener("click", () => { pause(); reset(); });
  dSlider.addEventListener("input", () => { state.d = parseInt(dSlider.value, 10); dVal.textContent = state.d; pause(); reset(); });
  kSlider.addEventListener("input", () => { state.k = Math.min(parseInt(kSlider.value, 10), state.d); kVal.textContent = state.k; pause(); reset(); });

  onResize(root, draw);
  reset();
})();
