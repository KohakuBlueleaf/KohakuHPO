/* Discretized batch Thompson sampling on a 1D GP.
 *
 * A GP is fit on the observed points. A finite candidate pool is laid down (the discretization),
 * q independent posterior function realizations are drawn jointly on a fine grid, and each
 * realization's pool-minimizer becomes one batch point. "Evaluate batch" plays one loop iteration:
 * the chosen batch is evaluated on the true function and the GP refits.
 */
"use strict";

(function () {
  const root = document.getElementById("demo-ts");
  if (!root) return;

  const cv = root.querySelector(".ts-canvas");
  const qSlider = root.querySelector(".ctl-q");
  const qVal = root.querySelector(".ctl-q-val");
  const chkTrue = root.querySelector(".ctl-true");
  const chkMean = root.querySelector(".ctl-mean");

  const GRID_N = 150, POOL_N = 44;
  const DRAW_COLORS = ["#22d3ee", "#a855f7", "#fbbf24", "#34d399", "#f87171", "#60a5fa", "#f472b6", "#a3e635"];

  let seed = 3;
  let rng = new RNG(seed);
  let fSeed = 11;
  let obs = [];      // {x, y}
  let q = 4;
  let result = null; // {draws, mu, sd, poolIdx, batchIdx}

  /* A fixed wiggly multimodal true function built from a few random cosines. */
  function makeTrueF(s) {
    const r = new RNG(s);
    const terms = [];
    for (let i = 0; i < 4; i++) terms.push({ a: r.uniform(0.25, 0.8), w: r.uniform(4, 16), p: r.uniform(0, 6.28) });
    const trend = r.uniform(-0.6, 0.6);
    return (x) => {
      let v = trend * (x - 0.5);
      for (const t of terms) v += t.a * Math.cos(t.w * x + t.p);
      return v;
    };
  }
  let trueF = makeTrueF(fSeed);

  const grid = Array.from({ length: GRID_N }, (_, i) => [i / (GRID_N - 1)]);

  function initObs() {
    obs = [];
    const r = new RNG(fSeed + 100);
    for (let i = 0; i < 6; i++) {
      const x = (i + 0.5) / 6 + r.uniform(-0.06, 0.06);
      obs.push({ x: clamp(x, 0, 1), y: trueF(x) + r.normal() * 0.02 });
    }
  }

  function poolIndices() {
    // stratified pool over the grid: the discretization of the box
    const idx = [];
    const r = new RNG(seed * 7 + 5);
    for (let i = 0; i < POOL_N; i++) {
      const lo = (i / POOL_N) * (GRID_N - 1), hi = ((i + 1) / POOL_N) * (GRID_N - 1);
      idx.push(Math.round(clamp(lo + r.random() * (hi - lo), 0, GRID_N - 1)));
    }
    return idx;
  }

  function resample() {
    const X = obs.map((o) => [o.x]);
    const y = obs.map((o) => o.y);
    const gp = new MiniGP(X, y, 0.09, 3e-4);
    const { draws, mu, sd } = gp.sample(grid, q, rng);
    const poolIdx = poolIndices();
    const batchIdx = draws.map((dr) => {
      let bi = poolIdx[0];
      for (const pi of poolIdx) if (dr[pi] < dr[bi]) bi = pi;
      return bi;
    });
    result = { draws, mu, sd, poolIdx, batchIdx };
    draw();
  }

  function evaluateBatch() {
    if (!result) return;
    const seen = new Set();
    for (const bi of result.batchIdx) {
      if (seen.has(bi)) continue;
      seen.add(bi);
      const x = grid[bi][0];
      obs.push({ x, y: trueF(x) + rng.normal() * 0.02 });
    }
    resample();
  }

  function draw() {
    const w = cv.parentElement.clientWidth - 2;
    const h = Math.max(300, Math.min(400, w * 0.48));
    const ctx = setupCanvas(cv, w, h);
    ctx.clearRect(0, 0, w, h);
    const padL = 10, padR = 10, padT = 14, padB = 40;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    // y-range from band and truth
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < GRID_N; i++) {
      lo = Math.min(lo, result.mu[i] - 2.4 * result.sd[i]);
      hi = Math.max(hi, result.mu[i] + 2.4 * result.sd[i]);
      if (chkTrue.checked) { const tv = trueF(grid[i][0]); lo = Math.min(lo, tv); hi = Math.max(hi, tv); }
      for (const dr of result.draws) { lo = Math.min(lo, dr[i]); hi = Math.max(hi, dr[i]); }
    }
    for (const o of obs) { lo = Math.min(lo, o.y); hi = Math.max(hi, o.y); }
    const span = hi - lo || 1;
    lo -= span * 0.07; hi += span * 0.07;
    const X = (x) => padL + x * plotW;
    const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

    // 2-sigma band
    ctx.beginPath();
    for (let i = 0; i < GRID_N; i++) {
      const x = X(grid[i][0]), y = Y(result.mu[i] + 2 * result.sd[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = GRID_N - 1; i >= 0; i--) ctx.lineTo(X(grid[i][0]), Y(result.mu[i] - 2 * result.sd[i]));
    ctx.closePath();
    ctx.fillStyle = "rgba(59,130,246,0.13)";
    ctx.fill();

    // true function
    if (chkTrue.checked) {
      ctx.strokeStyle = "rgba(226,232,240,0.5)";
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < GRID_N; i++) {
        const x = X(grid[i][0]), y = Y(trueF(grid[i][0]));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // posterior mean
    ctx.strokeStyle = "rgba(148,163,184,0.9)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < GRID_N; i++) {
      const x = X(grid[i][0]), y = Y(result.mu[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // pool ticks
    ctx.strokeStyle = "rgba(100,116,139,0.7)";
    for (const pi of result.poolIdx) {
      const x = X(grid[pi][0]);
      ctx.beginPath();
      ctx.moveTo(x, padT + plotH + 8); ctx.lineTo(x, padT + plotH + 15);
      ctx.stroke();
    }
    ctx.fillStyle = "#64748b";
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--mono");
    ctx.textAlign = "left";
    ctx.fillText("candidate pool (Sobol discretization)", padL, padT + plotH + 32);

    // q posterior draws + their pool-argmins
    result.draws.forEach((dr, k) => {
      const col = DRAW_COLORS[k % DRAW_COLORS.length];
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.8;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < GRID_N; i++) {
        const x = X(grid[i][0]), y = Y(dr[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      const bi = result.batchIdx[k];
      const bx = X(grid[bi][0]), by = Y(dr[bi]);
      // batch pick: vertical line + diamond
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.4;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx, padT + plotH + 6); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(bx, by - 6); ctx.lineTo(bx + 5.5, by); ctx.lineTo(bx, by + 6); ctx.lineTo(bx - 5.5, by);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = "#0b1120";
      ctx.stroke();
    });

    // posterior-mean argmin (the greedy single pick), for contrast
    if (chkMean.checked) {
      let mi = result.poolIdx[0];
      for (const pi of result.poolIdx) if (result.mu[pi] < result.mu[mi]) mi = pi;
      const mx = X(grid[mi][0]), my = Y(result.mu[mi]);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(mx, my, 8, 0, 6.283); ctx.stroke();
      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "center";
      ctx.fillText("greedy mean pick", mx, my - 14);
    }

    // observations
    for (const o of obs) {
      ctx.fillStyle = "#e2e8f0";
      ctx.beginPath(); ctx.arc(X(o.x), Y(o.y), 4, 0, 6.283); ctx.fill();
      ctx.strokeStyle = "#0b1120";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "right";
    ctx.fillText(obs.length + " observations", padL + plotW, padT + 4);
  }

  /* wiring */
  qSlider.addEventListener("input", () => { q = parseInt(qSlider.value, 10); qVal.textContent = q; resample(); });
  chkTrue.addEventListener("change", draw);
  chkMean.addEventListener("change", draw);
  root.querySelector(".btn-redraw").addEventListener("click", resample);
  root.querySelector(".btn-eval").addEventListener("click", evaluateBatch);
  root.querySelector(".btn-newf").addEventListener("click", () => {
    fSeed = (fSeed * 1103515245 + 12345) % 2147483647;
    trueF = makeTrueF(fSeed);
    initObs();
    resample();
  });
  cv.addEventListener("click", (e) => {
    const rect = cv.getBoundingClientRect();
    const padL = 10, plotW = rect.width - 20;
    const x = clamp((e.clientX - rect.left - padL) / plotW, 0, 1);
    obs.push({ x, y: trueF(x) + rng.normal() * 0.02 });
    resample();
  });
  onResize(root, draw);

  initObs();
  resample();
})();
