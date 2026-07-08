/* Mask distribution explorer.
 *
 * Left panel : one draw of the move-weight vector alpha over d=25 coordinates, as bars.
 * Right panel: the marginal law of a single weight (histogram over many draws), showing the
 *              polarization of the soft Beta mask and its two limits (Proposition 2).
 * Bottom     : the sparsity anneal rho(v) with the derived endpoints, d slider.
 *
 * Sampling mirrors src mask laws exactly:
 *   dense: alpha_j = 1
 *   hard : alpha_j ~ Bernoulli(rho), at least one active
 *   soft : m_j ~ Beta(rho c0, (1-rho) c0), alpha = clip(m * rho d / sum(m), 0, 1)
 */
"use strict";

(function () {
  const root = document.getElementById("demo-mask");
  if (!root) return;

  const DIM = 25;
  const state = { mode: "soft", rho: 0.35, c0: 0.4, seed: 7 };
  let rng = new RNG(state.seed);

  const barsCv = root.querySelector(".mask-bars");
  const histCv = root.querySelector(".mask-hist");
  const schedCv = root.querySelector(".mask-sched");
  const rhoSlider = root.querySelector(".ctl-rho");
  const c0Slider = root.querySelector(".ctl-c0");
  const rhoVal = root.querySelector(".ctl-rho-val");
  const c0Val = root.querySelector(".ctl-c0-val");
  const c0Wrap = root.querySelector(".ctl-c0-wrap");
  const massEl = root.querySelector(".mask-mass");
  const dSlider = root.querySelector(".ctl-d");
  const dVal = root.querySelector(".ctl-d-val");

  function sampleMask() {
    const a = new Float64Array(DIM);
    if (state.mode === "dense") { a.fill(1); return a; }
    if (state.mode === "hard") {
      const p = clamp(state.rho, 1 / DIM, 1);
      let any = false;
      for (let j = 0; j < DIM; j++) { a[j] = rng.random() < p ? 1 : 0; any = any || a[j] > 0; }
      if (!any) a[rng.int(DIM)] = 1;
      return a;
    }
    const c0 = Math.max(state.c0, 1e-3);
    const al = Math.max(state.rho * c0, 1e-3), be = Math.max((1 - state.rho) * c0, 1e-3);
    let sum = 0;
    for (let j = 0; j < DIM; j++) { a[j] = rng.beta(al, be); sum += a[j]; }
    const target = state.rho * DIM;
    for (let j = 0; j < DIM; j++) a[j] = clamp((a[j] * target) / Math.max(sum, 1e-12), 0, 1);
    return a;
  }

  let lastMask = null;

  function drawBars() {
    const w = barsCv.parentElement.clientWidth - 2;
    const h = 190;
    const ctx = setupCanvas(barsCv, w, h);
    ctx.clearRect(0, 0, w, h);
    const padL = 8, padB = 22, padT = 12;
    const plotW = w - padL - 6, plotH = h - padT - padB;
    const bw = plotW / DIM;
    // guide line at alpha = 1
    ctx.strokeStyle = "rgba(148,163,184,0.25)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL + plotW, padT);
    ctx.stroke();
    ctx.setLineDash([]);
    const a = lastMask;
    let mass = 0;
    for (let j = 0; j < DIM; j++) {
      mass += a[j];
      const bh = a[j] * plotH;
      const x = padL + j * bw + 1.5;
      const grd = ctx.createLinearGradient(0, padT + plotH - bh, 0, padT + plotH);
      grd.addColorStop(0, a[j] > 0.98 ? "#67e8f9" : "#22d3ee");
      grd.addColorStop(1, "rgba(34,211,238,0.25)");
      ctx.fillStyle = a[j] < 0.02 ? "rgba(100,116,139,0.35)" : grd;
      ctx.fillRect(x, padT + plotH - Math.max(bh, 2), bw - 3, Math.max(bh, 2));
    }
    ctx.fillStyle = "#64748b";
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--mono");
    ctx.textAlign = "left";
    ctx.fillText("coordinate j = 1 … " + DIM, padL, h - 6);
    ctx.textAlign = "right";
    ctx.fillText("α=1", padL + plotW, padT - 3);
    massEl.textContent = "Σα = " + mass.toFixed(2) + "  (target ρd = " + (state.rho * DIM).toFixed(2) + ")";
  }

  function drawHist() {
    const w = histCv.parentElement.clientWidth - 2;
    const h = 190;
    const ctx = setupCanvas(histCv, w, h);
    ctx.clearRect(0, 0, w, h);
    const padL = 8, padB = 22, padT = 12;
    const plotW = w - padL - 6, plotH = h - padT - padB;
    const NB = 40, counts = new Float64Array(NB);
    const hr = new RNG(1234);
    const NS = 4000;
    const saved = rng; rng = hr; // reuse sampleMask with a fixed rng for a stable histogram
    for (let s = 0; s < NS / DIM; s++) {
      const a = sampleMask();
      for (let j = 0; j < DIM; j++) counts[Math.min(NB - 1, Math.floor(a[j] * NB))]++;
    }
    rng = saved;
    const maxC = Math.max(...counts);
    for (let b = 0; b < NB; b++) {
      const bh = (counts[b] / maxC) * plotH;
      const x = padL + (b / NB) * plotW;
      ctx.fillStyle = "rgba(168,85,247,0.75)";
      ctx.fillRect(x + 0.5, padT + plotH - bh, plotW / NB - 1, bh);
    }
    ctx.fillStyle = "#64748b";
    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--mono");
    ctx.textAlign = "left"; ctx.fillText("0", padL, h - 6);
    ctx.textAlign = "center"; ctx.fillText("marginal law of a single weight αⱼ", padL + plotW / 2, h - 6);
    ctx.textAlign = "right"; ctx.fillText("1", padL + plotW, h - 6);
  }

  function drawSched() {
    const w = schedCv.parentElement.clientWidth - 2;
    const h = 180;
    const ctx = setupCanvas(schedCv, w, h);
    ctx.clearRect(0, 0, w, h);
    const d = parseInt(dSlider.value, 10);
    const B = 300, q = 4, batches = B / q;
    const rhoInit = clamp(3 / Math.sqrt(d), 0.45, 0.7);
    const rhoMin = clamp(1 / Math.sqrt(d), 0.1, 0.25);
    const tau = Math.max(8, 0.25 * batches);
    const padL = 44, padB = 26, padT = 12, padR = 10;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const y2px = (r) => padT + (1 - r / 0.8) * plotH;
    // grid
    ctx.strokeStyle = "rgba(51,65,85,0.6)";
    ctx.fillStyle = "#64748b";
    ctx.font = "10.5px " + getComputedStyle(document.body).getPropertyValue("--mono");
    ctx.textAlign = "right";
    for (const r of [0.2, 0.4, 0.6, 0.8]) {
      ctx.beginPath(); ctx.moveTo(padL, y2px(r)); ctx.lineTo(padL + plotW, y2px(r)); ctx.stroke();
      ctx.fillText(r.toFixed(1), padL - 6, y2px(r) + 3.5);
    }
    // endpoints
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(34,211,238,0.5)";
    ctx.beginPath(); ctx.moveTo(padL, y2px(rhoInit)); ctx.lineTo(padL + plotW, y2px(rhoInit)); ctx.stroke();
    ctx.strokeStyle = "rgba(236,72,153,0.5)";
    ctx.beginPath(); ctx.moveTo(padL, y2px(rhoMin)); ctx.lineTo(padL + plotW, y2px(rhoMin)); ctx.stroke();
    ctx.setLineDash([]);
    // curve
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let v = 0; v <= batches; v++) {
      const r = rhoMin + (rhoInit - rhoMin) * Math.exp(-v / tau);
      const x = padL + (v / batches) * plotW;
      if (v === 0) ctx.moveTo(x, y2px(r)); else ctx.lineTo(x, y2px(r));
    }
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.textAlign = "left";
    ctx.fillStyle = "#67e8f9";
    ctx.fillText("ρ_init = clamp(3/√d) = " + rhoInit.toFixed(2), padL + 6, y2px(rhoInit) - 5);
    ctx.fillStyle = "#f9a8d4";
    ctx.fillText("ρ_min = clamp(1/√d) = " + rhoMin.toFixed(2), padL + plotW * 0.42, y2px(rhoMin) - 5);
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "center";
    ctx.fillText("region visits v  (τ = 0.25·B/q = " + tau.toFixed(0) + ")", padL + plotW / 2, h - 8);
  }

  function redraw() {
    rhoVal.textContent = state.rho.toFixed(2);
    c0Val.textContent = state.c0.toFixed(2);
    c0Wrap.style.opacity = state.mode === "soft" ? 1 : 0.35;
    lastMask = sampleMask();
    drawBars();
    drawHist();
  }

  /* wiring */
  root.querySelectorAll(".seg-mode button").forEach((b) => {
    b.addEventListener("click", () => {
      root.querySelectorAll(".seg-mode button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.mode = b.dataset.mode;
      redraw();
    });
  });
  rhoSlider.addEventListener("input", () => { state.rho = parseFloat(rhoSlider.value); redraw(); });
  c0Slider.addEventListener("input", () => { state.c0 = parseFloat(c0Slider.value); redraw(); });
  root.querySelector(".btn-roll").addEventListener("click", () => { lastMask = sampleMask(); drawBars(); });
  root.querySelector(".btn-limit-hard").addEventListener("click", () => {
    state.mode = "soft"; state.c0 = 0.05;
    c0Slider.value = "0.05";
    root.querySelectorAll(".seg-mode button").forEach((x) => x.classList.toggle("active", x.dataset.mode === "soft"));
    redraw();
  });
  root.querySelector(".btn-limit-dense").addEventListener("click", () => {
    state.mode = "soft"; state.rho = 1.0;
    rhoSlider.value = "1";
    root.querySelectorAll(".seg-mode button").forEach((x) => x.classList.toggle("active", x.dataset.mode === "soft"));
    redraw();
  });
  dSlider.addEventListener("input", () => { dVal.textContent = dSlider.value; drawSched(); });

  onResize(root, () => { drawBars(); drawHist(); drawSched(); });
  redraw();
  drawSched();
})();
