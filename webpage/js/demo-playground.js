/* 2D many-basin playground.
 *
 * Landscape2D is a faithful 2-D port of the many-basin benchmark family: many broad local basins with
 * near-identical visible value, a few of which hide a much deeper narrow core; the start point sits in
 * an ordinary basin. MiniS3Turbo is a faithful miniature of the optimizer: trust regions with batch
 * Thompson sampling from a local GP, coordinate masks, the headline scout strategies (none / random /
 * switch / reactive, the last a k-controlled dose of evidence-gated escape), promotion gates,
 * importance-based pruning, and the bounded focus burst, with every constant computed by the same
 * derivations as the real implementation (specialized to d=2, q=4).
 */
"use strict";

/* ================= landscape ================= */
class Landscape2D {
  constructor(seed, nBasins = 18, nGlobal = 2) {
    const r = new RNG(seed * 977 + 13);
    this.dim = 2;
    this.outerSigma = 0.18;
    this.innerSigma = 0.045;
    this.innerDepth = 0.14;
    const centers = [];
    for (let i = 0; i < nBasins; i++) centers.push([0.05 + 0.9 * r.random(), 0.05 + 0.9 * r.random()]);
    // pick global-core basins spread apart
    const globalIdx = [r.int(nBasins)];
    while (globalIdx.length < nGlobal) {
      let best = -1, bestD = -1;
      for (let i = 0; i < nBasins; i++) {
        if (globalIdx.includes(i)) continue;
        let dmin = Infinity;
        for (const g of globalIdx) {
          const dx = centers[i][0] - centers[g][0], dy = centers[i][1] - centers[g][1];
          dmin = Math.min(dmin, (dx * dx + dy * dy) / 2);
        }
        if (dmin > bestD) { bestD = dmin; best = i; }
      }
      globalIdx.push(best);
    }
    const gset = new Set(globalIdx);
    this.basins = centers.map((c, i) => ({
      center: c,
      level: 0.09 + 0.035 * r.random(),
      so: this.outerSigma * r.uniform(0.85, 1.15),
      si: this.innerSigma * r.uniform(0.85, 1.15),
      global: gset.has(i),
    }));
    this.globalIdx = globalIdx;
    // start near a non-global basin
    const cand = [];
    for (let i = 0; i < nBasins; i++) if (!gset.has(i)) cand.push(i);
    const x0i = cand[r.int(cand.length)];
    this.x0 = [
      clamp(centers[x0i][0] + r.normal() * this.outerSigma * 0.35, 0, 1),
      clamp(centers[x0i][1] + r.normal() * this.outerSigma * 0.35, 0, 1),
    ];
    this.optimum = Math.min(...this.basins.filter((b) => b.global).map((b) => b.level - this.innerDepth));
  }
  value(u) {
    let vmin = Infinity;
    for (const b of this.basins) {
      const dx = u[0] - b.center[0], dy = u[1] - b.center[1];
      const rms2 = (dx * dx + dy * dy) / 2;
      let v = b.level + rms2 / (2 * b.so * b.so);
      if (b.global) v -= this.innerDepth * Math.exp(-rms2 / (2 * b.si * b.si));
      vmin = Math.min(vmin, v);
    }
    return vmin - this.optimum;
  }
  nearestBasin(u) {
    let bi = 0, bd = Infinity;
    for (let i = 0; i < this.basins.length; i++) {
      const b = this.basins[i];
      const dx = u[0] - b.center[0], dy = u[1] - b.center[1];
      const rms = Math.sqrt((dx * dx + dy * dy) / 2);
      if (rms < bd) { bd = rms; bi = i; }
    }
    return { idx: bi, rms: bd };
  }
  /* Render the landscape once into an offscreen canvas (low value = bright). */
  heatmap(px = 220) {
    const off = document.createElement("canvas");
    off.width = px; off.height = px;
    const ctx = off.getContext("2d");
    const img = ctx.createImageData(px, px);
    const vals = new Float64Array(px * px);
    let vmin = Infinity, vmax = -Infinity;
    for (let iy = 0; iy < px; iy++)
      for (let ix = 0; ix < px; ix++) {
        const v = this.value([(ix + 0.5) / px, (iy + 0.5) / px]);
        vals[iy * px + ix] = v;
        vmin = Math.min(vmin, v); vmax = Math.max(vmax, v);
      }
    const cap = Math.min(vmax, 1.1); // squash the dull far field
    for (let i = 0; i < px * px; i++) {
      const t = Math.pow(clamp((vals[i] - vmin) / (cap - vmin), 0, 1), 0.62);
      const [r, g, b] = lerpColorStops(LANDSCAPE_STOPS, t);
      img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return off;
  }
}

/* ================= miniature optimizer ================= */
const PG_Q = 4, PG_BUDGET = 300;

class MiniS3Turbo {
  constructor(landscape, { strategy = "switch", mask = "hard", seed = 0, escapeK = 0.75 } = {}) {
    this.f = landscape;
    this.strategy = strategy;
    this.mask = mask;
    this.escapeK = escapeK > 0 ? escapeK : 0.75;
    this.rng = new RNG(seed * 6151 + 21);
    this.dim = 2;
    this.U = []; this.y = [];
    this.regions = [];
    this.mined = new Set();
    this.askCount = 0;
    this.globalFail = 0;
    this.improvedThisTell = false;
    this.lastRidx = []; this.lastKind = [];
    this.focusIdx = null; this.focusLeft = 0;
    this.escapeValue = 0; // reactive escape signal E in [0,1]
    this.events = [];
    this.curve = [];
    this.basinsSeen = new Set();
    this.done = false;
    this.derive();
    // observe x0 first (the planted start)
    this.observe([this.f.x0.slice()], [-1], ["x0"]);
  }

  /* ---- derived constants (the section-5 rules, at d=2, q=4, B=300) ---- */
  derive() {
    const d = this.dim, q = PG_Q, batches = PG_BUDGET / q;
    const maskedOrScout = this.mask !== "dense" || this.strategy !== "none";
    this.nInit = Math.max(2 * q, 8);
    this.pool = 96;
    this.maxData = 64;
    this.succTol = 3;
    this.rhoInit = clamp(3 / Math.sqrt(d), 0.45, 0.7);
    this.rhoMin = clamp(1 / Math.sqrt(d), 0.1, 0.25);
    this.rhoTau = Math.max(8, 0.25 * batches);
    const dEff = this.mask !== "dense" ? this.rhoInit * d : d;
    this.failTol = Math.max(4, Math.ceil(dEff / q));
    this.lInit = 0.35;
    this.lMin = 0.02;
    this.lMax = maskedOrScout ? 0.9 : 1.6;
    this.novelDist = 0.6 / Math.sqrt(6);
    this.maxRegions = (this.strategy === "none" || this.strategy === "random") ? 1 : 1 + Math.min(3, Math.floor(batches / 25));
    this.scoutPeriod = this.strategy === "none" ? 0 : Math.max(1, Math.ceil(1 / (q * 0.06)));
    this.stagnationAfter = Math.max(4, Math.ceil(1.5 * this.failTol));
    this.candRadius = Math.max(this.lInit * 0.9, 1.5 * this.novelDist);
    this.candWarmup = Math.max(4, Math.ceil(1.5 * q));
    this.focusSlots = Math.max(1, q - 1);
    this.focusBatches = Math.max(2, Math.ceil(0.15 * batches));
    this.beta = 0.08; this.eta = 0.06; this.cAccept = 0.25;
    // reactive escape: base rate rho_0 = 1/(escape_k * sqrt d), arm gate at half the base, and a
    // memory decay for the escape value (§4). In the real optimizer d is 20-30 so rho_0 is small; at
    // d=2 that formula saturates, so the miniature rescales rho_0 into a visible band across the
    // recommended escape_k range while preserving the monotone "small k scouts more" behavior.
    const rho0 = 1 / (this.escapeK * Math.sqrt(d));
    this.escapeBase = clamp(0.06 + 0.5 * (rho0 - 0.5), 0.03, 0.6);
    this.armGate = Math.max(0.12, 0.5 * this.escapeBase);
    this.escapeDecay = 0.8;
  }

  Sy() { return robustScale(this.y); }
  best() {
    let bi = 0;
    for (let i = 1; i < this.y.length; i++) if (this.y[i] < this.y[bi]) bi = i;
    return { u: this.U[bi], y: this.y[bi] };
  }
  rho(r) { return this.rhoMin + (this.rhoInit - this.rhoMin) * Math.exp(-r.visits / this.rhoTau); }

  /* ---- masks ---- */
  sampleMask(rho) {
    const d = this.dim, a = new Float64Array(d);
    if (this.mask === "dense") { a.fill(1); return a; }
    if (this.mask === "hard") {
      const p = clamp(rho, 1 / d, 1);
      let any = false;
      for (let j = 0; j < d; j++) { a[j] = this.rng.random() < p ? 1 : 0; any = any || a[j] > 0; }
      if (!any) a[this.rng.int(d)] = 1;
      return a;
    }
    const c0 = 0.4;
    let sum = 0;
    for (let j = 0; j < d; j++) { a[j] = this.rng.beta(Math.max(rho * c0, 1e-3), Math.max((1 - rho) * c0, 1e-3)); sum += a[j]; }
    for (let j = 0; j < d; j++) a[j] = clamp((a[j] * rho * d) / Math.max(sum, 1e-12), 0, 1);
    return a;
  }
  applyMask(region, pts, dense) {
    const rho = this.rho(region);
    return pts.map((p) => {
      const a = dense ? [1, 1] : this.sampleMask(rho);
      return [
        clamp(region.center[0] + a[0] * (p[0] - region.center[0]), 0, 1),
        clamp(region.center[1] + a[1] * (p[1] - region.center[1]), 0, 1),
      ];
    });
  }

  /* ---- local Thompson step inside one region ---- */
  trainNear(center, radius, localOnly) {
    const n = this.y.length;
    if (radius !== null && localOnly) {
      const loc = [];
      for (let i = 0; i < n; i++) {
        const dx = this.U[i][0] - center[0], dy = this.U[i][1] - center[1];
        if (Math.sqrt((dx * dx + dy * dy) / 2) <= radius) loc.push(i);
      }
      if (loc.length >= 4) {
        loc.sort((a, b) => this.y[a] - this.y[b]);
        const idx = loc.slice(0, this.maxData);
        return { X: idx.map((i) => this.U[i]), Y: idx.map((i) => this.y[i]) };
      }
    }
    if (n <= this.maxData) return { X: this.U, Y: this.y };
    const d2 = this.U.map((u, i) => {
      const dx = u[0] - center[0], dy = u[1] - center[1];
      return [dx * dx + dy * dy, i];
    }).sort((a, b) => a[0] - b[0]);
    const near = d2.slice(0, Math.floor(this.maxData / 2)).map((x) => x[1]);
    const byY = this.y.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const best = byY.slice(0, this.maxData - near.length).map((x) => x[1]);
    const idx = Array.from(new Set([...near, ...best])).slice(0, this.maxData);
    return { X: idx.map((i) => this.U[i]), Y: idx.map((i) => this.y[i]) };
  }

  localTS(region, n, dense = false) {
    if (n <= 0) return [];
    const isCand = region.kind === "candidate";
    if (isCand && region.visits <= this.candWarmup) {
      const raw = [];
      for (let k = 0; k < n; k++)
        raw.push([
          clamp(region.center[0] + this.rng.uniform(-0.5, 0.5) * region.radius, 0, 1),
          clamp(region.center[1] + this.rng.uniform(-0.5, 0.5) * region.radius, 0, 1),
        ]);
      return this.applyMask(region, raw, dense);
    }
    const { X, Y } = this.trainNear(region.center, isCand ? region.radius : null, isCand);
    let gp = null;
    try { gp = new MiniGP(X, Y, Math.max(0.06, region.radius * 0.4), 1e-4); } catch (e) { gp = null; }
    const lo = [clamp(region.center[0] - region.radius / 2, 0, 1), clamp(region.center[1] - region.radius / 2, 0, 1)];
    const hi = [clamp(region.center[0] + region.radius / 2, 0, 1), clamp(region.center[1] + region.radius / 2, 0, 1)];
    const raw = [];
    for (let k = 0; k < this.pool; k++)
      raw.push([this.rng.uniform(lo[0], hi[0]), this.rng.uniform(lo[1], hi[1])]);
    const box = this.applyMask(region, raw, dense);
    if (!gp) return box.slice(0, n);
    const { draws } = gp.sample(box, n, this.rng);
    return draws.map((dr) => {
      let bi = 0;
      for (let i = 1; i < box.length; i++) if (dr[i] < dr[bi]) bi = i;
      return box[bi];
    });
  }

  farthestPoint() {
    let best = null, bestD = -1;
    for (let k = 0; k < 512; k++) {
      const c = [this.rng.random(), this.rng.random()];
      let dmin = Infinity;
      for (const u of this.U) {
        const dx = c[0] - u[0], dy = c[1] - u[1];
        dmin = Math.min(dmin, dx * dx + dy * dy);
      }
      if (dmin > bestD) { bestD = dmin; best = c; }
    }
    return best;
  }

  /* ---- regions ---- */
  ensureMain() {
    if (this.regions.length || !this.y.length) return;
    const b = this.best();
    this.regions.push({ center: b.u.slice(), radius: this.lInit, kind: "main", bestY: b.y, bestU: b.u.slice(), visits: 0, succ: 0, fail: 0, warmup: 0 });
  }
  mainIndex() {
    this.ensureMain();
    for (let i = 0; i < this.regions.length; i++) if (this.regions[i].kind === "main") return i;
    return 0;
  }
  importance(r) {
    const s = this.Sy();
    const unc = Math.sqrt(Math.log(this.y.length + 2) / (r.visits + 1));
    let nov = 0;
    for (const o of this.regions) {
      if (o === r) continue;
      const dx = r.center[0] - o.center[0], dy = r.center[1] - o.center[1];
      nov = nov === 0 ? Math.hypot(dx, dy) / Math.sqrt(2) : Math.min(nov, Math.hypot(dx, dy) / Math.sqrt(2));
    }
    return r.bestY - this.beta * s * unc - this.eta * s * nov;
  }
  candidateIndex() {
    let bi = null, bv = Infinity;
    for (let i = 0; i < this.regions.length; i++) {
      if (this.regions[i].kind !== "candidate") continue;
      const v = this.importance(this.regions[i]);
      if (v < bv) { bv = v; bi = i; }
    }
    return bi;
  }
  rankRegions() {
    this.ensureMain();
    const warm = [], cold = [];
    this.regions.forEach((r, i) => (r.warmup > 0 ? warm : cold).push(i));
    warm.sort((a, b) => this.regions[b].warmup - this.regions[a].warmup || this.regions[a].bestY - this.regions[b].bestY);
    cold.sort((a, b) => this.importance(this.regions[a]) - this.importance(this.regions[b]));
    return warm.concat(cold);
  }
  novel(u) {
    for (const r of this.regions) {
      const dx = u[0] - r.center[0], dy = u[1] - r.center[1];
      if (Math.hypot(dx, dy) / Math.sqrt(2) < this.novelDist) return false;
    }
    return true;
  }
  accept(yv, q, marginK) {
    if (this.y.length < this.nInit) return false;
    const thr = quantile(this.y, q);
    return yv <= thr || yv <= Math.min(...this.y) + marginK * this.Sy();
  }
  focusGate(yv) {
    if (this.y.length < this.nInit) return false;
    return yv <= quantile(this.y, 0.35) || yv <= Math.min(...this.y) + this.cAccept * this.Sy();
  }
  addCandidate(u, yv) {
    if (this.maxRegions <= 1) return false;
    const key = u.map((x) => x.toFixed(5)).join(",");
    if (this.mined.has(key) || !this.novel(u)) return false;
    this.regions.push({ center: u.slice(), radius: this.candRadius, kind: "candidate", bestY: yv, bestU: u.slice(), visits: 1, succ: 0, fail: 0, warmup: this.candWarmup });
    this.mined.add(key);
    this.events.push({ t: "cand", msg: `ask ${this.askCount}: promoted candidate at (${u[0].toFixed(2)}, ${u[1].toFixed(2)}) f=${yv.toFixed(3)}` });
    this.dropExcess();
    return true;
  }
  dropExcess() {
    while (this.regions.length > this.maxRegions) {
      const prot = new Set();
      this.regions.forEach((r, i) => { if (r.kind === "main" || r.warmup > 0) prot.add(i); });
      const order = this.regions.map((r, i) => [this.importance(r), i]).sort((a, b) => b[0] - a[0]);
      let drop = null;
      for (const [, i] of order) if (!prot.has(i)) { drop = i; break; }
      if (drop === null) for (const [, i] of order) if (this.regions[i].kind !== "main") { drop = i; break; }
      if (drop === null) drop = order[0][1];
      this.regions.splice(drop, 1);
      if (this.focusIdx !== null) {
        if (drop === this.focusIdx) { this.focusIdx = null; this.focusLeft = 0; }
        else if (drop < this.focusIdx) this.focusIdx -= 1;
      }
    }
  }
  validFocus() {
    return (this.strategy === "switch" || this.strategy === "reactive") &&
      this.focusIdx !== null && this.focusLeft > 0 &&
      this.focusIdx >= 0 && this.focusIdx < this.regions.length &&
      this.regions[this.focusIdx].kind === "candidate";
  }

  /* ---- batch helpers ---- */
  localOne(ridx, dense = false) {
    if (ridx === null) return { u: [this.rng.random(), this.rng.random()], ri: -1, k: "sobol" };
    return { u: this.localTS(this.regions[ridx], 1, dense)[0], ri: ridx, k: dense ? "focus" : "local" };
  }
  mainBatch(n) {
    const main = this.mainIndex(), out = [];
    for (let i = 0; i < Math.max(0, n); i++) out.push(this.localOne(main));
    return out;
  }
  localBatch(n) {
    const ranked = this.rankRegions(), out = [];
    for (let s = 0; s < Math.max(0, n); s++) out.push(this.localOne(ranked.length ? ranked[s % ranked.length] : null));
    return out;
  }
  wantScout() {
    if (this.strategy === "none" || this.scoutPeriod <= 0) return false;
    if (this.strategy === "reactive") {
      // always spend the derived base rate rho_0, reallocated up by the escape value E (§4)
      if (this.validFocus()) return true;
      const rate = this.escapeBase + (1 - this.escapeBase) * this.escapeValue;
      return this.rng.random() < rate;
    }
    const periodic = this.askCount % this.scoutPeriod === 0;
    if (this.strategy === "random") return periodic;
    return periodic || this.globalFail >= this.stagnationAfter;
  }

  /* ---- strategy select ---- */
  select(q) {
    if (this.strategy === "none") return this.localBatch(q);
    if (this.strategy === "random") {
      const nS = this.wantScout() ? 1 : 0;
      const picks = this.localBatch(q - nS);
      for (let i = 0; i < nS; i++) {
        picks.push({ u: this.farthestPoint(), ri: -1, k: "scout" });
        this.events.push({ t: "scout", msg: `ask ${this.askCount}: far probe fired` });
      }
      return picks;
    }
    // sidecar / switch / reactive: protected main + a side slot, or a focus burst when armed
    if ((this.strategy === "switch" || this.strategy === "reactive") && this.validFocus()) {
      const picks = [];
      const f = this.focusIdx;
      for (let i = 0; i < Math.min(this.focusSlots, Math.max(1, q - 1)); i++) picks.push(this.localOne(f, true));
      const main = this.mainIndex();
      while (picks.length < q) picks.push(this.localOne(main));
      this.focusLeft -= 1;
      if (this.focusLeft === 0) this.events.push({ t: "focus", msg: `ask ${this.askCount}: focus episode ended` });
      return picks;
    }
    const nSide = Math.min(this.wantScout() ? 1 : 0, Math.max(0, q - 1));
    const picks = this.mainBatch(q - nSide);
    for (let i = 0; i < nSide; i++) {
      const ci = this.candidateIndex();
      if (ci !== null && this.rng.random() < 0.75) {
        picks.push(this.localOne(ci));
      } else {
        picks.push({ u: this.farthestPoint(), ri: -1, k: "scout" });
        this.events.push({ t: "scout", msg: `ask ${this.askCount}: far probe fired` });
      }
    }
    return picks;
  }

  armFocus(yv) {
    if (!this.focusGate(yv)) return;
    let bi = null, bv = Infinity;
    this.regions.forEach((r, i) => {
      if (r.kind === "candidate" && r.bestY < bv) { bv = r.bestY; bi = i; }
    });
    if (bi !== null) {
      this.focusIdx = bi; this.focusLeft = this.focusBatches;
      this.events.push({ t: "focus", msg: `ask ${this.askCount}: FOCUS BURST on candidate (${this.regions[bi].center[0].toFixed(2)}, ${this.regions[bi].center[1].toFixed(2)}) for ${this.focusBatches} asks` });
    }
  }

  onTell(ub, yb, kinds) {
    const aq = this.strategy === "random" ? 0.65 : 0.55;
    const mk = this.strategy === "random" ? 2.0 : 1.0;
    for (let i = 0; i < ub.length; i++) {
      if (kinds[i] === "scout" && this.accept(yb[i], aq, mk)) {
        const added = this.addCandidate(ub[i], yb[i]);
        // switch commits unconditionally; reactive commits only once evidence clears the gate
        if (added && this.strategy === "switch") this.armFocus(yb[i]);
        if (added && this.strategy === "reactive" && this.escapeValue > this.armGate) this.armFocus(yb[i]);
      }
    }
    if (this.strategy === "reactive") this.updateEscapeValue();
    const isMulti = this.strategy === "switch" || this.strategy === "reactive";
    if (isMulti) {
      // reactive only forces a candidate/arm once the escape value has cleared the gate
      const armOk = this.strategy === "switch" || this.escapeValue > this.armGate;
      if (armOk && this.focusIdx === null && this.wantScout() && this.regions.length < this.maxRegions) {
        const order = this.y.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
        for (const [v, i] of order) {
          if (this.focusGate(v) && this.addCandidate(this.U[i], v)) { this.armFocus(v); break; }
        }
      }
      if (this.focusIdx !== null && this.improvedThisTell)
        this.focusLeft = Math.max(this.focusLeft, Math.floor(this.focusBatches / 2));
    }
  }

  /* Reactive escape value E in [0,1]: rises when a recently-worked candidate region is spatially
   * distinct from the incumbent (by the derived novelty radius) AND competitive with the main; decays
   * otherwise. Never reads whether a far basin exists — only the outcome of candidates already planted. */
  updateEscapeValue() {
    const main = this.regions[this.mainIndex()];
    let signal = 0;
    const sy = this.Sy();
    // "competitive" = near the good end of the archive, not necessarily beating the incumbent (a far
    // candidate rarely does at once). A distinct candidate that clears the value gate is evidence.
    const good = this.y.length >= this.nInit ? quantile(this.y, 0.4) : Infinity;
    for (const r of this.regions) {
      if (r.kind !== "candidate") continue;
      const dx = r.center[0] - main.center[0], dy = r.center[1] - main.center[1];
      const dist = Math.hypot(dx, dy) / Math.sqrt(2);
      const distinct = dist >= this.novelDist;
      const competitive = r.bestY <= good || r.bestY <= main.bestY + this.cAccept * sy;
      if (distinct && competitive) signal = 1;
    }
    this.escapeValue = clamp(this.escapeDecay * this.escapeValue + (1 - this.escapeDecay) * signal, 0, 1);
  }

  /* ---- observe / region box update ---- */
  observe(pts, ridx, kinds) {
    const yv = pts.map((p) => this.f.value(p));
    const before = this.y.length ? Math.min(...this.y) : Infinity;
    for (let i = 0; i < pts.length; i++) {
      this.U.push(pts[i]); this.y.push(yv[i]);
      this.curve.push(Math.min(this.curve.length ? this.curve[this.curve.length - 1] : Infinity, yv[i]));
      const nb = this.f.nearestBasin(pts[i]);
      if (nb.rms <= 0.2) this.basinsSeen.add(nb.idx);
      this.ptsLog.push({ u: pts[i], y: yv[i], kind: kinds[i], age: this.askCount });
    }
    this.ensureMain();
    // per-batch trust-region update on non-scout slots, grouped by region
    const touched = new Map();
    for (let i = 0; i < pts.length; i++) {
      const ri = ridx[i];
      if (ri >= 0 && ri < this.regions.length && kinds[i] !== "scout") {
        const rec = touched.get(ri);
        if (!rec) touched.set(ri, { u: pts[i], y: yv[i], count: 1 });
        else { rec.count += 1; if (yv[i] < rec.y) { rec.u = pts[i]; rec.y = yv[i]; } }
      }
    }
    for (const [ri, rec] of touched) this.updateRegion(this.regions[ri], rec.u, rec.y, rec.count);
    const after = Math.min(...this.y);
    this.improvedThisTell = after < before - 1e-9;
    if (this.improvedThisTell && isFinite(before))
      this.events.push({ t: "best", msg: `ask ${this.askCount}: new best f = ${after.toFixed(4)}` });
    this.globalFail = this.improvedThisTell ? 0 : this.globalFail + 1;
    this.onTell(pts, yv, kinds);
    this.dropExcess();
  }
  updateRegion(r, u, yv, count) {
    r.visits += count;
    r.warmup = Math.max(0, r.warmup - count);
    if (yv < r.bestY - 1e-9) {
      r.bestY = yv; r.bestU = u.slice(); r.center = u.slice();
      r.succ += 1; r.fail = 0;
      if (r.succ >= this.succTol) {
        r.radius = Math.min(this.lMax, r.radius * 1.5);
        r.succ = 0;
        if (r.kind === "main") this.events.push({ t: "box", msg: `ask ${this.askCount}: main box grew to ℓ=${r.radius.toFixed(2)}` });
      }
    } else {
      r.fail += 1; r.succ = 0;
      if (r.fail >= this.failTol) {
        r.radius = Math.max(this.lMin, r.radius / 2);
        r.fail = 0;
        if (r.kind === "main") this.events.push({ t: "box", msg: `ask ${this.askCount}: main box shrank to ℓ=${r.radius.toFixed(2)}` });
      }
    }
  }

  /* ---- one ask/tell step ---- */
  get ptsLog() { return (this._pts ||= []); }
  step() {
    if (this.done) return;
    const q = Math.min(PG_Q, PG_BUDGET - this.y.length);
    if (q <= 0) { this.done = true; return; }
    if (this.y.length < this.nInit) {
      const pts = [];
      for (let i = 0; i < q; i++) pts.push([this.rng.random(), this.rng.random()]);
      this.observe(pts, pts.map(() => -1), pts.map(() => "init"));
      return;
    }
    this.ensureMain();
    this.askCount += 1;
    const picks = this.select(q);
    this.observe(picks.map((p) => p.u), picks.map((p) => p.ri), picks.map((p) => p.k));
    if (this.y.length >= PG_BUDGET) this.done = true;
  }
}

/* ================= UI harness ================= */
(function () {
  const root = document.getElementById("demo-playground");
  if (!root) return;

  const STRATS = ["none", "random", "switch", "reactive"];
  const CMP_COLORS = ["#94a3b8", "#fbbf24", "#c084fc", "#22d3ee"]; // sparkline + cell-label colors, by strategy
  const KIND_COLOR = { x0: "#ffffff", init: "#64748b", sobol: "#64748b", local: "#22d3ee", scout: "#fbbf24", focus: "#c084fc" };
  const state = { strategy: "reactive", mask: "hard", seed: 10, speed: 6, escapeK: 0.75, playing: false, compare: false };

  let land = null, heat = null, opt = null, opts = null; // opts = 4 instances in compare mode
  let timer = null;

  const singleWrap = root.querySelector(".pg-single");
  const compareWrap = root.querySelector(".pg-compare");
  const mainCv = root.querySelector(".pg-canvas");
  const sparkCv = root.querySelector(".pg-spark");
  const logEl = root.querySelector(".pg-log");
  const statEvals = root.querySelector(".st-evals .v");
  const statBest = root.querySelector(".st-best .v");
  const statBasins = root.querySelector(".st-basins .v");
  const statCore = root.querySelector(".st-core");
  const statCoreV = root.querySelector(".st-core .v");
  const cmpCvs = Array.from(root.querySelectorAll(".pg-cell canvas"));
  const cmpBest = Array.from(root.querySelectorAll(".pg-cell .pg-cell-best"));
  const cmpCells = Array.from(root.querySelectorAll(".pg-cell"));
  const btnPlay = root.querySelector(".btn-play");
  const kWrap = root.querySelector(".ctl-k-wrap");
  const kSlider = root.querySelector(".ctl-k");
  const kVal = root.querySelector(".ctl-k-val");

  function newLandscape() {
    land = new Landscape2D(state.seed);
    heat = land.heatmap(230);
  }
  function newOpt() {
    opt = new MiniS3Turbo(land, { strategy: state.strategy, mask: state.mask, seed: state.seed, escapeK: state.escapeK });
    opts = STRATS.map((s) => new MiniS3Turbo(land, { strategy: s, mask: state.mask, seed: state.seed, escapeK: state.escapeK }));
    logEl.innerHTML = "";
    drawAll();
  }
  function syncKvisibility() {
    // the escape_k dial only applies to the reactive scout; dim it otherwise (still usable in race mode)
    if (!kWrap) return;
    kWrap.style.opacity = state.strategy === "reactive" || state.compare ? "1" : "0.4";
  }

  function drawLandscapeInto(ctx, size, o, revealCores) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, 0, 0, size, size);
    const P = (u) => [u[0] * size, u[1] * size];
    // hidden cores
    if (revealCores) {
      for (const b of land.basins) {
        if (!b.global) continue;
        const [x, y] = P(b.center);
        ctx.strokeStyle = "rgba(52,211,153,0.85)";
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(x, y, 9, 0, 6.283); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    if (!o) return;
    // evaluated points
    for (const p of o.ptsLog) {
      const [x, y] = P(p.u);
      ctx.fillStyle = KIND_COLOR[p.kind] || "#22d3ee";
      ctx.globalAlpha = p.kind === "init" || p.kind === "sobol" ? 0.55 : 0.85;
      const r = p.kind === "scout" ? 3.2 : p.kind === "focus" ? 2.8 : 2.3;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.283); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // region boxes
    o.regions.forEach((r, i) => {
      const lo = P([clamp(r.center[0] - r.radius / 2, 0, 1), clamp(r.center[1] - r.radius / 2, 0, 1)]);
      const hi = P([clamp(r.center[0] + r.radius / 2, 0, 1), clamp(r.center[1] + r.radius / 2, 0, 1)]);
      const isFocus = o.focusIdx === i && o.validFocus();
      ctx.lineWidth = r.kind === "main" ? 2 : 1.5;
      ctx.strokeStyle = r.kind === "main" ? "#22d3ee" : isFocus ? "#c084fc" : "#f472b6";
      ctx.setLineDash(r.kind === "main" ? [] : [5, 4]);
      if (isFocus) {
        ctx.save();
        ctx.shadowColor = "#c084fc"; ctx.shadowBlur = 10;
      }
      ctx.strokeRect(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]);
      if (isFocus) ctx.restore();
      ctx.setLineDash([]);
    });
    // x0 cross
    {
      const [x, y] = P(land.x0);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y);
      ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5);
      ctx.stroke();
    }
    // best point ring
    const b = o.best();
    {
      const [x, y] = P(b.u);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6.5, 0, 6.283); ctx.stroke();
      ctx.strokeStyle = "rgba(34,211,238,0.9)";
      ctx.beginPath(); ctx.arc(x, y, 9.5, 0, 6.283); ctx.stroke();
    }
  }

  function drawSingle() {
    const w = mainCv.parentElement.clientWidth - 2;
    const size = Math.min(w, 560);
    const ctx = setupCanvas(mainCv, size, size);
    drawLandscapeInto(ctx, size, opt, true);
  }

  function drawSpark() {
    const w = sparkCv.parentElement.clientWidth - 2, h = 84;
    const ctx = setupCanvas(sparkCv, w, h);
    ctx.clearRect(0, 0, w, h);
    const src = state.compare ? opts : [opt];
    const cols = state.compare ? CMP_COLORS : ["#22d3ee"];
    const vmax = 0.5;
    const toY = (v) => h - 6 - Math.sqrt(clamp(v, 0, vmax) / vmax) * (h - 14); // sqrt expands the low range
    ctx.strokeStyle = "rgba(52,211,153,0.6)";
    ctx.setLineDash([3, 3]);
    const coreY = toY(0.04);
    ctx.beginPath(); ctx.moveTo(0, coreY); ctx.lineTo(w, coreY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#34d399";
    ctx.font = "9.5px " + getComputedStyle(document.body).getPropertyValue("--mono");
    ctx.fillText("core threshold", 4, coreY - 3);
    src.forEach((o, k) => {
      if (!o || !o.curve.length) return;
      ctx.strokeStyle = cols[k % cols.length];
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      o.curve.forEach((v, i) => {
        const x = (i / (PG_BUDGET - 1)) * w;
        const y = toY(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  function drawCompare() {
    const w = (cmpCvs[0].parentElement.clientWidth || 260) - 2;
    opts.forEach((o, i) => {
      const ctx = setupCanvas(cmpCvs[i], w, w);
      drawLandscapeInto(ctx, w, o, true);
      cmpCells[i].querySelector(".pg-cell-label").style.color = CMP_COLORS[i];
      const b = o.curve.length ? o.curve[o.curve.length - 1] : NaN;
      cmpBest[i].textContent = "best " + (isFinite(b) ? b.toFixed(4) : "…") + " · " + o.y.length + " evals";
      cmpCells[i].style.outline = b <= 0.04 ? "2px solid rgba(52,211,153,0.8)" : "none";
      cmpCells[i].style.borderRadius = "8px";
    });
  }

  function refreshStats() {
    const o = state.compare ? opts[3] : opt; // in race mode the panel tracks the switch instance
    statEvals.textContent = o.y.length + " / " + PG_BUDGET;
    const b = o.curve.length ? o.curve[o.curve.length - 1] : NaN;
    statBest.textContent = isFinite(b) ? b.toFixed(4) : "…";
    statBasins.textContent = o.basinsSeen.size + " / " + land.basins.length;
    const hit = b <= 0.04;
    statCore.classList.toggle("hit", hit);
    statCoreV.textContent = hit ? "reached ✓" : "not yet";
  }

  function flushEvents() {
    const src = state.compare ? opts[3] : opt; // in compare mode, log the switch instance
    if (!src.events.length) return;
    for (const ev of src.events) {
      const div = document.createElement("div");
      div.className = "ev-" + ev.t;
      div.textContent = ev.msg;
      logEl.appendChild(div);
    }
    if (state.compare) opts.forEach((o) => (o.events = []));
    else src.events = [];
    while (logEl.children.length > 80) logEl.removeChild(logEl.firstChild);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function drawAll() {
    if (state.compare) drawCompare(); else drawSingle();
    drawSpark();
    refreshStats();
    flushEvents();
  }

  function tick() {
    if (state.compare) {
      let anyLeft = false;
      for (const o of opts) { o.step(); anyLeft = anyLeft || !o.done; }
      if (!anyLeft) pause();
    } else {
      opt.step();
      if (opt.done) pause();
    }
    drawAll();
  }
  function play() {
    if (state.playing) return;
    state.playing = true;
    btnPlay.textContent = "❚❚ pause";
    timer = setInterval(tick, 1000 / state.speed);
  }
  function pause() {
    state.playing = false;
    btnPlay.textContent = "▶ play";
    if (timer) clearInterval(timer);
    timer = null;
  }
  function finish() {
    pause();
    const run = (o) => { let guard = 0; while (!o.done && guard++ < 200) o.step(); };
    if (state.compare) opts.forEach(run); else run(opt);
    drawAll();
  }

  /* wiring */
  root.querySelectorAll(".seg-strat button").forEach((b) => {
    b.addEventListener("click", () => {
      root.querySelectorAll(".seg-strat button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.strategy = b.dataset.strat;
      syncKvisibility();
      pause(); newOpt();
    });
  });
  root.querySelector(".sel-mask").addEventListener("change", (e) => {
    state.mask = e.target.value;
    pause(); newOpt();
  });
  if (kSlider) {
    kSlider.addEventListener("input", () => {
      state.escapeK = parseFloat(kSlider.value);
      kVal.textContent = state.escapeK.toFixed(2);
      pause(); newOpt();
    });
  }
  btnPlay.addEventListener("click", () => (state.playing ? pause() : play()));
  root.querySelector(".btn-step").addEventListener("click", () => { pause(); tick(); });
  root.querySelector(".btn-finish").addEventListener("click", finish);
  root.querySelector(".btn-reset").addEventListener("click", () => { pause(); newOpt(); });
  root.querySelector(".btn-newland").addEventListener("click", () => {
    state.seed = (state.seed % 9999) + 1 + Math.floor(Math.random() * 37);
    root.querySelector(".ctl-seed").value = state.seed;
    pause(); newLandscape(); newOpt();
  });
  root.querySelector(".ctl-seed").addEventListener("change", (e) => {
    state.seed = parseInt(e.target.value, 10) || 1;
    pause(); newLandscape(); newOpt();
  });
  const spd = root.querySelector(".ctl-speed");
  spd.addEventListener("input", () => {
    state.speed = parseInt(spd.value, 10);
    root.querySelector(".ctl-speed-val").textContent = state.speed + "×";
    if (state.playing) { pause(); play(); }
  });
  root.querySelector(".btn-compare").addEventListener("click", (e) => {
    state.compare = !state.compare;
    e.target.textContent = state.compare ? "◱ single view" : "⊞ race all four";
    singleWrap.style.display = state.compare ? "none" : "";
    compareWrap.style.display = state.compare ? "" : "none";
    syncKvisibility();
    pause(); newOpt();
  });

  onResize(root, drawAll);
  syncKvisibility();
  newLandscape();
  newOpt();
})();
