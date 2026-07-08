# Soft-Sparse Scout TuRBO: project page

A fully static, dependency-free project page / tech blog for the Soft-Sparse Scout TuRBO method, with
interactive demos in the style of an explorable explainer. Dark kohaku theme.

## Run it

No build step. Either open `index.html` directly in a browser, or serve the folder:

```bash
python -m http.server -d webpage 8080   # then open http://localhost:8080
```

Everything is self-contained (vendored KaTeX + uPlot, baked-in benchmark data), so the folder can be
copied verbatim to any static host (GitHub Pages, nginx, the homepage).

## What is on the page

- **Batch Thompson sampling, live**: a 1-D GP with the exact discretized batch-TS rule the method uses
  (candidate pool, q joint posterior draws, per-draw pool argmin). Click to add observations, or press
  "evaluate batch" to play the ask/tell loop.
- **Mask distribution explorer**: the dense / hard / soft move laws over 25 coordinates, the marginal law
  of a single weight, the two limits of Proposition 2 as one-click buttons, and the derived sparsity
  anneal with a dimension slider.
- **Many-basin playground**: a faithful 2-D port of the many-basin benchmark family plus a faithful
  miniature of the optimizer (trust regions, local GP + batch TS, masks, all four scout strategies,
  promotion gates, importance pruning, focus bursts, and the section-5 derivations specialized to d=2,
  q=4, B=300). Single view or a synchronized four-way race on the same landscape and seed. Default seed
  10 is chosen so that only `switch` reaches a hidden core.
- **Benchmark explorer**: the real measured curves (mean best-so-far over seeds) behind the paper tables,
  rendered with uPlot; one tab per task, per-task ranked legend, fixed per-method colors, log axes where
  the range spans decades.
- Full method exposition: problem assumptions, base method + Proposition 1, both axes with Propositions
  2 to 4, derived constants, benchmark tables with the mean-rank row, the six presets, limits, citation.

## Layout

```
index.html            the whole article
css/style.css         theme + layout + demo styling
js/common.js          seeded RNG, Beta/Gamma samplers, Cholesky, mini exact GP, canvas helpers
js/demo-ts.js         batch Thompson sampling demo
js/demo-mask.js       mask explorer
js/demo-playground.js Landscape2D + MiniS3Turbo (the 2-D optimizer port) + UI harness
js/demo-bench.js      uPlot benchmark explorer
js/main.js            KaTeX auto-render, TOC scroll-spy, copy-citation, #debug error overlay
data/bench_data.js    baked benchmark curves (regenerate with make_data.py)
vendor/               KaTeX + uPlot, vendored (no CDN, works offline)
assets/               the method note PDF
make_data.py          regenerates data/bench_data.js from the benchmark JSONL
```

## Regenerating the benchmark data

```bash
python webpage/make_data.py <dir-with-benchmark-jsonl>
```

reads the benchmark JSONL (`presets_vs_baselines_{academic,mn}.jsonl`, produced by the validated
benchmark harness) and rewrites `data/bench_data.js` with per-(task, method) mean curves.

## Debugging

Open the page with `#debug` appended to the URL to surface any runtime JS error as a red overlay
(used by the headless screenshot tests).
