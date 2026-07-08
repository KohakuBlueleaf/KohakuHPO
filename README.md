# KohakuHPO

### A modern, unified black-box optimization system

**Sample-efficient black-box optimization** for expensive functions: hyperparameter tuning and
anything shaped like it (simulator calibration, controller tuning, tool-parameter search). One
ask/tell core on numpy + torch (CPU/GPU), a unified interface over every usage style, our new
**S3-TuRBO** optimizer as the flagship, and clean from-scratch implementations of the standard
baselines (Sobol, CMA-ES, GP-BO, HEBO, TuRBO) behind the same contract.

This repository is two things at once. As a **package**, it is a practical, extensible HPO library
you can drop into a project (Part 1 below). As a **research artifact**, it is the reference
implementation and evaluation of **S3-TuRBO**, a new trust-region Bayesian optimizer
([Part 2: The S3-TuRBO method](#part-2-the-s3-turbo-method)). Read whichever half you came for.

---

## Part 1: The KohakuHPO package

```python
import kohakuhpo as khpo

space = khpo.SearchSpace({
    "lr":     ("log", 1e-5, 1e-1),
    "beta2":  ("float", 0.9, 0.999),
    "layers": ("int", 2, 12),
    "act":    ("cat", ["relu", "gelu", "silu"]),
})

result = khpo.minimize(objective, space, optimizer="s3turbo", budget=200, q=4)
print(result.best_config, result.best_value)
```

## Why this library

* **Built for expensive evaluations.** The target regime is budgets of hundreds of evaluations
  where each one costs seconds to hours: GP surrogates, trust regions, batch proposals. If your
  function is cheap and you can afford millions of calls, use CMA-ES directly or gradients.
* **One core, every interface.** The same run can be driven closure-style, ask/tell-style, or as
  a for-loop, and you can switch between them mid-run, because they are views over one `Study`.
* **A real new method, honestly benchmarked.** On a 9-task synthetic suite, S3-TuRBO ranks first
  against CMA-ES, GP-BO, HEBO, TuRBO and Sobol; a mask-axis ablation shows every mask beats
  no-mask and the adaptive mask ranks best. On a **real** scikit-learn HPO suite (SVC tuning + AutoML
  model-family search, 5 seeds, budget 50), both S3-TuRBO scouts beat every baseline, and the default
  `reactive` scout (dial `escape_k=0.75`) ranks first overall (avg rank 1.80 vs 2.20 for pure-local
  `none`). The interactive project page in
  [`webpage/`](webpage) shows every mechanism live, and the method note has the math.
* **Extensible by decorator.** New optimizers, parameter kinds, acquisitions (and even new mask
  laws or scout strategies *inside* S3-TuRBO) are one `@REGISTRY.register(...)` away, or a
  dotted path away with no registration at all.

## Install

```bash
pip install -e .            # from a checkout
# deps: numpy, scipy, torch, cmaes, tqdm, matplotlib
```

Python >= 3.10. Optional extras: `pip install -e .[dev]` (pytest, ruff, black),
`.[real]` (scikit-learn for the real-HPO examples).

## Three ways to use it

**Closure**: hand over a function, get a result:

```python
result = khpo.minimize(f, space, "s3turbo", budget=300, q=4, workers=8, x0=known_good)
result.best_config, result.best_value, result.best_so_far   # trace ready to plot
```

**Ask/tell**: you own the evaluation loop (cluster jobs, lab hardware, a human in the loop):

```python
study = khpo.Study(space, {"name": "s3turbo", "budget": 300}, seed=0)  # reactive scout by default
while study.n_evals < 300:
    configs = study.ask(4)
    study.tell(configs, [f(c) for c in configs])
```

**Iterator**: the loop is a for-loop:

```python
for batch in study.loop(budget=300, q=4):
    batch.report([f(c) for c in batch.configs])
```

All three share the `Study` state, so warm up manually and then hand the remaining budget to
`study.optimize(...)`. Direction (`"min"`/`"max"`), warm starts (`x0` is evaluated first),
failures (exceptions / NaN become a finite penalty, never a crash), and injecting observations
that were never asked (`study.tell(old_config, old_value)`) are all handled at this layer. No
optimizer ever needs to know.

## The optimizers

Every optimizer is registered behind the same ask/tell contract, so they are interchangeable in any
of the interfaces above.

| key | method |
|---|---|
| `s3turbo` | **S3-TuRBO** (Soft-Sparse Scout TuRBO): trust-region batch Thompson sampling with a coordinate-mask axis and a scout/escape axis; internal constants derived from the problem. The flagship, detailed in [Part 2](#part-2-the-s3-turbo-method) |
| `turbo` / `warped_turbo` | TuRBO with its default batch Thompson-sampling acquisition (optionally HEBO-warped) |
| `gpbo` | global GP + log-EI (pluggable acquisition), constant-liar batching |
| `hebo` | warped-GP + rank-combined EI/PI/UCB, from scratch |
| `cmaes` | CMA-ES (via the `cmaes` package) |
| `sobol`, `random` | quasirandom / uniform floors |

The default optimizer is `s3turbo`; its configuration is covered in Part 2. See
[docs/optimizers.md](docs/optimizers.md) for the full catalogue, per-optimizer notes, and guidance on
choosing the batch size `q`.

## GPU and batched objectives

```python
khpo.use_device("cuda:0")                      # GP math (Cholesky, L-BFGS, draws) on GPU
khpo.minimize(f_batch, space, "s3turbo", budget=400, q=32, vectorized=True)   # f(list[dict])
```

For fully tensor-native workflows, skip config dicts entirely and drive the cube-level optimizer
with arrays (`SearchSpace.from_dim(d)` + `opt.ask/tell`); see [docs/gpu.md](docs/gpu.md).

## Extending

```python
@khpo.OPTIMIZER.register("my_opt")
class MyOpt(khpo.Optimizer):
    def _ask(self, q):                 # (q, d) in [0,1]^d, that's the whole contract
        ...
```

The same pattern covers parameter codecs (`PARAM`), acquisitions (`ACQUISITION`), and S3-TuRBO's
own axes (`MASK`, `SCOUT`); [docs/extending.md](docs/extending.md) has a worked example of each.

## Repository layout

```
src/kohakuhpo/
├── registry.py        Registry + build(): named registries, dotted-path escape hatch
├── space.py           SearchSpace + parameter codecs (PARAM registry)
├── optimizer.py       the cube-level ask/tell base contract
├── study.py           Study / Trial / Batch / Result: direction, x0, failures, the interfaces
├── run.py             minimize() / maximize()
├── device.py          global device/dtype policy for GP tensors
├── surrogate/         exact ARD Matern-5/2 GP (+ warps) and acquisitions
├── optimizers/        random, sobol, cmaes, gpbo, hebo, turbo, and
│   └── s3turbo/       the flagship: masks.py + scouts.py + regions.py + optimizer.py
└── benchmarks/        classic BO test functions + the many-basin family (OBJECTIVE registry)

docs/                  interfaces, optimizers, extending, gpu
examples/              runnable walkthroughs (quickstart -> custom components -> GPU -> sklearn HPO -> adaptive mask -> reactive scout)
webpage/               the interactive S3-TuRBO project page (static; open index.html)
tests/                 contract tests over all registered optimizers + focused gates
```

## Documentation

Start at [docs/README.md](docs/README.md) (the mental model and the doc map), then the runnable
[examples/](examples). The S3-TuRBO method itself (assumptions, propositions, derived constants,
full benchmark tables) is in [docs/s3turbo-method.md](docs/s3turbo-method.md), with an
interactive companion on the project page ([`webpage/`](webpage), no build step: open
`index.html` or `python -m http.server -d webpage`).

## Development

```bash
pip install -e .[dev]
pytest tests/ -q          # the whole suite runs in about a minute on CPU
ruff check src/ tests/ && black --check src/ tests/
```

---

# Part 2: The S3-TuRBO method

### Adaptive Sparse Moves and Evidence-Gated Escape for Trust-Region Bayesian Optimization

The flagship optimizer, **S3-TuRBO** (Soft-Sparse Scout TuRBO), is the research contribution of this
repository. It extends single-region trust-region Bayesian optimization (TuRBO) along two independent,
separable axes, and derives almost every internal constant from the problem so that the optimizer does
not itself become a tuning problem. The full treatment (assumptions, propositions, derivations, and
benchmark tables) is the method note [docs/s3turbo-method.md](docs/s3turbo-method.md); the
[interactive project page](webpage) shows every mechanism live.

### Two separable contributions

* **An adaptive soft-sparse local move.** A trust-region step perturbs a masked subset of coordinates.
  The mask is a single Beta-law family whose `dense` and `hard` variants are limiting cases; the
  `adaptive` default learns its own sparsity and concentration online from which coordinates have paid
  off, so it fits the landscape without being told the regime. The active fraction is derived as
  `1/sqrt(d)`, not tuned. This axis is what carries the method on unimodal and simple landscapes, and it
  stands on its own with the escape switched off.

* **An evidence-gated escape (the reactive scout).** To reach far-apart basins, the scout spends a
  small derived base rate of far probes, `rho_0 = 1/(escape_k * sqrt(d))`, and lets the outcome of the
  candidate regions it has already planted modulate that spend: an escape value `E` rises when a planted
  candidate proves spatially distinct (by the already-derived novelty radius) and competitive, and
  decays when candidates prove redundant. It does not try to predict a far basin from local data (a
  local search's surrogate never samples the far region, so its data cannot reveal one); it reacts to
  cheap speculative outcomes. The whole escape axis reduces to one dial, `escape_k`.

Because the two axes are separate, the guidance splits cleanly: if you know the problem is close to
unimodal, use the mask alone (`scout_strategy="none"`); if the regime is unknown or multi-basin, add the
reactive scout (the default) and lower `escape_k` toward `0.5` as separated basins become more likely.

### The one dial: `escape_k`

```python
khpo.minimize(f, space, {"name": "s3turbo", "budget": 300}, budget=300, q=4)  # default: reactive, escape_k=0.75
```

Large `escape_k` approaches pure-local `none` (right for a near-unimodal problem); small `escape_k`
scouts aggressively toward `switch`-like escape (right for a multi-basin one). The default
`escape_k=0.75` is the best all-round setting; the recommended range is `0.5`-`1.0`. `switch` remains
available directly as the strongest raw escape for near-certain hidden cores, and the fixed masks
(`dense`, `hard`, `soft`) and named `preset`s are kept as static reference configurations for the
ablation.

### What the benchmarks show

Honestly benchmarked against clean from-scratch baselines (Sobol, CMA-ES, GP-BO, HEBO, TuRBO), reported
as average rank (lower is better):

* **Synthetic general (8 smooth tasks, d up to 25).** S3-TuRBO leads: pure-local `none` and the default
  `reactive` sit ahead of every external baseline, and a mask-axis ablation shows every mask beats
  no-mask with the adaptive mask ranking best.
* **Real scikit-learn HPO (5 tasks, budget 50).** The default `reactive` (`escape_k=0.75`) ranks
  **first overall** (avg rank 1.80 vs 2.20 for pure-local `none`), ahead of every baseline. This is the
  regime the method is built for.
* **Many-basin escape.** A smaller `escape_k` reaches narrower far cores; `escape_k=0.1` matches the
  aggressive `switch` scout on the hardest barrier-separated task, where pure-local methods stay trapped.
* **Bayesmark (HEBO's NeurIPS 2020 BBO Challenge suite).** On this deliberately low-dimensional, largely
  unimodal benchmark (HEBO's home turf), S3-TuRBO with the scout off scores highest and beats a HEBO
  reimplementation, and the default reactive ties it, evidence that the soft-sparse mask is a standalone
  contribution competitive with a competition winner on its own turf.

Full tables, curves, and the diagonal "each suite won by its matching `escape_k`" summary are in
[docs/s3turbo-method.md](docs/s3turbo-method.md).

## Citation

```bibtex
@misc{kohaku2026s3turbo,
  title  = {S3-TuRBO: Adaptive Sparse Moves and Evidence-Gated Escape
            for Trust-Region Bayesian Optimization},
  author = {Shih-Ying Yeh},
  year   = {2026},
  note   = {KohakuHPO: https://github.com/KohakuBlueleaf/KohakuHPO}
}
```

Apache-2.0. Built by [Kohaku BlueLeaf](https://github.com/KohakuBlueleaf).
