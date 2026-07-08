# KohakuHPO

**Sample-efficient black-box optimization** for expensive functions: hyperparameter tuning and
anything shaped like it (simulator calibration, controller tuning, tool-parameter search). One
ask/tell core on numpy + torch (CPU/GPU), a unified interface over every usage style, our new
**S3-TuRBO** optimizer as the flagship, and clean from-scratch implementations of the standard
baselines behind the same contract.

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

| key | method |
|---|---|
| `s3turbo` | **S3-TuRBO** (Soft-Sparse Scout TuRBO): trust-region batch Thompson sampling with a coordinate-mask axis and a scout/escape axis; internal constants derived from the problem |
| `turbo` / `warped_turbo` | TuRBO with its default batch-TS acquisition (optionally HEBO-warped) |
| `gpbo` | global GP + log-EI (pluggable acquisition), constant-liar batching |
| `hebo` | warped-GP + rank-combined EI/PI/UCB, from scratch |
| `cmaes` | CMA-ES (via the `cmaes` package) |
| `sobol`, `random` | quasirandom / uniform floors |

The default is the **adaptive soft mask** with the `reactive` scout: the mask learns its own
concentration online from which coordinates pay off, so it adapts to the landscape without being told
the regime. In an 8-task ablation it is the best-ranked mask and every mask beats no-mask, and
S3-TuRBO ranks first against CMA-ES, TuRBO, GP-BO, HEBO, and Sobol. The escape axis reduces to one
dial, `escape_k`:

```python
khpo.minimize(f, space, {"name": "s3turbo", "budget": 300}, budget=300, q=4)  # reactive, escape_k=0.75
```

`reactive` spends a small derived base rate of far probes, `rho_0 = 1/(escape_k * sqrt(d))`, and lets
the outcome of the candidate regions it has already planted modulate that spend. Large `escape_k`
approaches pure-local `none` (right for a near-unimodal problem); small `escape_k` scouts aggressively
toward `switch`-like escape (right for a multi-basin one). The default `escape_k=0.75` ranks first on
the real sklearn HPO suite and second only to pure-local on the smooth synthetic suite; lower it toward
`0.5` for spaces likely to hide separated basins (model-family search), raise it above `1` for
near-certainly unimodal ones. Use `switch` directly when far basins are near-certain to hide a better
core (the strongest raw escape but costly elsewhere, §7.3), or `random` for the naive periodic-probe
baseline. The fixed masks (`dense`, `hard`, `soft`) and the named `preset`s (`balanced`, `rugged`, ...)
are kept as static reference configurations for the ablation; pick one only when the regime is known
and you want a fully fixed setup.

See [docs/optimizers.md](docs/optimizers.md) for the catalogue, the adaptive-mask math and benchmark,
and guidance on choosing `q`.

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

## Citation

```bibtex
@misc{kohaku2026s3turbo,
  title  = {Soft-Sparse Scout TuRBO: Trust-Region Thompson Sampling
            with Orthogonal Move and Escape Axes},
  author = {Shih-Ying Yeh},
  year   = {2026},
  note   = {KohakuHPO: https://github.com/KohakuBlueleaf/KohakuHPO}
}
```

Apache-2.0. Built by [Kohaku BlueLeaf](https://github.com/KohakuBlueleaf).
