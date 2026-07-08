# Extending the framework

Everything swappable lives in a registry. Register a class/function, or reference it by dotted
path (no registration needed); `build(spec, REGISTRY, **kw)` resolves it at construction time.

```python
from kohakuhpo import build, OPTIMIZER
build("s3turbo", OPTIMIZER, space=space, seed=0)                # registry name
build({"name": "s3turbo", "preset": "rugged"}, OPTIMIZER, space=space)  # name + kwargs
build("my_pkg.opt.MyOpt", None, space=space)                    # dotted path
build(MyOpt, None, space=space)                                 # a class
build(instance)                                                 # passthrough
```

## A new optimizer

Subclass `Optimizer`, implement `_ask(q) -> (q, d)` in `[0,1]^d` (minimization); extend `tell`
only if you need feedback, and call `super().tell`. That is the whole contract; the Study layer
gives you typed configs, direction, failures and parallel evaluation for free.

```python
import numpy as np
from kohakuhpo import Optimizer, OPTIMIZER

@OPTIMIZER.register("my_opt")
class MyOpt(Optimizer):
    def _ask(self, q: int) -> np.ndarray:
        if len(self.y) == 0:
            return self.rng.random((q, self.dim))
        return self.best[0][None] + self.rng.normal(0, 0.05, (q, self.dim))
```

```python
khpo.minimize(f, space, "my_opt", budget=100)        # or "my_pkg.MyOpt" without registration
```

Useful inherited pieces: `self.U / self.y` (all observations), `self.best`,
`self.train_set(max_n)` (best-capped observations for a surrogate),
`kohakuhpo.surrogate.GP` (fit / predict / joint-sample), `kohakuhpo.device.tensor_kw()`.

## A new parameter kind

The codec contract is `decode(u: float) -> value` and `encode(value) -> u in [0,1]`.

```python
from kohakuhpo import PARAM

@PARAM.register("qlog")
class QLog:
    def __init__(self, lo, hi, step): ...
    def decode(self, u): ...
    def encode(self, v): ...

space = khpo.SearchSpace({"bs": {"name": "qlog", "lo": 8, "hi": 512, "step": 8}})
```

A pre-built codec instance (anything with `decode`/`encode`) can be passed directly as the spec.

## A new acquisition (for `gpbo`)

```python
from kohakuhpo import ACQUISITION

@ACQUISITION.register("greedy")
def greedy(mean, std, best):
    return -mean          # maximized score

khpo.minimize(f, space, {"name": "gpbo", "acquisition": "greedy"}, budget=100)
```

## A new S3-TuRBO mask law

Contract: `mask(rng, n, dim, rho) -> alpha in [0,1]^(n, dim)`; a candidate is
`center + alpha * (raw - center)`. Keep expected active mass near `rho * dim` so the derived
active fraction `rho = 1/sqrt(d)` keeps its meaning.

```python
from kohakuhpo import MASK

@MASK.register("topk")
def mask_topk(rng, n, dim, rho):
    k = max(1, int(round(rho * dim)))
    m = np.zeros((n, dim))
    for row in m:
        row[rng.choice(dim, size=k, replace=False)] = 1.0
    return m

khpo.minimize(f, space, {"name": "s3turbo", "mask_distribution": "topk"}, budget=100)
```

## A new S3-TuRBO scout strategy

Subclass `ScoutStrategy` and override up to three methods: `want_scout(opt)` (spend a scout slot
this ask?), `select(opt, q)` (allocate the batch; return `(points, region_indices, kinds)` where
kind `"scout"` marks probe slots), and `on_tell(opt, points, values)` (promote observations into
candidate regions). The optimizer provides the shared machinery: `opt._local_batch` /
`opt._main_batch` / `opt._local_one` (masked local Thompson picks), `opt._farthest_point`,
`opt._accept` / `opt._add_candidate` / `opt._mine_archive`.

```python
from kohakuhpo import SCOUT, ScoutStrategy

@SCOUT.register("always_probe")
class AlwaysProbe(ScoutStrategy):
    def want_scout(self, opt):
        return True

    def select(self, opt, q):
        picks, ridx, kinds = opt._local_batch(q - 1)
        picks.append(opt._farthest_point())
        ridx.append(-1)
        kinds.append("scout")
        return picks, ridx, kinds

khpo.minimize(f, space, {"name": "s3turbo", "scout_strategy": "always_probe"}, budget=100)
```

## A new benchmark objective

Register a callable class with a `.space` attribute (and optionally `.x0`) into `OBJECTIVE`; the
comparison example and your own harnesses can then build it by name.
