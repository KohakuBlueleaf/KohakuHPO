# kohakuhpo

Sample-efficient **black-box optimization** for expensive functions (hyperparameter tuning and
beyond). The philosophy: *one ask/tell core, everything swappable is a registry entry.*

## Documentation map

| Doc | What it covers |
|---|---|
| [interfaces.md](interfaces.md) | The Study object and the three usage modes (ask/tell, iterator, closure), direction, x0, failures |
| [optimizers.md](optimizers.md) | The optimizer catalogue, S3-TuRBO's axes and presets, which to use when |
| [extending.md](extending.md) | Custom optimizers, parameter codecs, acquisitions, mask laws, scout strategies |
| [gpu.md](gpu.md) | Device policy, vectorized objectives, raw cube mode, process pools |
| [s3turbo-method.md](s3turbo-method.md) | The full S3-TuRBO method note: assumptions, propositions, derived constants, benchmark tables |

## The one mental model

1. An **optimizer** is a pure cube searcher: `ask(q) -> (q, d) in [0,1]^d`, `tell(U, y)`,
   always minimizing. It knows nothing about parameter types, direction, budgets or parallelism.
2. A **space** is a codec: typed parameters `<->` cube axes. The only place types exist; each
   parameter kind is a `PARAM` registry entry.
3. A **Study** owns a run: space + optimizer + trial history + direction + failure policy. The
   ask/tell, iterator and closure interfaces are views over the same Study, so they can be mixed
   within one run.
4. Everything swappable lives in a **registry**; `build(spec, REGISTRY)` resolves a name, dotted
   path, dict, class or instance to a concrete object once, at build time.

## Quickstart

```python
import kohakuhpo as khpo

space = khpo.SearchSpace({
    "lr": ("log", 1e-5, 1e-1),
    "layers": ("int", 2, 12),
    "act": ("cat", ["relu", "gelu", "silu"]),
})

result = khpo.minimize(objective, space, optimizer="s3turbo", budget=200, q=4)
print(result.best_config, result.best_value)
```

Runnable walkthroughs live in [`examples/`](../examples).
