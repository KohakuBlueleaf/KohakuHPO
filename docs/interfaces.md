# Interfaces: one Study, three ways to drive it

A `Study` owns a run. Construct it with a space and an optimizer spec; everything below operates
on the same trial history, so the modes compose mid-run.

```python
study = khpo.Study(
    space,
    {"name": "s3turbo", "preset": "multibasin", "budget": 300},
    seed=0,
    direction="min",           # or "max": bigger values are better everywhere you see them
    x0={"lr": 3e-4, ...},      # optional warm start (a config dict, or a list of them)
    failure_value=None,        # None -> 1e9 for min, -1e9 for max
)
```

## 1. ask/tell (HEBO-style)

You control evaluation completely; the Study never calls your code.

```python
while study.n_evals < 300:
    configs = study.ask(4)                 # list of typed config dicts
    study.tell(configs, [f(c) for c in configs])

study.best_config, study.best_value, study.trials
```

Rules:

* `tell` matches configs to pending asked trials by equality; a config that was never asked is
  accepted as an **injected observation** (import old runs, add a known-good point).
* A value of `None`, NaN or inf marks the trial **failed**; it is recorded at `failure_value` so
  the optimizer sees a bad-but-finite score.
* Batch-coherent tells (tell exactly what the last ask returned, together) give trust-region
  methods the cleanest per-batch feedback; out-of-order tells are accepted.

## 2. Iterator

The loop is a for-loop; `report()` gates the next batch.

```python
for batch in study.loop(budget=300, q=4, progress=True):   # budget is a TOTAL: a warm study resumes
    batch.report([f(c) for c in batch.configs])
```

`progress=True` (also accepted by `optimize` and `minimize`/`maximize`) shows a tqdm bar with the
running best value in its postfix.

## 3. Closure

Evaluation handled for you: sequential, process-pool, or vectorized.

```python
result = study.optimize(f, budget=300, q=4, workers=8)          # pool of 8 processes
result = study.optimize(f_batch, budget=300, q=8, vectorized=True)  # f(list[dict]) -> values
```

Or in one call, without touching Study:

```python
result = khpo.minimize(f, space, "s3turbo", budget=300, q=4)
result = khpo.maximize(reward, space, "s3turbo", budget=300, q=4)
```

`Result` carries `best_config`, `best_value`, `history` (per-evaluation values, in order),
`best_so_far` (the running incumbent trace, ready to plot) and the full `trials` list.

## Mixing modes

```python
study = khpo.Study(space, "s3turbo", x0=known_good)
cfgs = study.ask(4); study.tell(cfgs, hand_evaluated)   # manual warm-up
study.optimize(f, budget=300, q=4)                      # hand over the rest of the budget
```

## Warm start and resume

* `x0` configs are served by the first `ask` calls, so the known start is evaluated first in
  every mode.
* To resume from a previous run, replay its trials: `study.tell(old_configs, old_values)` before
  continuing (deterministic optimizers rebuild equivalent state from the observations).
