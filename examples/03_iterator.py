"""Iterator: the optimization loop as a for-loop. Each Batch carries configs; report() the values
to unlock the next batch. Failures (None / NaN) are absorbed by the study's failure policy."""

import kohakuhpo as khpo
from kohakuhpo.benchmarks import ManyBasin

problem = ManyBasin(dim=10, seed=0)  # many look-alike basins, a few hide a deep narrow core

study = khpo.Study(
    problem.space,
    {"name": "s3turbo", "preset": "multibasin", "budget": 200},
    seed=0,
    x0=problem.x0,
)

for batch in study.loop(budget=200, q=4):
    values = []
    for cfg in batch.configs:
        v = problem(cfg)
        values.append(None if v > 10 else v)  # pretend huge values crash the pipeline
    batch.report(values)

print(f"best regret: {study.best_value:.4f}  (core reached: {study.best_value <= 0.04})")
print(f"failed trials absorbed: {sum(t.state == 'failed' for t in study.trials)}")
