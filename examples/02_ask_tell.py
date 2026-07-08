"""Ask/tell: full control over evaluation. Your training loop, cluster, or lab equipment sits
between ask() and tell(); the Study never calls your code."""

import kohakuhpo as khpo

space = khpo.SearchSpace({"x": ("float", -5.0, 5.0), "y": ("float", -5.0, 5.0)})


def himmelblau(cfg: dict) -> float:
    x, y = cfg["x"], cfg["y"]
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


# x0 seeds a known starting point: it is returned by the first ask() before anything else.
study = khpo.Study(
    space,
    {"name": "s3turbo", "preset": "balanced", "budget": 100},
    seed=0,
    x0={"x": 0.0, "y": 0.0},
)

while study.n_evals < 100:
    configs = study.ask(4)
    values = [himmelblau(c) for c in configs]
    study.tell(configs, values)
    if study.n_evals % 20 == 0:
        print(f"[{study.n_evals:3d}] best = {study.best_value:.5f}")

print(f"best config: {study.best_config}")

# A result you obtained elsewhere can be injected at any time; it is just another observation.
study.tell({"x": 3.0, "y": 2.0}, himmelblau({"x": 3.0, "y": 2.0}))
print(f"after injecting the known optimum: best = {study.best_value:.5f}")
