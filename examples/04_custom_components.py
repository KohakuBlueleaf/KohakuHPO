"""Extending the framework: a custom optimizer, a custom parameter codec, and a custom S3-TuRBO
mask law, each selected by name after one decorator."""

import numpy as np

import kohakuhpo as khpo
from kohakuhpo import MASK, OPTIMIZER, PARAM


# 1) A new optimizer: implement _ask(q) -> (q, d) in [0,1]^d; usable from every interface.
@OPTIMIZER.register("hill_climber")
class HillClimber(khpo.Optimizer):
    """Gaussian steps around the incumbent, with a fixed step size."""

    def __init__(self, space, seed: int = 0, step: float = 0.1) -> None:
        super().__init__(space, seed)
        self.step = step

    def _ask(self, q: int) -> np.ndarray:
        if len(self.y) == 0:
            return self.rng.random((q, self.dim))
        center = self.best[0]
        return center[None] + self.rng.normal(0.0, self.step, (q, self.dim))


# 2) A new parameter kind: quantized float (decode/encode contract).
@PARAM.register("qfloat")
class QuantizedFloat:
    def __init__(self, lo, hi, step):
        self.lo, self.hi, self.step = lo, hi, step

    def decode(self, u):
        return round((self.lo + u * (self.hi - self.lo)) / self.step) * self.step

    def encode(self, v):
        return (v - self.lo) / (self.hi - self.lo)


# 3) A new S3-TuRBO mask law: exact top-k active coordinates per step.
@MASK.register("topk")
def mask_topk(rng, n, dim, rho):
    k = max(1, int(round(rho * dim)))
    m = np.zeros((n, dim))
    for row in m:
        row[rng.choice(dim, size=k, replace=False)] = 1.0
    return m


space = khpo.SearchSpace(
    {
        "x": ("float", 0.0, 1.0),
        "dropout": {"name": "qfloat", "lo": 0.0, "hi": 0.5, "step": 0.05},
    }
)


def f(cfg):
    return (cfg["x"] - 0.3) ** 2 + (cfg["dropout"] - 0.1) ** 2


print("custom optimizer :", khpo.minimize(f, space, "hill_climber", budget=60, seed=0).best_value)
print(
    "custom mask law  :",
    khpo.minimize(
        f, space, {"name": "s3turbo", "mask_distribution": "topk"}, budget=60, seed=0
    ).best_value,
)
