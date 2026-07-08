"""Classic synthetic optimization test functions from the BO/HPO literature (Surjanovic & Bingham; BoTorch).

Standard functions for benchmarking Bayesian optimization, with known global minima so regret is exact.
Each is a registered OBJECTIVE taking a config over the unit cube ``[0,1]^d`` (we rescale internally to the
function's canonical domain), returning ``f(x) - f_min`` (regret, >= 0, lower better). ``.space`` is the unit
cube; ``.x0`` is a fixed mediocre start (for the start-from-one leg).

Branin and Hartmann-6 are the two most-used BO benchmarks; Ackley/Rastrigin/Levy/Griewank are common
multimodal stressors; Rosenbrock/Styblinski-Tang are common valley/separable cases.
"""

import numpy as np

from kohakuhpo.registry import OBJECTIVE
from kohakuhpo.space import SearchSpace


def _cube(dim: int) -> SearchSpace:
    return SearchSpace({f"x{i}": ("float", 0.0, 1.0) for i in range(dim)})


class _Classic:
    """Base: rescale the unit cube to ``[lo, hi]^d``, evaluate ``_f``, return regret ``f - f_min``."""

    lo, hi, f_min, dim_default = -5.0, 5.0, 0.0, 2

    def __init__(self, dim: int | None = None, noise: float = 0.0, seed: int = 0) -> None:
        self.dim = dim or self.dim_default
        self.noise = noise
        self.space = _cube(self.dim)
        self._nrng = np.random.default_rng(seed + 7919)
        self.x0 = self.space.to_config(np.full(self.dim, 0.5))  # centre of the cube = a fixed start

    def _x(self, cfg: dict) -> np.ndarray:
        u = np.array([cfg[f"x{i}"] for i in range(self.dim)])
        return self.lo + u * (self.hi - self.lo)

    def _f(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(self, cfg: dict) -> float:
        v = self._f(self._x(cfg)) - self.f_min
        if self.noise:
            v += float(self._nrng.normal(0.0, self.noise))
        return float(v)


@OBJECTIVE.register("branin")
class Branin(_Classic):
    """2-D Branin-Hoo; 3 global minima, f_min=0.397887. Domain x1 in [-5,10], x2 in [0,15]."""

    dim_default = 2
    f_min = 0.397887

    def _x(self, cfg):  # non-square domain
        u = np.array([cfg["x0"], cfg["x1"]])
        return np.array([-5 + u[0] * 15, u[1] * 15])

    def _f(self, x):
        a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
        return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


@OBJECTIVE.register("ackley")
class Ackley(_Classic):
    """Ackley; many local minima, global min 0 at origin. Domain [-32.768, 32.768]^d."""

    lo, hi, f_min, dim_default = -32.768, 32.768, 0.0, 5

    def _f(self, x):
        d = len(x)
        s1 = (x * x).sum()
        s2 = np.cos(2 * np.pi * x).sum()
        return -20 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20 + np.e


@OBJECTIVE.register("rosenbrock")
class Rosenbrock(_Classic):
    """Rosenbrock banana valley; global min 0 at all-ones. Domain [-5,10]^d."""

    lo, hi, f_min, dim_default = -5.0, 10.0, 0.0, 4

    def _f(self, x):
        return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


@OBJECTIVE.register("rastrigin")
class Rastrigin(_Classic):
    """Rastrigin; highly multimodal regular lattice, global min 0 at origin. Domain [-5.12,5.12]^d."""

    lo, hi, f_min, dim_default = -5.12, 5.12, 0.0, 5

    def _f(self, x):
        return float(10 * len(x) + (x * x - 10 * np.cos(2 * np.pi * x)).sum())


@OBJECTIVE.register("levy")
class Levy(_Classic):
    """Levy; multimodal, global min 0 at all-ones. Domain [-10,10]^d."""

    lo, hi, f_min, dim_default = -10.0, 10.0, 0.0, 5

    def _f(self, x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        wm = w[:-1]
        mid = ((wm - 1) ** 2 * (1 + 10 * np.sin(np.pi * wm + 1) ** 2)).sum()
        return float(term1 + mid + term3)


@OBJECTIVE.register("griewank")
class Griewank(_Classic):
    """Griewank; many local minima, global min 0 at origin. Domain [-600,600]^d."""

    lo, hi, f_min, dim_default = -600.0, 600.0, 0.0, 5

    def _f(self, x):
        s = (x * x).sum() / 4000
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return float(s - p + 1)


_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array(
    [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
)
_H6_A = np.array(
    [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
)
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)
_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


@OBJECTIVE.register("hartmann3")
class Hartmann3(_Classic):
    """Hartmann-3 (a top BO benchmark); f_min=-3.86278. Domain [0,1]^3 (so the cube IS the domain)."""

    lo, hi, f_min, dim_default = 0.0, 1.0, -3.86278, 3

    def _f(self, x):
        inner = (_H3_A * (x[None, :] - _H3_P) ** 2).sum(1)
        return float(-(_H_ALPHA * np.exp(-inner)).sum())


@OBJECTIVE.register("hartmann6")
class Hartmann6(_Classic):
    """Hartmann-6 (THE canonical 6-D BO benchmark); f_min=-3.32237. Domain [0,1]^6."""

    lo, hi, f_min, dim_default = 0.0, 1.0, -3.32237, 6

    def _f(self, x):
        inner = (_H6_A * (x[None, :] - _H6_P) ** 2).sum(1)
        return float(-(_H_ALPHA * np.exp(-inner)).sum())


@OBJECTIVE.register("styblinski_tang")
class StyblinskiTang(_Classic):
    """Styblinski-Tang; global min -39.16599*d at x_i=-2.903534. Domain [-5,5]^d."""

    lo, hi, dim_default = -5.0, 5.0, 5

    def __init__(self, dim: int | None = None, **kw) -> None:
        d = dim or self.dim_default
        self.f_min = -39.166166 * d
        super().__init__(dim=d, **kw)

    def _f(self, x):
        return float(0.5 * (x**4 - 16 * x * x + 5 * x).sum())


@OBJECTIVE.register("dixon_price")
class DixonPrice(_Classic):
    """Dixon-Price valley; global min 0. Domain [-10,10]^d."""

    lo, hi, f_min, dim_default = -10.0, 10.0, 0.0, 5

    def _f(self, x):
        i = np.arange(2, len(x) + 1)
        return float((x[0] - 1) ** 2 + (i * (2 * x[1:] ** 2 - x[:-1]) ** 2).sum())


@OBJECTIVE.register("powell")
class Powell(_Classic):
    """Powell singular function over groups of 4 coordinates; global min 0. Domain [-4,5]^d."""

    lo, hi, f_min, dim_default = -4.0, 5.0, 0.0, 8

    def _f(self, x):
        n4 = (len(x) // 4) * 4
        v = 0.0
        for j in range(0, n4, 4):
            a, b, c, d = x[j], x[j + 1], x[j + 2], x[j + 3]
            v += (a + 10 * b) ** 2 + 5 * (c - d) ** 2 + (b - 2 * c) ** 4 + 10 * (a - d) ** 4
        return float(v)
