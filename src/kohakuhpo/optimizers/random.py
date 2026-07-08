"""Random and Sobol baselines: the sanity floors every method must clear."""

import numpy as np
from scipy.stats import qmc

from kohakuhpo.optimizer import Optimizer
from kohakuhpo.registry import OPTIMIZER


@OPTIMIZER.register("random")
class RandomSearch(Optimizer):
    """Uniform random points in the cube (ignores observations)."""

    def _ask(self, q: int) -> np.ndarray:
        return self.rng.random((q, self.dim))


@OPTIMIZER.register("sobol")
class SobolSearch(Optimizer):
    """Scrambled Sobol sequence: low-discrepancy cube coverage."""

    def __init__(self, space, seed: int = 0) -> None:
        super().__init__(space, seed)
        self._engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def _ask(self, q: int) -> np.ndarray:
        return self._engine.random(q)
