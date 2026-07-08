"""CMA-ES (Hansen & Ostermeier) in the unit cube, via the pure-python ``cmaes`` library.

Runs with mean 0.5 and the given ``sigma``; ``ask`` drains points one at a time and ``tell``
forwards complete generations (``popsize`` blocks) to the underlying sampler.
"""

import numpy as np
from cmaes import CMA

from kohakuhpo.optimizer import Optimizer
from kohakuhpo.registry import OPTIMIZER


@OPTIMIZER.register("cmaes")
class CMAESOpt(Optimizer):
    """Ask/tell wrapper over :class:`cmaes.CMA` bounded to ``[0,1]^d``."""

    def __init__(
        self, space, seed: int = 0, sigma: float = 0.2, popsize: int | None = None
    ) -> None:
        super().__init__(space, seed)
        self._es = CMA(
            mean=np.full(self.dim, 0.5),
            sigma=sigma,
            bounds=np.tile([0.0, 1.0], (self.dim, 1)),
            seed=seed,
            population_size=popsize,
        )
        self._buffer: list[tuple[np.ndarray, float]] = []

    def _ask(self, q: int) -> np.ndarray:
        return np.array([self._es.ask() for _ in range(q)])

    def tell(self, U: np.ndarray, y: np.ndarray) -> None:
        super().tell(U, y)
        U = np.atleast_2d(np.asarray(U, float))
        y = np.asarray(y, float).reshape(-1)
        self._buffer.extend(zip(U, y, strict=True))
        while len(self._buffer) >= self._es.population_size:
            gen = self._buffer[: self._es.population_size]
            self._buffer = self._buffer[self._es.population_size :]
            self._es.tell([(u, float(v)) for u, v in gen])
