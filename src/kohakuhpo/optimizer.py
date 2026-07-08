"""The cube-level ask/tell contract every optimizer implements.

An optimizer proposes and receives points purely in the space's ``[0,1]^d`` cube and always
minimizes; types, direction, budgets and parallelism live in :class:`~kohakuhpo.study.Study`.

    opt.tell(U, y)        # register evaluated unit points U (m, d) and scores y (m,)
    U_next = opt.ask(q)   # propose q new unit points (q, d)

Subclasses implement ``_ask``; ``tell`` may be extended but must call ``super().tell``.
"""

import numpy as np

from kohakuhpo.space import SearchSpace


class Optimizer:
    """Ask/tell base: accumulates observations ``(U, y)``; subclasses implement ``_ask``."""

    def __init__(self, space: SearchSpace, seed: int = 0) -> None:
        self.space = space
        self.dim = space.dim
        self.rng = np.random.default_rng(seed)
        self.U = np.empty((0, self.dim))
        self.y = np.empty((0,))

    def tell(self, U: np.ndarray, y: np.ndarray) -> None:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        self.U = np.concatenate([self.U, U], axis=0)
        self.y = np.concatenate([self.y, y], axis=0)

    def ask(self, q: int = 1) -> np.ndarray:
        """Propose ``q`` unit points ``(q, d)``, clipped to the cube."""
        return np.clip(self._ask(q), 0.0, 1.0)

    def _ask(self, q: int) -> np.ndarray:
        raise NotImplementedError

    @property
    def best(self) -> tuple[np.ndarray | None, float]:
        """The incumbent ``(u*, y*)``; ``(None, inf)`` before any observation."""
        if len(self.y) == 0:
            return None, float("inf")
        i = int(np.argmin(self.y))
        return self.U[i], float(self.y[i])

    def train_set(self, max_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Observations for a surrogate fit, capped to the ``max_n`` best points (an exact GP is
        O(n^3), so long runs subselect toward the good region)."""
        if len(self.y) <= max_n:
            return self.U, self.y
        idx = np.argpartition(self.y, max_n)[:max_n]
        return self.U[idx], self.y[idx]
