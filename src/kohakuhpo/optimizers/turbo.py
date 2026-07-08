"""TuRBO (Eriksson et al. 2019): trust-region BO with batch Thompson sampling.

A local GP over a box trust region centered on the incumbent; the batch is ``q`` joint posterior
draws over a Sobol pool restricted to the region, argmin each (TuRBO's default batch acquisition, so
Thompson sampling is intrinsic to the method). The region side grows after ``succ_tol`` consecutive
improvements and halves after ``fail_tol`` failures. ``warp=True`` adds HEBO's input/output warps to
the local GP (``warped_turbo``).
"""

import numpy as np
import torch
from scipy.stats import qmc

from kohakuhpo.device import tensor_kw
from kohakuhpo.optimizer import Optimizer
from kohakuhpo.registry import OPTIMIZER
from kohakuhpo.surrogate import GP, output_warp


@OPTIMIZER.register("turbo")
class Turbo(Optimizer):
    """Single trust region + local GP + discretized batch Thompson sampling."""

    def __init__(
        self,
        space,
        seed: int = 0,
        n_init: int = 8,
        pool: int = 512,
        max_data: int = 128,
        l_init: float = 0.4,
        l_min: float = 0.02,
        l_max: float = 1.6,
        succ_tol: int = 3,
        fail_tol: int = 4,
        warp: bool = False,
    ) -> None:
        super().__init__(space, seed)
        self.n_init, self.pool, self.max_data = n_init, pool, max_data
        self.l = l_init
        self.l_init, self.l_min, self.l_max = l_init, l_min, l_max
        self.succ_tol, self.fail_tol = succ_tol, fail_tol
        self.warp = warp
        self._succ = self._fail = 0
        self._prev_best = float("inf")
        self._sobol = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        torch.manual_seed(seed)  # seeds every device; posterior draws use the global RNG

    def tell(self, U, y) -> None:
        super().tell(U, y)
        best = float(self.y.min())
        improved = best < self._prev_best - 1e-9
        self._succ = self._succ + 1 if improved else 0
        self._fail = 0 if improved else self._fail + 1
        self._prev_best = best
        if self._succ >= self.succ_tol:
            self.l, self._succ = min(self.l * 2.0, self.l_max), 0
        elif self._fail >= self.fail_tol:
            self.l, self._fail = max(self.l / 2.0, self.l_min), 0

    def _ask(self, q: int) -> np.ndarray:
        if len(self.y) < self.n_init:
            return self._sobol.random(q)
        center = self.best[0]
        tu, ty = self.train_set(self.max_data)
        if self.warp:
            ty = output_warp(ty)
        gp = GP(
            torch.tensor(tu, **tensor_kw()),
            torch.tensor(ty, **tensor_kw()),
            warp_input=self.warp,
        )
        raw = self._sobol.random(self.pool)
        lo = np.clip(center - self.l / 2, 0.0, 1.0)
        hi = np.clip(center + self.l / 2, 0.0, 1.0)
        cand = lo + raw * (hi - lo)
        draws = gp.sample(torch.tensor(cand, **tensor_kw()), q)
        idx = torch.argmin(draws, dim=1).cpu().numpy()
        return cand[idx]


@OPTIMIZER.register("warped_turbo")
class WarpedTurbo(Turbo):
    """TuRBO with input + output warping on by default, for non-stationary or skewed targets."""

    def __init__(self, space, seed: int = 0, **kw) -> None:
        kw.setdefault("warp", True)
        super().__init__(space, seed=seed, **kw)
