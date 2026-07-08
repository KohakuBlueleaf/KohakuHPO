"""HEBO-style optimizer: warped-GP surrogate + rank-combined multi-acquisition, from scratch.

Three ingredients reassembled from the published method: a Kumaraswamy input-warped GP (handles
non-stationarity), a signed-log output warp (handles skew and penalty cliffs), and a robust
acquisition that rank-combines log-EI, PI and UCB over a Sobol pool so a pick must score well
under all three. Batches use constant-liar refits. Captures HEBO's algorithmic essence; not
bit-identical to the published implementation.
"""

import numpy as np
import torch
from scipy.stats import qmc

from kohakuhpo.device import tensor_kw
from kohakuhpo.optimizer import Optimizer
from kohakuhpo.registry import OPTIMIZER
from kohakuhpo.surrogate import GP, log_ei, output_warp, prob_improve, upper_conf


@OPTIMIZER.register("hebo")
class HEBO(Optimizer):
    """Input+output-warped GP with rank-combined EI/PI/UCB acquisition."""

    def __init__(
        self,
        space,
        seed: int = 0,
        n_init: int = 8,
        pool: int = 2048,
        beta: float = 2.0,
        max_data: int = 128,
    ) -> None:
        super().__init__(space, seed)
        self.n_init, self.pool, self.beta, self.max_data = n_init, pool, beta, max_data
        self._sobol = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def _rank(self, v: torch.Tensor) -> np.ndarray:
        """Competition rank of a maximized score, normalized to [0,1] (0 = best)."""
        order = torch.argsort(v, descending=True).cpu().numpy()
        r = np.empty(len(order))
        r[order] = np.arange(len(order))
        return r / max(len(order) - 1, 1)

    def _ask(self, q: int) -> np.ndarray:
        if len(self.y) < self.n_init:
            return self._sobol.random(q)
        tu, ty = self.train_set(self.max_data)
        x = torch.tensor(tu, **tensor_kw())
        yw = torch.tensor(output_warp(ty), **tensor_kw())
        picks = []
        best = float(yw.min())
        for _ in range(q):
            gp = GP(x, yw, warp_input=True)
            cand = torch.tensor(self._sobol.random(self.pool), **tensor_kw())
            mean, std = gp.predict(cand)
            score = (
                self._rank(log_ei(mean, std, best))
                + self._rank(prob_improve(mean, std, best))
                + self._rank(upper_conf(mean, std, self.beta))
            )
            j = int(np.argmin(score))
            u = cand[j]
            picks.append(u.cpu().numpy())
            x = torch.cat([x, u[None]], dim=0)
            yw = torch.cat([yw, torch.tensor([best], **tensor_kw())])
        return np.array(picks)
