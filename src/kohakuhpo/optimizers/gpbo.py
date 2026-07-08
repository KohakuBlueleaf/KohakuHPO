"""GP-BO: global exact-GP surrogate + log-EI acquisition, batched by constant-liar.

Sobol points until ``n_init`` observations exist; afterwards fit the GP, maximize the acquisition
over a Sobol candidate pool, and build a batch of ``q`` by refitting with a pessimistic fantasy
score after each pick (constant-liar batch diversity). The acquisition is an
:data:`~kohakuhpo.registry.ACQUISITION` spec (default ``"log_ei"``).
"""

import numpy as np
import torch
from scipy.stats import qmc

from kohakuhpo.device import tensor_kw
from kohakuhpo.optimizer import Optimizer
from kohakuhpo.registry import ACQUISITION, OPTIMIZER
from kohakuhpo.surrogate import GP


@OPTIMIZER.register("gpbo")
class GPBO(Optimizer):
    """Global GP + acquisition-over-pool with constant-liar batching."""

    def __init__(
        self,
        space,
        seed: int = 0,
        n_init: int = 8,
        pool: int = 2048,
        max_data: int = 128,
        acquisition: str = "log_ei",
    ) -> None:
        super().__init__(space, seed)
        self.n_init = n_init
        self.pool = pool
        self.max_data = max_data
        self.acquisition = ACQUISITION.get(acquisition)
        self._sobol = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def _ask(self, q: int) -> np.ndarray:
        if len(self.y) < self.n_init:
            return self._sobol.random(q)
        tu, ty = self.train_set(self.max_data)
        x = torch.tensor(tu, **tensor_kw())
        y = torch.tensor(ty, **tensor_kw())
        picks = []
        best = float(self.y.min())
        for _ in range(q):
            gp = GP(x, y)
            cand = torch.tensor(self._sobol.random(self.pool), **tensor_kw())
            mean, std = gp.predict(cand)
            j = int(torch.argmax(self.acquisition(mean, std, best)))
            u = cand[j]
            picks.append(u.cpu().numpy())
            x = torch.cat([x, u[None]], dim=0)
            y = torch.cat([y, torch.tensor([best], **tensor_kw())])
        return np.array(picks)
