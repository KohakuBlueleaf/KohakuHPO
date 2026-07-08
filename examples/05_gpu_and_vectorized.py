"""GPU + batched objectives.

Two independent levers: use_device() moves the optimizer's own GP work (Cholesky, L-BFGS,
posterior draws) to a GPU; vectorized=True hands the whole batch to your objective in one call,
which is the natural shape for objectives that are themselves batched torch code. For fully
cube-native workflows, skip config dicts and drive the optimizer directly with arrays.
"""

import numpy as np
import torch

import kohakuhpo as khpo

if torch.cuda.is_available():
    khpo.use_device("cuda:0")  # GP surrogate math now runs on the GPU (float32 + safe jitter)

space = khpo.SearchSpace.from_dim(16)


def batched_objective(configs: list[dict]) -> np.ndarray:
    """Evaluate a whole batch at once, e.g. one forward pass of a simulator."""
    U = torch.tensor(space.to_units(configs))
    return ((U - 0.5) ** 2).sum(dim=1).cpu().numpy()


result = khpo.minimize(
    batched_objective, space, "s3turbo", budget=120, q=8, seed=0, vectorized=True
)
print("vectorized best:", result.best_value)

# Raw cube mode: arrays in, arrays out, no dicts anywhere.
opt = khpo.build("s3turbo", khpo.OPTIMIZER, space=khpo.SearchSpace.from_dim(16), seed=0)
for _ in range(15):
    U = opt.ask(8)
    y = ((torch.tensor(U) - 0.5) ** 2).sum(dim=1).cpu().numpy()
    opt.tell(U, y)
print("raw-cube best  :", opt.best[1])
