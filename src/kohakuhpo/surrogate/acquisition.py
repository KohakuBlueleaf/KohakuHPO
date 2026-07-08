"""Acquisition functions over a GP posterior, for minimization.

Each takes posterior ``(mean, std)`` at candidate points and returns a MAXIMIZED score. Registered
in :data:`~kohakuhpo.registry.ACQUISITION` so GP-based optimizers can select one by name.
"""

import torch

from kohakuhpo.registry import ACQUISITION


@ACQUISITION.register("log_ei")
def log_ei(mean: torch.Tensor, std: torch.Tensor, best: float) -> torch.Tensor:
    """Log Expected Improvement ``log E[max(best - f, 0)]``, numerically stabilized."""
    z = (best - mean) / std
    normal = torch.distributions.Normal(0.0, 1.0)
    ei = std * (z * normal.cdf(z) + torch.exp(normal.log_prob(z)))
    return torch.log(ei.clamp_min(1e-12))


@ACQUISITION.register("pi")
def prob_improve(mean: torch.Tensor, std: torch.Tensor, best: float) -> torch.Tensor:
    """Probability of improvement ``P(f < best)``."""
    return torch.distributions.Normal(0.0, 1.0).cdf((best - mean) / std)


@ACQUISITION.register("ucb")
def upper_conf(mean: torch.Tensor, std: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Lower confidence bound ``-(mean - beta std)``, returned as a maximized score."""
    return -(mean - beta * std)
