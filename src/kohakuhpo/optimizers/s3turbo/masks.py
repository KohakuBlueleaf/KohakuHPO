"""The local-move axis: coordinate-mask laws in the :data:`~kohakuhpo.registry.MASK` registry.

Contract: ``mask(rng, n, dim, rho) -> alpha in [0,1]^(n, dim)``. A candidate is then
``x = center + alpha * (raw - center)``, so ``alpha=1`` takes the box sample and ``alpha=0``
keeps the center. All laws have expected active mass ``rho * dim`` per row. Register a new law
to extend S3-TuRBO's move axis without touching the optimizer.
"""

import numpy as np

from kohakuhpo.registry import MASK


@MASK.register("dense")
def mask_dense(rng, n, dim, rho):
    """Every coordinate takes the full box sample (the plain TuRBO move)."""
    return np.ones((n, dim))


@MASK.register("hard")
def mask_hard(rng, n, dim, rho):
    """Bernoulli(rho) 0/1 mask: each coordinate fully moved or frozen; at least one active."""
    p = float(np.clip(rho, 1.0 / dim, 1.0))
    m = (rng.random((n, dim)) < p).astype(float)
    empty = np.where(m.sum(1) == 0)[0]
    if len(empty):
        m[empty, rng.integers(0, dim, size=len(empty))] = 1.0
    return m


@MASK.register("soft")
def mask_soft(rng, n, dim, rho, concentration=0.4):
    """Polarized ``Beta(rho c0, (1-rho) c0)`` weights (c0 < 1), renormalized to mass ``rho dim``.

    Most weight sits near 0 or 1 while allowing graded partial moves; ``c0 -> 0`` recovers the
    hard Bernoulli law and ``rho = 1`` recovers dense.
    """
    c0 = max(float(concentration), 1e-3)
    a = max(float(rho) * c0, 1e-3)
    b = max((1.0 - float(rho)) * c0, 1e-3)
    m = rng.beta(a, b, size=(n, dim))
    target = float(rho) * dim
    return np.clip(m * (target / np.maximum(m.sum(1, keepdims=True), 1e-12)), 0.0, 1.0)


MASK_ALIASES = {"dense": "dense", "none": "dense", "soft": "soft", "soft_beta": "soft",
                "beta": "soft", "hard": "hard", "bernoulli": "hard", "sparse": "hard"}  # fmt: skip
