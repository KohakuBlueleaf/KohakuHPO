"""The many-basin benchmark family S3-TuRBO's scout axis is built for.

The start basin at ``x0`` is shallow and easy; the deeper basins sit *far away across a high plateau*,
so a local trust region cannot walk to them (between basins the value is bad, so a growing box is
punished and shrinks back). Reaching a deep basin requires a jump, which is exactly what a scout
supplies. A few of the far basins hide the deepest cores. Returns regret ``f(x) - f_min >= 0``.

This is deliberately barrier-separated: an earlier version used wide overlapping wells, which formed
one connected bowl a local optimizer could drift across, so the scout was never needed and a plain
trust region matched it. Here the plateau between basins makes the escape real, and only a scout
(``sidecar``/``switch``) crosses it.
"""

from dataclasses import dataclass

import numpy as np

from kohakuhpo.registry import OBJECTIVE
from kohakuhpo.space import SearchSpace


@dataclass
class Basin:
    center: np.ndarray
    depth: float
    sigma: float
    kind: str  # "global" (a deep core) | "local"


@OBJECTIVE.register("many_basin")
class ManyBasin:
    """Gaussian-well mixture on a high plateau. ``n_basins`` wells of moderate width sit far apart;
    ``n_global`` of them are deep cores. The value is ``plateau - max_i bump_i``, so between wells it
    is ~plateau (bad) and each well is a separate attraction region reachable only by landing in it.
    ``x0`` starts in a shallow near well; the deep cores are a jump away."""

    def __init__(
        self,
        dim: int = 25,
        seed: int = 0,
        n_basins: int = 8,
        n_global: int = 2,
        sigma: float = 0.2,
        near_depth: float = 0.5,
        deep_depth: float = 1.0,
        local_depth_spread: float = 0.1,
        plateau: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.seed = seed
        self.space = SearchSpace.from_dim(dim)

        # start basin near one corner; every other basin in the far half, so a plateau separates them
        axis = rng.random(dim)
        centers = rng.random((n_basins, dim))
        centers[0] = axis * 0.18
        for i in range(1, n_basins):
            centers[i] = np.where(axis < 0.5, 0.55 + 0.45 * rng.random(dim), 0.45 * rng.random(dim))
        # deep cores: the n_global far basins farthest from the start
        far_order = np.argsort(((centers - centers[0][None]) ** 2).mean(1))[::-1]
        global_idx = [int(i) for i in far_order[:n_global]]
        global_set = set(global_idx)

        self.basins: list[Basin] = []
        for i, c in enumerate(centers):
            if i == 0:
                depth, kind = near_depth, "local"  # shallow, easy start basin
            elif i in global_set:
                depth, kind = deep_depth, "global"  # deep core, far away
            else:
                depth = near_depth + float(rng.uniform(0.0, local_depth_spread))
                kind = "local"
            self.basins.append(
                Basin(center=c, depth=depth, sigma=float(sigma * rng.uniform(0.9, 1.1)), kind=kind)
            )

        self.global_idx = np.array(global_idx, dtype=int)
        self.x0_basin = 0
        self.plateau = float(plateau)
        self.x0u = np.clip(centers[0] + rng.normal(0.0, sigma * 0.25, dim), 0.0, 1.0)
        self.x0 = self.space.to_config(self.x0u)
        self.optimum = plateau - max(b.depth for b in self.basins)  # bottom of the deepest core

    def _values(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        best_bump = np.zeros(len(U))
        for b in self.basins:
            rms2 = ((U - b.center[None]) ** 2).mean(axis=1)
            best_bump = np.maximum(best_bump, b.depth * np.exp(-rms2 / (2.0 * b.sigma**2)))
        return self.plateau - best_bump - self.optimum

    def __call__(self, cfg: dict) -> float:
        u = self.space.to_unit(cfg)
        return float(self._values(u[None])[0])

    def nearest_basin(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index of and RMS distance to the nearest basin center, per row of ``U``."""
        U = np.atleast_2d(U)
        centers = np.array([b.center for b in self.basins])
        rms = np.sqrt(((U[:, None, :] - centers[None]) ** 2).mean(axis=2))
        idx = rms.argmin(axis=1)
        return idx, rms[np.arange(len(U)), idx]

    def basin_kind(self, idx: np.ndarray) -> list[str]:
        return [self.basins[int(i)].kind for i in np.asarray(idx).reshape(-1)]

    def in_core(self, U: np.ndarray) -> np.ndarray:
        """True where a point sits inside a deep global core (within 1.5 sigma of its center)."""
        U = np.atleast_2d(U)
        out = np.zeros(len(U), dtype=bool)
        for gi in self.global_idx:
            b = self.basins[int(gi)]
            rms = np.sqrt(((U - b.center[None]) ** 2).mean(axis=1))
            out |= rms < 1.5 * b.sigma
        return out
