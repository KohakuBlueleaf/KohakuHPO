"""Search space: typed parameters <-> a normalized ``[0, 1]^d`` cube.

Optimizers work purely in the unit cube; the space is the one place that knows parameter types.
Each parameter kind is a codec in the :data:`~kohakuhpo.registry.PARAM` registry with the contract
``decode(u: float) -> value`` / ``encode(value) -> u in [0,1]``.

Built-in tuple specs:

* ``("float", lo, hi)``:     linear continuous
* ``("log", lo, hi)``:       log-scaled continuous (lo, hi > 0)
* ``("int", lo, hi)``:       integer, rounded from a continuous axis
* ``("cat", [choices...])``: categorical; one axis binned to a choice

A spec may also be a ``{"name": <kind|dotted.path>, **kw}`` dict or an already-built codec
instance (anything with ``decode``/``encode``).
"""

import numpy as np

from kohakuhpo.registry import PARAM, build


@PARAM.register("float")
class FloatParam:
    """Linear continuous parameter on ``[lo, hi]``."""

    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(hi)

    def decode(self, u: float) -> float:
        return self.lo + u * (self.hi - self.lo)

    def encode(self, v) -> float:
        return (float(v) - self.lo) / (self.hi - self.lo)


@PARAM.register("log")
class LogParam:
    """Log-scaled continuous parameter on ``[lo, hi]``, ``lo, hi > 0``."""

    def __init__(self, lo: float, hi: float) -> None:
        if lo <= 0 or hi <= 0:
            raise ValueError(f"log param needs positive bounds, got ({lo}, {hi})")
        self.lo, self.hi = float(np.log(lo)), float(np.log(hi))

    def decode(self, u: float) -> float:
        return float(np.exp(self.lo + u * (self.hi - self.lo)))

    def encode(self, v) -> float:
        return (float(np.log(v)) - self.lo) / (self.hi - self.lo)


@PARAM.register("int")
class IntParam:
    """Integer parameter on ``[lo, hi]``, rounded from a continuous axis."""

    def __init__(self, lo: int, hi: int) -> None:
        self.lo, self.hi = int(lo), int(hi)

    def decode(self, u: float) -> int:
        return int(round(self.lo + u * (self.hi - self.lo)))

    def encode(self, v) -> float:
        return (int(v) - self.lo) / max(self.hi - self.lo, 1)


@PARAM.register("cat")
class CatParam:
    """Categorical parameter: one axis binned into ``len(choices)`` cells."""

    def __init__(self, choices) -> None:
        self.choices = list(choices)

    def decode(self, u: float):
        idx = min(int(u * len(self.choices)), len(self.choices) - 1)
        return self.choices[idx]

    def encode(self, v) -> float:
        return (self.choices.index(v) + 0.5) / len(self.choices)


def _build_codec(spec):
    """Resolve one parameter spec (tuple / dict / codec instance) to a codec object."""
    if isinstance(spec, tuple | list):
        kind, *args = spec
        return PARAM.get(kind)(*args)
    if isinstance(spec, dict):
        return build(spec, PARAM)
    if hasattr(spec, "decode") and hasattr(spec, "encode"):
        return spec
    raise ValueError(f"cannot interpret parameter spec {spec!r}")


class SearchSpace:
    """An ordered ``name -> codec`` map; converts between config dicts and unit points."""

    def __init__(self, params: dict) -> None:
        self.names = list(params)
        self.codecs = [_build_codec(params[n]) for n in self.names]
        self.dim = len(self.names)

    @classmethod
    def from_dim(cls, dim: int) -> "SearchSpace":
        """A raw ``[0,1]^dim`` cube with identity params ``x0..x{dim-1}``."""
        return cls({f"x{i}": ("float", 0.0, 1.0) for i in range(dim)})

    def to_config(self, u: np.ndarray) -> dict:
        """Unit point ``u (d,)`` -> typed config dict."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return {
            name: codec.decode(float(ui))
            for name, codec, ui in zip(self.names, self.codecs, u, strict=True)
        }

    def to_unit(self, cfg: dict) -> np.ndarray:
        """Typed config dict -> unit point ``u (d,)``."""
        u = np.array(
            [codec.encode(cfg[name]) for name, codec in zip(self.names, self.codecs, strict=True)]
        )
        return np.clip(u, 0.0, 1.0)

    def to_configs(self, U: np.ndarray) -> list[dict]:
        """Unit points ``(n, d)`` -> list of config dicts."""
        return [self.to_config(u) for u in np.atleast_2d(U)]

    def to_units(self, cfgs: list[dict]) -> np.ndarray:
        """List of config dicts -> unit points ``(n, d)``."""
        return np.stack([self.to_unit(c) for c in cfgs])

    def __repr__(self) -> str:
        kinds = ", ".join(
            f"{n}: {type(c).__name__}" for n, c in zip(self.names, self.codecs, strict=True)
        )
        return f"SearchSpace(d={self.dim}; {kinds})"
