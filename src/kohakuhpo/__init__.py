"""KohakuHPO: sample-efficient black-box optimization (HPO and beyond).

One ask/tell contract in a normalized ``[0,1]^d`` cube; a :class:`SearchSpace` codec owns the
parameter types; a :class:`Study` owns a run and exposes ask/tell, an iterator, and a closure
driver over the same state. Optimizers, parameter kinds, acquisitions, and S3-TuRBO's mask/scout
axes are all registries, so user extensions are a decorator (or a dotted path) away.

Importing the package registers all built-in optimizers and benchmark objectives.
"""

from kohakuhpo import benchmarks, optimizers  # noqa: F401  (register built-ins)
from kohakuhpo.device import use_device
from kohakuhpo.optimizer import Optimizer
from kohakuhpo.optimizers.s3turbo import PRESETS, S3Turbo, ScoutStrategy
from kohakuhpo.registry import (
    ACQUISITION,
    MASK,
    OBJECTIVE,
    OPTIMIZER,
    PARAM,
    SCOUT,
    Registry,
    build,
)
from kohakuhpo.run import maximize, minimize
from kohakuhpo.space import SearchSpace
from kohakuhpo.study import Batch, Result, Study, Trial
from kohakuhpo.surrogate import GP

__all__ = [
    "SearchSpace",
    "Optimizer",
    "Study",
    "Trial",
    "Batch",
    "Result",
    "minimize",
    "maximize",
    "use_device",
    "Registry",
    "build",
    "OPTIMIZER",
    "OBJECTIVE",
    "PARAM",
    "ACQUISITION",
    "MASK",
    "SCOUT",
    "S3Turbo",
    "PRESETS",
    "ScoutStrategy",
    "GP",
]

__version__ = "0.1.0"
