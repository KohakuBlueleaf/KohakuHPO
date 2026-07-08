"""Importing this package registers every built-in optimizer in ``OPTIMIZER``."""

from kohakuhpo.optimizers import (  # noqa: F401  (register on import)
    cmaes,
    gpbo,
    hebo,
    random,
    s3turbo,
    turbo,
)
from kohakuhpo.optimizers.s3turbo import PRESETS, S3Turbo

__all__ = ["S3Turbo", "PRESETS"]
