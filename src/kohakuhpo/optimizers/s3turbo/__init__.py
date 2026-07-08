"""S3-TuRBO package: importing it registers the optimizer, mask laws and scout strategies."""

from kohakuhpo.optimizers.s3turbo.masks import MASK_ALIASES, mask_dense, mask_hard, mask_soft
from kohakuhpo.optimizers.s3turbo.optimizer import PRESETS, S3Turbo
from kohakuhpo.optimizers.s3turbo.regions import Region
from kohakuhpo.optimizers.s3turbo.scouts import (
    SCOUT_ALIASES,
    NoScout,
    RandomScout,
    ScoutStrategy,
    SidecarScout,
    SwitchScout,
)

__all__ = [
    "S3Turbo",
    "PRESETS",
    "Region",
    "ScoutStrategy",
    "NoScout",
    "RandomScout",
    "SidecarScout",
    "SwitchScout",
    "MASK_ALIASES",
    "SCOUT_ALIASES",
    "mask_dense",
    "mask_hard",
    "mask_soft",
]
