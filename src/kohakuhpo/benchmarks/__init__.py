"""Importing this package registers every built-in benchmark objective in ``OBJECTIVE``."""

from kohakuhpo.benchmarks import classic  # noqa: F401  (register on import)
from kohakuhpo.benchmarks.manybasin import ManyBasin

__all__ = ["ManyBasin", "classic"]
