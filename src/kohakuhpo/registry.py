"""Build-time component dispatch: named registries + a dotted-path escape hatch.

The framework follows an *annotate -> label -> select* philosophy: configuration selects a concrete
class or callable **once**, at build time, via :func:`build`. The selected object is then held as a
plain attribute and called directly in the hot loop; there is no per-step ``if mode == ...``
dispatch anywhere.

A *spec* accepted by :func:`build` is one of:

* an already-built instance               -> returned unchanged
* a class                                 -> instantiated with the build-site kwargs
* a registry key string ``"s3turbo"``     -> looked up in ``registry``, then called
* a dotted path ``"pkg.module.Cls"``      -> imported, then called
* a dict ``{"name": <key|path>, **kw}``   -> resolved, called with ``kw`` + kwargs

One registry exists per swappable axis (see the bottom of this module). Built-ins register into
them at import time; user code registers its own entries or passes a dotted path / class straight
to ``build``; extending the framework never requires editing it.
"""

import importlib
from collections.abc import Callable
from typing import Any


def import_object(path: str) -> Any:
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


class Registry:
    """A named ``str -> class/callable`` map with a decorator-based ``register``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None) -> Callable[[Callable], Callable]:
        """Return a decorator registering the wrapped object under ``key``.

        ``key`` defaults to the object's ``__name__``.
        """

        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            name = key or obj.__name__
            if name in self._items:
                raise KeyError(f"{self.name!r} already has an entry named {name!r}")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            raise KeyError(f"unknown {self.name} {key!r}; registered: {self.keys()}")
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items)


# One registry per swappable axis. MASK and SCOUT are S3-TuRBO's two internal axes.
OPTIMIZER = Registry("optimizer")
OBJECTIVE = Registry("objective")
PARAM = Registry("param")
ACQUISITION = Registry("acquisition")
MASK = Registry("mask")
SCOUT = Registry("scout")


def _resolve(name: str, registry: Registry | None) -> Callable[..., Any]:
    if "." in name:
        return import_object(name)
    if registry is None:
        raise ValueError(f"{name!r} is not a dotted path and no registry was provided")
    return registry.get(name)


def build(spec: Any, registry: Registry | None = None, **kwargs: Any) -> Any:
    """Resolve ``spec`` to a concrete object, passing ``kwargs`` at construction.

    Build-time only; never call this inside an ask/tell loop.
    """
    match spec:
        case None:
            return None
        case dict():
            opts = dict(spec)
            name = opts.pop("name")
            return _resolve(name, registry)(**opts, **kwargs)
        case str():
            return _resolve(spec, registry)(**kwargs)
        case type():
            return spec(**kwargs)
        case _:
            return spec
