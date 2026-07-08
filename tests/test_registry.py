"""Registry + build dispatch: every spec form resolves to the same concrete object."""

import pytest

from kohakuhpo import OPTIMIZER, Registry, SearchSpace, build
from kohakuhpo.optimizers.random import RandomSearch


def test_build_dispatch_forms():
    space = SearchSpace.from_dim(3)
    by_name = build("random", OPTIMIZER, space=space, seed=0)
    by_dict = build({"name": "random"}, OPTIMIZER, space=space, seed=0)
    by_class = build(RandomSearch, None, space=space, seed=0)
    by_path = build("kohakuhpo.optimizers.random.RandomSearch", None, space=space, seed=0)
    assert type(by_name) is type(by_dict) is type(by_class) is type(by_path) is RandomSearch
    assert build(by_name) is by_name  # already-built passthrough
    assert build(None) is None


def test_registry_errors():
    r = Registry("thing")

    @r.register("a")
    def a():
        return 1

    with pytest.raises(KeyError):
        r.register("a")(lambda: 2)
    with pytest.raises(KeyError):
        r.get("missing")
    assert "a" in r and r.keys() == ["a"]


def test_dict_spec_forwards_kwargs():
    space = SearchSpace.from_dim(2)
    opt = build({"name": "s3turbo", "preset": "multibasin"}, OPTIMIZER, space=space, seed=1)
    assert opt.scout_name == "switch" and opt.mask == "hard" and opt.tr_update == "batch"
