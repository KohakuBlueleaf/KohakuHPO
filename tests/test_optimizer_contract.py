"""Every registered optimizer honors the ask/tell contract and runs a short loop cleanly."""

import numpy as np
import pytest

from kohakuhpo import OPTIMIZER, SearchSpace, build

DIM = 4
BUDGET = 24
Q = 4


def sphere(u: np.ndarray) -> float:
    return float(((u - 0.3) ** 2).sum())


@pytest.mark.parametrize("name", OPTIMIZER.keys())
def test_ask_tell_contract(name):
    space = SearchSpace.from_dim(DIM)
    opt = build(name, OPTIMIZER, space=space, seed=0)
    while len(opt.y) < BUDGET:
        U = opt.ask(Q)
        assert U.shape == (Q, DIM)
        assert np.all(U >= 0.0) and np.all(U <= 1.0)
        opt.tell(U, np.array([sphere(u) for u in U]))
    assert len(opt.y) == BUDGET
    u_best, y_best = opt.best
    assert np.isfinite(y_best) and u_best.shape == (DIM,)


@pytest.mark.parametrize("name", ["random", "sobol", "cmaes"])
def test_seed_determinism(name):
    space = SearchSpace.from_dim(DIM)
    a = build(name, OPTIMIZER, space=space, seed=7).ask(8)
    b = build(name, OPTIMIZER, space=space, seed=7).ask(8)
    assert np.allclose(a, b)


def test_variable_batch_size():
    space = SearchSpace.from_dim(DIM)
    opt = build("s3turbo", OPTIMIZER, space=space, seed=0)
    for q in (1, 4, 2, 3):
        U = opt.ask(q)
        assert U.shape == (q, DIM)
        opt.tell(U, np.array([sphere(u) for u in U]))
