"""Study gates: interface equivalence, direction, x0 ordering, failures, injection, vectorized."""

import numpy as np
import pytest

from kohakuhpo import SearchSpace, Study, minimize
from kohakuhpo.run import maximize

SPACE = {"a": ("float", 0.0, 1.0), "b": ("float", 0.0, 1.0), "c": ("float", 0.0, 1.0)}


def f(cfg: dict) -> float:
    return (cfg["a"] - 0.2) ** 2 + (cfg["b"] - 0.7) ** 2 + (cfg["c"] - 0.5) ** 2


def test_interfaces_are_equivalent():
    values = {}
    study_at = Study(SearchSpace(SPACE), "sobol", seed=5)
    while study_at.n_evals < 16:
        cfgs = study_at.ask(4)
        study_at.tell(cfgs, [f(c) for c in cfgs])
    values["ask_tell"] = [t.value for t in study_at.completed]

    study_it = Study(SearchSpace(SPACE), "sobol", seed=5)
    for batch in study_it.loop(budget=16, q=4):
        batch.report([f(c) for c in batch.configs])
    values["iterator"] = [t.value for t in study_it.completed]

    values["closure"] = minimize(f, SearchSpace(SPACE), "sobol", budget=16, q=4, seed=5).history

    assert values["ask_tell"] == values["iterator"] == values["closure"]


def test_direction_max():
    res_min = minimize(f, SearchSpace(SPACE), "sobol", budget=16, q=4, seed=1)
    res_max = maximize(lambda c: -f(c), SearchSpace(SPACE), "sobol", budget=16, q=4, seed=1)
    assert res_max.best_value == pytest.approx(-res_min.best_value)
    assert res_max.best_config == res_min.best_config


def test_x0_served_first():
    x0 = {"a": 0.2, "b": 0.7, "c": 0.5}
    study = Study(SearchSpace(SPACE), "sobol", seed=0, x0=x0)
    cfgs = study.ask(3)
    assert {k: pytest.approx(v) for k, v in cfgs[0].items()} == x0
    study.tell(cfgs, [f(c) for c in cfgs])
    assert study.best_value == pytest.approx(0.0)


def test_failures_become_penalty():
    study = Study(SearchSpace(SPACE), "sobol", seed=0)
    cfgs = study.ask(3)
    study.tell(cfgs, [0.5, None, float("nan")])
    states = [t.state for t in study.completed]
    assert states == ["done", "failed", "failed"]
    assert study.best_value == pytest.approx(0.5)
    assert all(t.value == pytest.approx(1e9) for t in study.completed if t.state == "failed")


def test_exception_in_closure_is_failure():
    def bad(cfg):
        if cfg["a"] > 0.5:
            raise RuntimeError("boom")
        return f(cfg)

    res = minimize(bad, SearchSpace(SPACE), "sobol", budget=12, q=4, seed=0)
    assert np.isfinite(res.best_value)
    assert any(t.state == "failed" for t in res.trials)


def test_injected_observation():
    study = Study(SearchSpace(SPACE), "sobol", seed=0)
    study.tell({"a": 0.2, "b": 0.7, "c": 0.5}, 0.0)
    assert study.n_evals == 1 and study.best_value == pytest.approx(0.0)


def test_vectorized_objective():
    def fv(cfgs: list[dict]):
        return [f(c) for c in cfgs]

    res = minimize(fv, SearchSpace(SPACE), "sobol", budget=16, q=4, seed=2, vectorized=True)
    ref = minimize(f, SearchSpace(SPACE), "sobol", budget=16, q=4, seed=2)
    assert res.history == ref.history


def test_loop_requires_report():
    study = Study(SearchSpace(SPACE), "sobol", seed=0)
    gen = study.loop(budget=8, q=4)
    next(gen)
    with pytest.raises(RuntimeError):
        next(gen)


def test_mixed_modes_share_state():
    study = Study(SearchSpace(SPACE), "sobol", seed=3)
    cfgs = study.ask(4)
    study.tell(cfgs, [f(c) for c in cfgs])
    res = study.optimize(f, budget=12, q=4)
    assert study.n_evals == 12
    assert len(res.history) == 12


def test_best_so_far_trace():
    res = minimize(f, SearchSpace(SPACE), "random", budget=20, q=4, seed=0)
    assert len(res.best_so_far) == 20
    assert all(a >= b for a, b in zip(res.best_so_far, res.best_so_far[1:], strict=False))
    assert res.best_so_far[-1] == pytest.approx(res.best_value)


def test_progress_bar_paths():
    res = minimize(f, SearchSpace(SPACE), "sobol", budget=12, q=4, seed=0, progress=True)
    assert len(res.history) == 12
    study = Study(SearchSpace(SPACE), "sobol", seed=0)
    for batch in study.loop(budget=8, q=4, progress=True, desc="test"):
        batch.report([f(c) for c in batch.configs])
    assert study.n_evals == 8
