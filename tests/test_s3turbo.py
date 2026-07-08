"""S3-TuRBO gates: presets, mask laws, scout behavior, tr_update, custom-axis registration."""

import numpy as np
import pytest

from kohakuhpo import MASK, OPTIMIZER, SCOUT, SearchSpace, build
from kohakuhpo.benchmarks import ManyBasin
from kohakuhpo.optimizers.s3turbo import PRESETS, S3Turbo, ScoutStrategy


def test_presets_resolve_axes():
    space = SearchSpace.from_dim(6)
    for preset, (mask, scout, tr) in PRESETS.items():
        opt = S3Turbo(space, seed=0, preset=preset)
        assert (opt.mask, opt.scout_name, opt.tr_update) == (mask, scout, tr)
    with pytest.raises(ValueError):
        S3Turbo(space, preset="nope")


def test_explicit_args_override_preset():
    space = SearchSpace.from_dim(6)
    opt = S3Turbo(space, preset="multibasin", mask_distribution="soft", tr_update="point")
    assert (opt.mask, opt.scout_name, opt.tr_update) == ("soft", "switch", "point")


def test_alias_registration():
    space = SearchSpace.from_dim(4)
    assert type(build("soft_sparse_scout_turbo", OPTIMIZER, space=space, seed=0)) is S3Turbo


@pytest.mark.parametrize("name,rho", [("dense", 0.5), ("hard", 0.3), ("soft", 0.3)])
def test_mask_mass_and_range(name, rho):
    rng = np.random.default_rng(0)
    a = MASK.get(name)(rng, 400, 25, rho)
    assert a.shape == (400, 25)
    assert np.all(a >= 0) and np.all(a <= 1)
    expected = 25.0 if name == "dense" else rho * 25
    assert abs(a.sum(1).mean() - expected) < 1.0


def test_soft_mask_limits():
    rng = np.random.default_rng(1)

    def mid_mass(c0):
        a = MASK.get("soft")(rng, 2000, 20, 0.3, concentration=c0)
        return np.mean((a > 0.05) & (a < 0.95))

    assert mid_mass(0.02) < 0.5 * mid_mass(1.0)  # c0 -> 0 polarizes toward the Bernoulli law
    dense_limit = MASK.get("soft")(rng, 50, 20, 1.0)
    assert dense_limit.mean() > 0.99 and (dense_limit > 0.9).mean() > 0.97  # rho=1 approaches dense


def test_hard_mask_never_empty():
    rng = np.random.default_rng(2)
    a = MASK.get("hard")(rng, 500, 30, 0.02)
    assert (a.sum(1) >= 1).all()


def test_default_is_adaptive_mask():
    space = SearchSpace.from_dim(8)
    opt = S3Turbo(space, seed=0)  # no preset, no explicit axes
    assert opt.adaptive_mask and opt.mask == "soft"
    assert opt.scout_name == "reactive"  # adaptive escape at the general-best dial is the default
    assert opt.escape_k == 0.75  # rho_0 = 1/(k sqrt d); k=0.75 is the all-round best default
    assert opt.tr_update == "batch"


def test_escape_k_sets_base_rate():
    space = SearchSpace.from_dim(25)
    for k in (2.0, 1.0, 0.5):
        opt = S3Turbo(space, seed=0, scout_strategy="reactive", escape_k=k, budget=200)
        opt._derive(4)
        assert abs(opt.escape_base - 1.0 / (k * np.sqrt(25))) < 1e-9
        assert abs(opt.arm_gate - 0.5 * opt.escape_base) < 1e-12


def _count_scout_picks(escape_k, prob, f, seed, budget=160, q=4):
    opt = S3Turbo(prob.space, seed=seed, scout_strategy="reactive", escape_k=escape_k, budget=budget)
    scouts = 0
    while len(opt.y) < budget:
        u = opt.ask(q)
        scouts += sum(1 for kind in opt._last_kind if kind == "scout")
        opt.tell(u, np.array([f(x) for x in u]))
    return scouts


def test_escape_k_dials_scout_frequency():
    # smaller escape_k => higher base rate => strictly more far-probe (scout) picks over a run
    prob = ManyBasin(dim=20, seed=1, sigma=0.30)

    def f(x):
        return float(prob._values(x[None])[0])

    aggressive = _count_scout_picks(0.5, prob, f, seed=0)
    timid = _count_scout_picks(4.0, prob, f, seed=0)
    assert aggressive > timid


def test_escape_value_in_unit_range():
    # the reactive escape value is a bounded EMA in [0, 1] throughout a run
    mb = ManyBasin(dim=20, seed=0, sigma=0.30)
    opt = S3Turbo(mb.space, seed=0, scout_strategy="reactive", escape_k=0.5, budget=200)
    while len(opt.y) < 200:
        u = opt.ask(4)
        opt.tell(u, np.array([float(mb._values(x[None])[0]) for x in u]))
        assert 0.0 <= opt._escape_value <= 1.0


def test_adaptive_mask_alias_uses_soft_family():
    space = SearchSpace.from_dim(8)
    opt = S3Turbo(space, seed=0, mask_distribution="adaptive", scout_strategy="none")
    assert opt.mask == "soft"
    assert opt.adaptive_mask


def test_adaptive_mask_smoke_run_updates_statistics():
    space = SearchSpace.from_dim(8)
    opt = S3Turbo(space, seed=0, mask_distribution="adaptive", scout_strategy="none", budget=48)
    _run(opt, lambda u: float(((u[:2] - 0.2) ** 2).sum()), 48)
    assert np.isfinite(opt.best[1])
    assert opt.mask_rho_current > 0.0
    assert opt.mask_concentration_current > 0.0
    assert len(opt._last_alpha) > 0


def test_adaptive_mask_updates_from_successful_coordinates():
    space = SearchSpace.from_dim(10)
    opt = S3Turbo(space, seed=0, mask_distribution="adaptive", scout_strategy="none", budget=80)
    opt._last_kind = ["local", "local"]
    opt._last_alpha = [
        np.array([1.0, 1.0] + [0.0] * 8),
        np.array([0.0, 0.0] + [1.0] * 8),
    ]
    opt.tell(np.array([[0.1] * 10, [0.9] * 10]), np.array([1.0, 2.0]))
    opt._last_kind = ["local", "local"]
    opt._last_alpha = [
        np.array([1.0, 1.0] + [0.0] * 8),
        np.array([0.0, 0.0] + [1.0] * 8),
    ]
    opt.tell(np.array([[0.2] * 10, [0.8] * 10]), np.array([0.5, 2.0]))
    opt._refresh_adaptive_mask()
    assert opt._mask_credit[:2].sum() > opt._mask_credit[2:].sum()
    assert opt.mask_effective_dim < 5.0
    assert opt.mask_confidence > 0.0


def _run(opt, f, budget, q=4):
    while len(opt.y) < budget:
        U = opt.ask(q)
        opt.tell(U, np.array([f(u) for u in U]))


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_preset_smoke_runs(preset):
    space = SearchSpace.from_dim(6)
    opt = S3Turbo(space, seed=0, preset=preset, budget=48)
    _run(opt, lambda u: float(((u - 0.4) ** 2).sum()), 48)
    assert np.isfinite(opt.best[1])


def test_switch_promotes_candidates_on_many_basin():
    prob = ManyBasin(dim=6, seed=3, n_basins=12, n_global=2)
    opt = S3Turbo(prob.space, seed=3, preset="multibasin", budget=160)
    opt.tell(prob.x0u[None], np.array([prob(prob.x0)]))
    _run(opt, lambda u: float(prob._values(u[None])[0]), 160)
    assert opt.max_regions > 1
    assert len(opt._mined) >= 1  # the switch scout found at least one candidate basin


def test_tr_update_point_shrinks_faster():
    space = SearchSpace.from_dim(6)

    def f(u):
        return float(np.random.default_rng(int(u.sum() * 1e6) % 2**31).random() + 1.0)

    radii = {}
    for mode in ("batch", "point"):
        opt = S3Turbo(space, seed=0, preset="balanced", tr_update=mode, budget=80)
        _run(opt, f, 80)
        radii[mode] = opt._regions[opt._main_index()].radius
    assert radii["point"] <= radii["batch"]


def test_custom_scout_strategy_selected_by_name():
    if "always_probe" not in SCOUT:

        @SCOUT.register("always_probe")
        class AlwaysProbe(ScoutStrategy):
            def want_scout(self, opt):
                return True

            def select(self, opt, q):
                picks, ridx, kinds = opt._local_batch(q - 1)
                picks.append(opt._farthest_point())
                ridx.append(-1)
                kinds.append("scout")
                return picks, ridx, kinds

    space = SearchSpace.from_dim(4)
    opt = S3Turbo(space, seed=0, scout_strategy="always_probe")
    _run(opt, lambda u: float((u**2).sum()), 32)
    assert "scout" in opt._last_kind
    assert np.isfinite(opt.best[1])


def test_custom_mask_selected_by_name():
    if "topk" not in MASK:

        @MASK.register("topk")
        def mask_topk(rng, n, dim, rho):
            k = max(1, int(round(rho * dim)))
            m = np.zeros((n, dim))
            for row in m:
                row[rng.choice(dim, size=k, replace=False)] = 1.0
            return m

    space = SearchSpace.from_dim(5)
    opt = S3Turbo(space, seed=0, mask_distribution="topk")
    _run(opt, lambda u: float((u**2).sum()), 32)
    assert np.isfinite(opt.best[1])
