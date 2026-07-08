"""GP surrogate gates: interpolation, joint sampling, warps, acquisitions."""

import numpy as np
import torch

from kohakuhpo.surrogate import GP, log_ei, output_warp, prob_improve, upper_conf


def _data(n=24, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, d))
    y = ((x - 0.4) ** 2).sum(1)
    kw = {"dtype": torch.float64}
    return torch.tensor(x, **kw), torch.tensor(y, **kw)


def test_gp_interpolates_training_points():
    x, y = _data()
    gp = GP(x, y)
    mean, std = gp.predict(x)
    assert float((mean - y).abs().max()) < 0.05
    assert bool((std >= 0).all())


def test_gp_joint_sample_shape_and_spread():
    x, y = _data()
    gp = GP(x, y)
    xq = torch.rand(30, 3, dtype=torch.float64)
    torch.manual_seed(0)
    draws = gp.sample(xq, 8)
    assert draws.shape == (8, 30)
    assert float(draws.std(0).mean()) > 0  # draws disagree away from data


def test_warped_gp_runs():
    x, y = _data()
    gp = GP(x, y, warp_input=True)
    mean, _ = gp.predict(x)
    assert float((mean - y).abs().max()) < 0.2


def test_output_warp_is_monotone():
    y = np.array([-5.0, 0.0, 1.0, 3.0, 1e6])
    w = output_warp(y)
    assert np.all(np.diff(w) > 0)
    assert np.argmin(w) == np.argmin(y)


def test_acquisitions_finite_and_ordered():
    mean = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    std = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    for acq in (log_ei, prob_improve):
        v = acq(mean, std, best=1.0)
        assert torch.isfinite(v).all()
        assert v[0] > v[2]  # lower predicted mean scores higher for minimization
    u = upper_conf(mean, std, beta=2.0)
    assert u[0] > u[2]
