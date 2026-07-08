"""SearchSpace codecs: round-trips, clipping, and custom parameter kinds."""

import numpy as np
import pytest

from kohakuhpo import PARAM, SearchSpace


def test_roundtrip_all_kinds():
    space = SearchSpace(
        {
            "lr": ("log", 1e-5, 1e-1),
            "w": ("float", -2.0, 3.0),
            "layers": ("int", 2, 12),
            "act": ("cat", ["relu", "gelu", "silu"]),
        }
    )
    assert space.dim == 4
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.random(4)
        cfg = space.to_config(u)
        cfg2 = space.to_config(space.to_unit(cfg))
        assert cfg == cfg2
        assert 1e-5 <= cfg["lr"] <= 1e-1
        assert isinstance(cfg["layers"], int) and 2 <= cfg["layers"] <= 12
        assert cfg["act"] in ("relu", "gelu", "silu")


def test_batch_helpers():
    space = SearchSpace.from_dim(3)
    U = np.random.default_rng(1).random((5, 3))
    cfgs = space.to_configs(U)
    assert len(cfgs) == 5
    assert np.allclose(space.to_units(cfgs), U)


def test_log_param_rejects_nonpositive():
    with pytest.raises(ValueError):
        SearchSpace({"lr": ("log", 0.0, 1.0)})


def test_custom_codec_registration():
    @PARAM.register("step_float")
    class StepFloat:
        def __init__(self, lo, hi, step):
            self.lo, self.hi, self.step = lo, hi, step

        def decode(self, u):
            v = self.lo + u * (self.hi - self.lo)
            return round(v / self.step) * self.step

        def encode(self, v):
            return (v - self.lo) / (self.hi - self.lo)

    space = SearchSpace({"x": {"name": "step_float", "lo": 0.0, "hi": 1.0, "step": 0.25}})
    assert space.to_config(np.array([0.4]))["x"] == 0.5


def test_codec_instance_passthrough():
    class Identity:
        def decode(self, u):
            return u

        def encode(self, v):
            return v

    space = SearchSpace({"x": Identity()})
    assert space.to_config(np.array([0.37]))["x"] == pytest.approx(0.37)
