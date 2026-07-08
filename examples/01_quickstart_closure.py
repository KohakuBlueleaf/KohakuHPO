"""Quickstart: one call. Define a space, hand kohakuhpo a function, get the best config back."""

import kohakuhpo as khpo


def objective(cfg: dict) -> float:
    """Anything that maps a config dict to a score (lower is better here)."""
    return (
        (cfg["lr"] - 3e-3) ** 2 / (cfg["lr"] * 3e-3)
        + (cfg["beta2"] - 0.98) ** 2 * 100
        + abs(cfg["layers"] - 6) * 0.05
        + (0.0 if cfg["act"] == "gelu" else 0.1)
    )


space = khpo.SearchSpace(
    {
        "lr": ("log", 1e-5, 1e-1),
        "beta2": ("float", 0.9, 0.999),
        "layers": ("int", 2, 12),
        "act": ("cat", ["relu", "gelu", "silu"]),
    }
)

result = khpo.minimize(objective, space, optimizer="s3turbo", budget=120, q=4, seed=0)

print(f"best value : {result.best_value:.5f}")
print(f"best config: {result.best_config}")
print(f"evaluations: {len(result.history)}")
