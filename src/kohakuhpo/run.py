"""One-call drivers: build a Study, run it against a closure, return the Result."""

from kohakuhpo.space import SearchSpace
from kohakuhpo.study import Result, Study


def minimize(
    objective,
    space: SearchSpace,
    optimizer="s3turbo",
    *,
    budget: int = 100,
    q: int = 1,
    seed: int = 0,
    x0: dict | list[dict] | None = None,
    workers: int = 1,
    vectorized: bool = False,
    failure_value: float | None = None,
    direction: str = "min",
    progress: bool = False,
) -> Result:
    """Minimize ``objective`` over ``space`` with ``budget`` evaluations in batches of ``q``.

    ``objective(config) -> float`` (or ``objective(list[dict]) -> sequence`` with
    ``vectorized=True``). ``x0`` configs are evaluated first; ``workers > 1`` runs a batch across
    a process pool; ``progress=True`` shows a tqdm bar. Returns a
    :class:`~kohakuhpo.study.Result`.
    """
    study = Study(
        space,
        optimizer,
        seed=seed,
        direction=direction,
        x0=x0,
        failure_value=failure_value,
    )
    return study.optimize(
        objective, budget=budget, q=q, workers=workers, vectorized=vectorized, progress=progress
    )


def maximize(objective, space: SearchSpace, optimizer="s3turbo", **kwargs) -> Result:
    """:func:`minimize` with ``direction="max"``: bigger objective values are better."""
    kwargs["direction"] = "max"
    return minimize(objective, space, optimizer, **kwargs)
