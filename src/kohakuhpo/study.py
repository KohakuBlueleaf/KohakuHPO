"""Study: one object owning a run; every user-facing interface is a view over it.

A :class:`Study` wires a :class:`~kohakuhpo.space.SearchSpace` to a cube-level optimizer and
handles what the optimizer deliberately does not know: typed configs, the optimization direction,
the trial history, warm-start points, and failed evaluations.

The three interfaces share the Study's state, so they can be mixed within one run:

* ask/tell:   ``configs = study.ask(q)`` then ``study.tell(configs, values)``
* iterator:   ``for batch in study.loop(budget, q): batch.report(values)``
* closure:    ``result = study.optimize(objective, budget, q, workers=...)``

Direction: the optimizer always minimizes; a ``direction="max"`` Study negates values on ``tell``
and reports user-space values everywhere. Failures: a ``None``/NaN/inf value (or an exception in
``optimize``) is recorded as ``failure_value`` (default ``1e9`` for min, ``-1e9`` for max), so a
crashed trial never aborts a run. Telling a config that was never asked is allowed (imported or
injected observations).
"""

import multiprocessing as mp
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from kohakuhpo.optimizer import Optimizer
from kohakuhpo.registry import OPTIMIZER, build
from kohakuhpo.space import SearchSpace


@dataclass
class Trial:
    """One suggested (or injected) configuration and, once told, its outcome."""

    id: int
    config: dict
    u: np.ndarray
    value: float | None = None
    state: str = "pending"  # "pending" | "done" | "failed"


class Batch:
    """The unit yielded by :meth:`Study.loop`: evaluate ``configs``, then ``report`` the values."""

    def __init__(self, study: "Study", trials: list[Trial]) -> None:
        self._study = study
        self.trials = trials
        self.reported = False

    @property
    def configs(self) -> list[dict]:
        return [t.config for t in self.trials]

    def report(self, values: Sequence[float | None]) -> None:
        """Report one value per config, in order; ``None``/NaN marks that trial failed."""
        if self.reported:
            raise RuntimeError("this batch was already reported")
        self._study._tell_trials(self.trials, values)
        self.reported = True

    def __iter__(self):
        return iter(self.trials)

    def __len__(self) -> int:
        return len(self.trials)


@dataclass
class Result:
    """Outcome of a driven run: the incumbent plus per-evaluation traces (user-space values)."""

    best_config: dict
    best_value: float
    history: list[float]
    best_so_far: list[float] = field(default_factory=list)
    trials: list[Trial] = field(default_factory=list)

    @classmethod
    def from_study(cls, study: "Study") -> "Result":
        done = study.completed
        history = [t.value for t in done]
        sign = 1.0 if study.direction == "min" else -1.0
        m = float("inf")
        best_so_far = [sign * (m := min(m, sign * v)) for v in history]
        return cls(study.best_config, study.best_value, history, best_so_far, list(study.trials))


def _eval_one(objective, config: dict) -> float | None:
    """Evaluate one config; exceptions and non-dict/float results reduce to a value or ``None``."""
    try:
        v = objective(config)
        v = v["score"] if isinstance(v, dict) else v
        return float(v)
    except Exception:
        return None


def _evaluate_batch(objective, configs: list[dict], vectorized: bool, pool) -> list[float | None]:
    """Evaluate a batch sequentially, across a process pool, or in one vectorized call."""
    if vectorized:
        try:
            return [float(v) for v in objective(configs)]
        except Exception:
            return [None] * len(configs)
    if pool is not None:
        return list(pool.map(_eval_one, [objective] * len(configs), configs))
    return [_eval_one(objective, c) for c in configs]


class Study:
    """Space + optimizer + trials + direction + failure policy.

    ``optimizer`` is anything :func:`~kohakuhpo.registry.build` accepts: a registry key
    (``"s3turbo"``), a ``{"name": ..., **kwargs}`` dict, a class, a dotted path, or an instance.
    ``x0`` (one config dict or a list) is served by the first ``ask`` calls before the optimizer
    proposes anything.
    """

    def __init__(
        self,
        space: SearchSpace,
        optimizer="s3turbo",
        *,
        seed: int = 0,
        direction: str = "min",
        x0: dict | list[dict] | None = None,
        failure_value: float | None = None,
    ) -> None:
        if direction not in ("min", "max"):
            raise ValueError(f"direction {direction!r}; choices ['min', 'max']")
        self.space = space
        self.direction = direction
        self._sign = 1.0 if direction == "min" else -1.0
        self.failure_value = failure_value if failure_value is not None else self._sign * 1e9
        self.optimizer: Optimizer = build(optimizer, OPTIMIZER, space=space, seed=seed)
        self.trials: list[Trial] = []
        self._pending: list[Trial] = []
        self._x0_queue = [] if x0 is None else ([x0] if isinstance(x0, dict) else list(x0))

    # ---- ask / tell ------------------------------------------------------------------------- #
    def ask(self, q: int = 1) -> list[dict]:
        """Propose ``q`` configs to evaluate; queued ``x0`` configs are served first."""
        return [t.config for t in self._ask_trials(q)]

    def _ask_trials(self, q: int) -> list[Trial]:
        trials: list[Trial] = []
        while self._x0_queue and len(trials) < q:
            cfg = self._x0_queue.pop(0)
            trials.append(self._new_trial(cfg, self.space.to_unit(cfg)))
        n = q - len(trials)
        if n > 0:
            for u in self.optimizer.ask(n):
                trials.append(self._new_trial(self.space.to_config(u), u))
        return trials

    def tell(self, configs, values=None) -> None:
        """Report outcomes: ``tell(config, value)`` or ``tell(list_of_configs, values)``.

        Configs are matched to pending asked trials by equality; an unmatched config becomes an
        injected observation. Values follow the study's direction; ``None``/NaN/inf is a failure.
        """
        if isinstance(configs, dict):
            configs, values = [configs], [values]
        trials = [self._match_pending(c) for c in configs]
        self._tell_trials(trials, values)

    def _new_trial(self, config: dict, u: np.ndarray) -> Trial:
        t = Trial(id=len(self.trials), config=config, u=np.asarray(u, dtype=float))
        self.trials.append(t)
        self._pending.append(t)
        return t

    def _match_pending(self, config: dict) -> Trial:
        for t in self._pending:
            if t.config == config:
                return t
        return self._new_trial(config, self.space.to_unit(config))

    def _tell_trials(self, trials: list[Trial], values) -> None:
        values = list(values)
        if len(values) != len(trials):
            raise ValueError(f"got {len(values)} values for {len(trials)} trials")
        U, ys = [], []
        for t, v in zip(trials, values, strict=True):
            failed = v is None or not np.isfinite(v)
            t.value = float(self.failure_value if failed else v)
            t.state = "failed" if failed else "done"
            if t in self._pending:
                self._pending.remove(t)
            U.append(t.u)
            ys.append(self._sign * t.value)
        self.optimizer.tell(np.stack(U), np.array(ys))

    # ---- state ------------------------------------------------------------------------------ #
    @property
    def n_evals(self) -> int:
        """Number of trials with a reported outcome (pending asks do not count)."""
        return sum(t.state != "pending" for t in self.trials)

    @property
    def completed(self) -> list[Trial]:
        return [t for t in self.trials if t.state != "pending"]

    @property
    def best_trial(self) -> Trial | None:
        done = self.completed
        if not done:
            return None
        return min(done, key=lambda t: self._sign * t.value)

    @property
    def best_config(self) -> dict | None:
        t = self.best_trial
        return None if t is None else t.config

    @property
    def best_value(self) -> float:
        t = self.best_trial
        return self._sign * float("inf") if t is None else t.value

    # ---- iterator interface ----------------------------------------------------------------- #
    def loop(self, budget: int, q: int = 1, progress: bool = False, desc: str | None = None):
        """Yield :class:`Batch` es until ``n_evals`` reaches ``budget`` (a running total, so a
        warm study resumes where it left off). Each batch must be reported before the next.
        ``progress=True`` shows a tqdm bar with the running best value."""
        bar = None
        if progress:
            bar = tqdm(total=budget, initial=self.n_evals, desc=desc or "optimize", unit="eval")
        try:
            while self.n_evals < budget:
                batch = Batch(self, self._ask_trials(min(q, budget - self.n_evals)))
                yield batch
                if not batch.reported:
                    raise RuntimeError("Batch.report(values) must be called before the next batch")
                if bar is not None:
                    bar.update(len(batch))
                    bar.set_postfix_str(f"best={self.best_value:.5g}")
        finally:
            if bar is not None:
                bar.close()

    # ---- closure interface ------------------------------------------------------------------ #
    def optimize(
        self,
        objective,
        budget: int,
        q: int = 1,
        workers: int = 1,
        vectorized: bool = False,
        progress: bool = False,
        desc: str | None = None,
    ) -> Result:
        """Drive ``objective`` for ``budget`` total evaluations; return a :class:`Result`.

        ``objective`` maps a config dict to a value. With ``vectorized=True`` it receives the
        whole batch (``list[dict] -> sequence of values``) in one call. ``workers > 1`` spreads a
        batch across a process pool (the objective must be picklable). Exceptions and non-finite
        values become failures under the study's failure policy. ``progress=True`` shows a tqdm
        bar with the running best value.
        """
        pool = None
        if workers > 1 and not vectorized:
            # forkserver, not fork: torch's threads/locks do not survive a fork.
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("forkserver"))
        try:
            for batch in self.loop(budget, q, progress=progress, desc=desc):
                batch.report(_evaluate_batch(objective, batch.configs, vectorized, pool))
        finally:
            if pool is not None:
                pool.shutdown()
        return Result.from_study(self)
