"""Cheap real-world AutoML-style HPO over several sklearn model families.

Unlike the SVC-only example, this search space includes a model-family choice plus mostly
conditional hyperparameters. Many coordinates are inactive for any one model, which is closer to
real AutoML/HPO and a better stress test for sparse/local optimizers.

Install the optional dependency first:

    pip install -e ".[real]"

Example:

    python examples/09_sklearn_automl_hpo.py --task digits --budget 100 --q 4 --seeds 3
"""

import argparse
import csv
import os
import signal
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import kohakuhpo as khpo

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

S3_PRESETS = {
    "s3_balanced": "balanced",
    "s3_rugged": "rugged",
    "s3_smooth": "smooth",
    "s3_soft_smooth": "soft_smooth",
    "s3_heterogeneous": "heterogeneous",
    "s3_multibasin": "multibasin",
}
BASELINES = ("turbo", "cmaes", "hebo", "sobol", "random")
METHODS = tuple(S3_PRESETS) + BASELINES
TASKS = ("digits", "wine", "breast_cancer")

SPACE = khpo.SearchSpace(
    {
        "model": ("cat", ["svc", "extra_trees", "random_forest", "knn", "logreg", "hgb"]),
        "scaler": ("cat", ["standard", "minmax", "none"]),
        "feature_percentile": ("int", 30, 100),
        "svc_C": ("log", 1e-3, 1e3),
        "svc_gamma": ("log", 1e-5, 10.0),
        "svc_kernel": ("cat", ["rbf", "poly", "sigmoid"]),
        "svc_degree": ("int", 2, 5),
        "svc_coef0": ("float", 0.0, 2.0),
        "tree_n_estimators": ("int", 20, 120),
        "tree_max_depth": ("int", 2, 32),
        "tree_min_samples_leaf": ("int", 1, 20),
        "tree_max_features": ("float", 0.1, 1.0),
        "tree_criterion": ("cat", ["gini", "entropy", "log_loss"]),
        "tree_bootstrap": ("cat", [False, True]),
        "knn_n_neighbors": ("int", 1, 30),
        "knn_weights": ("cat", ["uniform", "distance"]),
        "knn_p": ("int", 1, 2),
        "logreg_C": ("log", 1e-3, 100.0),
        "hgb_learning_rate": ("log", 1e-3, 0.3),
        "hgb_max_iter": ("int", 20, 100),
        "hgb_max_leaf_nodes": ("int", 3, 63),
        "hgb_min_samples_leaf": ("int", 5, 80),
        "hgb_l2_regularization": ("log", 1e-8, 10.0),
    }
)

DEFAULT_CONFIG = {
    "model": "svc",
    "scaler": "standard",
    "feature_percentile": 100,
    "svc_C": 10.0,
    "svc_gamma": 0.01,
    "svc_kernel": "rbf",
    "svc_degree": 3,
    "svc_coef0": 0.0,
    "tree_n_estimators": 80,
    "tree_max_depth": 16,
    "tree_min_samples_leaf": 2,
    "tree_max_features": 0.6,
    "tree_criterion": "gini",
    "tree_bootstrap": False,
    "knn_n_neighbors": 5,
    "knn_weights": "distance",
    "knn_p": 2,
    "logreg_C": 1.0,
    "hgb_learning_rate": 0.05,
    "hgb_max_iter": 60,
    "hgb_max_leaf_nodes": 31,
    "hgb_min_samples_leaf": 20,
    "hgb_l2_regularization": 1e-3,
}


class EvalTimeout(Exception):
    """Raised when one sklearn fit exceeds the per-evaluation wall-clock cap."""


class timeout_guard:
    """SIGALRM-based timeout context for cheap example objectives."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._old_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self
        self._old_handler = signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)
        return False

    def _raise(self, signum, frame) -> None:
        raise EvalTimeout(f"evaluation exceeded {self.seconds}s")


class AutoMLTask:
    """Deterministic holdout AutoML objective: minimize validation error."""

    def __init__(self, name: str, split_seed: int, eval_timeout: int) -> None:
        self.name = name
        match name:
            case "digits":
                x, y = load_digits(return_X_y=True)
                test_size = 0.30
            case "wine":
                x, y = load_wine(return_X_y=True)
                test_size = 0.35
            case "breast_cancer":
                x, y = load_breast_cancer(return_X_y=True)
                test_size = 0.30
            case _:
                raise ValueError(f"unknown task {name!r}; choices {TASKS}")
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y,
            random_state=split_seed,
        )
        self.eval_timeout = eval_timeout
        self.cache: dict[tuple, float] = {}

    @property
    def space(self) -> khpo.SearchSpace:
        return SPACE

    def objective(self, cfg: dict) -> float:
        key = tuple(sorted(cfg.items()))
        if key not in self.cache:
            try:
                with timeout_guard(self.eval_timeout):
                    self.cache[key] = 1.0 - self._score(cfg)
            except Exception:
                self.cache[key] = 1.0
        return self.cache[key]

    def _score(self, cfg: dict) -> float:
        model = build_model(cfg)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_valid)
        return float(accuracy_score(self.y_valid, pred))


def build_model(cfg: dict) -> Pipeline:
    steps = []
    percentile = int(cfg["feature_percentile"])
    if percentile < 100:
        steps.append(("select", SelectPercentile(f_classif, percentile=percentile)))
    if cfg["model"] in {"svc", "knn", "logreg"}:
        match cfg["scaler"]:
            case "standard":
                steps.append(("scale", StandardScaler()))
            case "minmax":
                steps.append(("scale", MinMaxScaler()))
            case "none":
                pass
    steps.append(("model", build_estimator(cfg)))
    return Pipeline(steps)


def build_estimator(cfg: dict):
    match cfg["model"]:
        case "svc":
            return SVC(
                C=cfg["svc_C"],
                gamma=cfg["svc_gamma"],
                kernel=cfg["svc_kernel"],
                degree=cfg["svc_degree"],
                coef0=cfg["svc_coef0"],
                cache_size=500,
                max_iter=5000,
                random_state=0,
            )
        case "extra_trees":
            return tree_model(ExtraTreesClassifier, cfg)
        case "random_forest":
            return tree_model(RandomForestClassifier, cfg)
        case "knn":
            return KNeighborsClassifier(
                n_neighbors=cfg["knn_n_neighbors"],
                weights=cfg["knn_weights"],
                p=cfg["knn_p"],
            )
        case "logreg":
            return LogisticRegression(C=cfg["logreg_C"], max_iter=300, random_state=0)
        case "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=cfg["hgb_learning_rate"],
                max_iter=cfg["hgb_max_iter"],
                max_leaf_nodes=cfg["hgb_max_leaf_nodes"],
                min_samples_leaf=cfg["hgb_min_samples_leaf"],
                l2_regularization=cfg["hgb_l2_regularization"],
                early_stopping=True,
                random_state=0,
            )
        case _:
            raise ValueError(f"unknown model {cfg['model']!r}")


def tree_model(cls, cfg: dict):
    return cls(
        n_estimators=cfg["tree_n_estimators"],
        max_depth=cfg["tree_max_depth"],
        min_samples_leaf=cfg["tree_min_samples_leaf"],
        max_features=cfg["tree_max_features"],
        criterion=cfg["tree_criterion"],
        bootstrap=cfg["tree_bootstrap"],
        n_jobs=1,
        random_state=0,
    )


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choices {METHODS}")
    return methods


def optimizer_spec(method: str, budget: int):
    if method in S3_PRESETS:
        return {"name": "s3turbo", "preset": S3_PRESETS[method], "budget": budget}
    return method


def run_one(args, method: str, seed: int) -> dict:
    task = AutoMLTask(args.task, args.split_seed, args.eval_timeout)
    x0 = None if args.no_x0 else DEFAULT_CONFIG
    study = khpo.Study(
        task.space,
        optimizer_spec(method, args.budget),
        seed=seed,
        x0=x0,
        failure_value=1.0,
    )
    start = time.perf_counter()
    result = study.optimize(
        task.objective,
        budget=args.budget,
        q=args.q,
        progress=args.progress,
        desc=f"{method}/seed{seed}",
    )
    wall_s = time.perf_counter() - start
    return {
        "method": method,
        "seed": seed,
        "final_loss": result.best_value,
        "final_score": 1.0 - result.best_value,
        "auc_loss": float(np.mean(result.best_so_far)),
        "wall_s": wall_s,
        "best_model": result.best_config["model"],
    }


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), sem


def mean_ranks(records: list[dict]) -> dict[str, float]:
    ranks: dict[str, list[int]] = {}
    for seed in sorted({record["seed"] for record in records}):
        seed_records = [record for record in records if record["seed"] == seed]
        for rank, record in enumerate(sorted(seed_records, key=lambda r: r["final_loss"]), 1):
            ranks.setdefault(record["method"], []).append(rank)
    return {method: float(np.mean(values)) for method, values in ranks.items()}


def print_summary(records: list[dict], args) -> None:
    ranks = mean_ranks(records)
    rows = []
    for method in sorted({record["method"] for record in records}):
        subset = [record for record in records if record["method"] == method]
        score, score_sem = mean_sem([record["final_score"] for record in subset])
        loss, loss_sem = mean_sem([record["final_loss"] for record in subset])
        auc, auc_sem = mean_sem([record["auc_loss"] for record in subset])
        wall, wall_sem = mean_sem([record["wall_s"] for record in subset])
        rows.append(
            (ranks[method], method, score, score_sem, loss, loss_sem, auc, auc_sem, wall, wall_sem)
        )
    print(
        f"\n{args.task} AutoML holdout accuracy; budget={args.budget}, q={args.q}, x0={not args.no_x0}"
    )
    print("loss = 1 - validation accuracy; auc_loss = mean(best-so-far loss), lower is better\n")
    print("  rank  method             score             loss              auc_loss          wall")
    for rank, method, score, score_sem, loss, loss_sem, auc, auc_sem, wall, wall_sem in sorted(
        rows
    ):
        print(
            f"  {rank:4.1f}  {method:17s}  "
            f"{score:.5f} ± {score_sem:.5f}  "
            f"{loss:.5f} ± {loss_sem:.5f}  "
            f"{auc:.5f} ± {auc_sem:.5f}  "
            f"{wall:.1f}s ± {wall_sem:.1f}s"
        )


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="digits")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=5)
    parser.add_argument("--no-x0", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    methods = parse_methods(args.methods)
    print(
        f"task={args.task}; methods={methods}; budget={args.budget}; q={args.q}; "
        f"seeds={args.seeds}; x0={not args.no_x0}; threads={args.threads}"
    )
    records = []
    with threadpool_limits(limits=args.threads):
        for method in methods:
            for seed in range(args.seeds):
                record = run_one(args, method, seed)
                records.append(record)
                print(
                    f"{method:17s} seed={seed:<2d} "
                    f"score={record['final_score']:.5f} "
                    f"loss={record['final_loss']:.5f} "
                    f"auc_loss={record['auc_loss']:.5f} "
                    f"best_model={record['best_model']:13s} "
                    f"wall={record['wall_s']:.1f}s"
                )
    print_summary(records, args)
    if args.csv is not None:
        write_csv(args.csv, records)
        print(f"\n[wrote {args.csv}]")


if __name__ == "__main__":
    main()
