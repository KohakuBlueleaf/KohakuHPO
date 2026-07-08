"""Cheap real-world HPO: compare optimizers on scikit-learn model tuning.

This is deliberately small enough to run many optimizer seeds while still doing genuine model
selection: every objective call trains/evaluates an sklearn pipeline on fixed CV folds. Use the
``real`` extra first:

    pip install -e ".[real]"

Example runs:

    CUDA_VISIBLE_DEVICES=2 python examples/08_sklearn_hpo.py --budget 60 --seeds 5
    CUDA_VISIBLE_DEVICES=2 python examples/08_sklearn_hpo.py --task digits_svc --budget 40
"""

import argparse
import csv
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

METHODS = ("s3turbo", "turbo", "hebo", "cmaes", "sobol", "random")
TASKS = ("breast_cancer_svc", "wine_svc", "digits_svc")

SVC_SPACE = khpo.SearchSpace(
    {
        "C": ("log", 1e-3, 1e3),
        "gamma": ("log", 1e-5, 10.0),
        "kernel": ("cat", ["rbf", "poly", "sigmoid"]),
        "degree": ("int", 2, 5),
        "coef0": ("float", 0.0, 2.0),
        "shrinking": ("cat", [False, True]),
        "class_weight": ("cat", [None, "balanced"]),
    }
)


@dataclass(frozen=True)
class SklearnTask:
    """Fixed-data, fixed-CV sklearn objective returning loss = 1 - CV score."""

    name: str
    x: np.ndarray
    y: np.ndarray
    scoring: str
    score_name: str
    n_splits: int = 3
    split_seed: int = 0

    @property
    def space(self) -> khpo.SearchSpace:
        return SVC_SPACE

    def objective(self, cfg: dict) -> float:
        model = make_pipeline(
            StandardScaler(),
            SVC(
                C=cfg["C"],
                gamma=cfg["gamma"],
                kernel=cfg["kernel"],
                degree=cfg["degree"],
                coef0=cfg["coef0"],
                shrinking=cfg["shrinking"],
                class_weight=cfg["class_weight"],
                cache_size=500,
                random_state=0,
            ),
        )
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.split_seed,
        )
        scores = cross_val_score(
            model,
            self.x,
            self.y,
            cv=cv,
            scoring=self.scoring,
            n_jobs=1,
            error_score="raise",
        )
        return float(1.0 - np.mean(scores))


@dataclass(frozen=True)
class RunRecord:
    method: str
    seed: int
    final_loss: float
    final_score: float
    auc_loss: float
    wall_s: float
    best_config: dict


def make_task(name: str, split_seed: int) -> SklearnTask:
    match name:
        case "breast_cancer_svc":
            x, y = load_breast_cancer(return_X_y=True)
            return SklearnTask(name, x, y, "roc_auc", "ROC-AUC", split_seed=split_seed)
        case "wine_svc":
            x, y = load_wine(return_X_y=True)
            return SklearnTask(name, x, y, "accuracy", "accuracy", split_seed=split_seed)
        case "digits_svc":
            x, y = load_digits(return_X_y=True)
            return SklearnTask(name, x, y, "accuracy", "accuracy", split_seed=split_seed)
        case _:
            raise ValueError(f"unknown task {name!r}; choices {TASKS}")


def optimizer_spec(method: str, preset: str, budget: int):
    if method == "s3turbo":
        return {"name": "s3turbo", "preset": preset, "budget": budget}
    return method


def configure_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            khpo.use_device("cuda:0")
            return f"cuda:0 ({torch.cuda.get_device_name(0)})"
        khpo.use_device("cpu")
        return "cpu"
    khpo.use_device(device)
    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        return f"{device} ({torch.cuda.get_device_name(idx)})"
    return device


def parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choices {METHODS}")
    return methods


def run_one(args, method: str, seed: int) -> RunRecord:
    task = make_task(args.task, args.split_seed)
    study = khpo.Study(
        task.space,
        optimizer_spec(method, args.preset, args.budget),
        seed=seed,
        failure_value=1.0,
    )
    start = time.perf_counter()
    result = study.optimize(
        task.objective,
        budget=args.budget,
        q=args.q,
        workers=args.workers,
        progress=args.progress,
        desc=f"{method}/seed{seed}",
    )
    wall_s = time.perf_counter() - start
    return RunRecord(
        method=method,
        seed=seed,
        final_loss=result.best_value,
        final_score=1.0 - result.best_value,
        auc_loss=float(np.mean(result.best_so_far)),
        wall_s=wall_s,
        best_config=result.best_config,
    )


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def mean_ranks(records: list[RunRecord]) -> dict[str, float]:
    ranks: dict[str, list[int]] = {}
    for seed in sorted({record.seed for record in records}):
        seed_records = [record for record in records if record.seed == seed]
        for rank, record in enumerate(sorted(seed_records, key=lambda r: r.final_loss), 1):
            ranks.setdefault(record.method, []).append(rank)
    return {method: float(np.mean(values)) for method, values in ranks.items()}


def print_summary(records: list[RunRecord], task: SklearnTask, budget: int, q: int) -> None:
    ranks = mean_ranks(records)
    print(f"\n{task.name}: maximize {task.score_name}; budget={budget}, q={q}")
    print("loss = 1 - score; auc_loss = mean(best-so-far loss), lower is better\n")
    print("  rank  method        score             loss              auc_loss          wall")
    rows = []
    for method in sorted({record.method for record in records}):
        method_records = [record for record in records if record.method == method]
        loss, loss_sem = mean_and_sem([record.final_loss for record in method_records])
        score, score_sem = mean_and_sem([record.final_score for record in method_records])
        auc, auc_sem = mean_and_sem([record.auc_loss for record in method_records])
        wall, wall_sem = mean_and_sem([record.wall_s for record in method_records])
        rows.append(
            (ranks[method], method, score, score_sem, loss, loss_sem, auc, auc_sem, wall, wall_sem)
        )
    for rank, method, score, score_sem, loss, loss_sem, auc, auc_sem, wall, wall_sem in sorted(
        rows
    ):
        print(
            f"  {rank:4.1f}  {method:12s}  "
            f"{score:.5f} ± {score_sem:.5f}  "
            f"{loss:.5f} ± {loss_sem:.5f}  "
            f"{auc:.5f} ± {auc_sem:.5f}  "
            f"{wall:.1f}s ± {wall_sem:.1f}s"
        )


def write_csv(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("method", "seed", "final_loss", "final_score", "auc_loss", "wall_s"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "method": record.method,
                    "seed": record.seed,
                    "final_loss": record.final_loss,
                    "final_score": record.final_score,
                    "auc_loss": record.auc_loss,
                    "wall_s": record.wall_s,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="breast_cancer_svc")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--preset", default="balanced", choices=sorted(khpo.PRESETS))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, cuda:1, or auto")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    methods = parse_methods(args.methods)
    device_label = configure_device(args.device)
    task = make_task(args.task, args.split_seed)
    print(
        f"device={device_label}; sklearn threads={args.threads}; workers={args.workers}; "
        f"methods={methods}"
    )

    records = []
    with threadpool_limits(limits=args.threads):
        for method in methods:
            for seed in range(args.seeds):
                record = run_one(args, method, seed)
                records.append(record)
                print(
                    f"{method:12s} seed={seed:<2d} "
                    f"score={record.final_score:.5f} "
                    f"loss={record.final_loss:.5f} "
                    f"auc_loss={record.auc_loss:.5f} "
                    f"wall={record.wall_s:.1f}s"
                )

    print_summary(records, task, args.budget, args.q)
    if args.csv is not None:
        write_csv(args.csv, records)
        print(f"\n[wrote {args.csv}]")


if __name__ == "__main__":
    main()
