"""Reproduce the KohakuHPO benchmark: S3-TuRBO vs prior work + the mask-axis ablation.

Parallel across workers (ProcessPoolExecutor + forkserver); each worker pins itself to one of the
given GPUs for the GP surrogate math. Two blocks:

  (1) COMPARISON:   S3-TuRBO (adaptive) vs prior work: TuRBO, GP-BO, HEBO, CMA-ES, Sobol.
  (2) ABLATION:     the mask axis, scout held at `random`, tr_update batch:
                      none(dense) / hard / soft(fixed) / adaptive, so the mask contribution is isolated.

Each (method, task, seed) is one job. Results are aggregated to per-task finals (mean over seeds)
and average rank. Usage:
    python examples/11_benchmark_suite.py --workers 16 --seeds 6           # CPU (float64)
    python examples/11_benchmark_suite.py --workers 16 --gpus 0 --seeds 6   # GP math on GPU 0
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

# ---- task definitions: (name, objective_key, dim, budget) ----
# General suite: comparison vs prior work and the mask ablation. The many-basin escape task is NOT
# here; it needs a scout, so it is run separately as the scout ablation (below), where it belongs.
TASKS = [
    ("Hartmann6", "hartmann6", 6, 120),
    ("Ackley25", "ackley", 25, 200),
    ("Griewank25", "griewank", 25, 200),
    ("Powell25", "powell", 25, 200),
    ("Rastrigin25", "rastrigin", 25, 200),
    ("Levy25", "levy", 25, 200),
    ("Rosenbrock25", "rosenbrock", 25, 200),
    ("StyblinskiTang25", "styblinski_tang", 25, 200),
]
# The scout axis only shows itself on a barrier-separated landscape where local search is trapped.
SCOUT_TASK = [("ManyBasin20", "many_basin", 20, 300)]


# ---- methods ----
def _s3(mask, scout="random"):
    return {
        "name": "s3turbo",
        "mask_distribution": mask,
        "scout_strategy": scout,
        "tr_update": "batch",
    }


COMPARISON = {
    "s3turbo (adaptive)": _s3("adaptive"),
    "turbo": "turbo",
    "gpbo": "gpbo",
    "hebo": "hebo",
    "cmaes": "cmaes",
    "sobol": "sobol",
}
ABLATION = {
    "mask=dense": _s3("dense"),
    "mask=hard": _s3("hard"),
    "mask=soft": _s3("soft"),
    "mask=adaptive": _s3("adaptive"),
}
# The scout (escape) axis is only exercised on a barrier-separated landscape, so it is ablated
# separately on the many-basin task where local methods get trapped and only a scout escapes.
SCOUT_ABLATION = {
    "scout=none": _s3("adaptive", "none"),
    "scout=random": _s3("adaptive", "random"),
    "scout=sidecar": _s3("adaptive", "sidecar"),
    "scout=switch": _s3("adaptive", "switch"),
}


def _worker_init(gpus):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import torch

    torch.set_num_threads(1)
    import kohakuhpo as khpo

    if gpus and torch.cuda.is_available():
        # pin this worker to one visible GPU round-robin by pid
        gpu = gpus[os.getpid() % len(gpus)]
        khpo.use_device(f"cuda:{gpu}")


def _job(args):
    import warnings

    warnings.filterwarnings("ignore")
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import kohakuhpo as khpo
    from kohakuhpo import OBJECTIVE, build

    label, spec, task, okey, dim, budget, seed, q = args
    if okey == "many_basin":
        prob = build(okey, OBJECTIVE, dim=dim, seed=seed)
        x0 = prob.x0
        sp = {**spec, "budget": budget} if isinstance(spec, dict) else spec
        study = khpo.Study(prob.space, sp, seed=seed, x0=x0)
        for batch in study.loop(budget=budget, q=q):
            study_vals = [prob(c) for c in batch.configs]
            batch.report(study_vals)
        final = study.best_value
    else:
        prob = build(okey, OBJECTIVE, dim=dim)
        sp = {**spec, "budget": budget} if isinstance(spec, dict) else spec
        res = khpo.minimize(prob, prob.space, sp, budget=budget, q=q, seed=seed)
        final = res.best_value
    return label, task, seed, float(final)


def run_block(methods, tasks, seeds, q, workers, gpus, title):
    jobs = []
    for label, spec in methods.items():
        for tname, okey, dim, budget in tasks:
            for s in range(seeds):
                jobs.append((label, spec, tname, okey, dim, budget, s, q))
    finals = defaultdict(lambda: defaultdict(list))  # method -> task -> [finals]
    ctx = get_context("forkserver")
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(gpus,)
    ) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            label, tname, seed, final = fut.result()
            finals[label][tname].append(final)
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(jobs)}] {title}", flush=True)
    # aggregate
    task_names = [t[0] for t in tasks]
    mean_final = {m: {t: float(np.mean(finals[m][t])) for t in task_names} for m in methods}
    ranks = {m: [] for m in methods}
    for t in task_names:
        vals = sorted(mean_final[m][t] for m in methods)
        for m in methods:
            v = mean_final[m][t]
            ranks[m].append(
                sum(i + 1 for i, x in enumerate(vals) if abs(x - v) < 1e-12)
                / sum(1 for x in vals if abs(x - v) < 1e-12)
            )
    avg = {m: float(np.mean(ranks[m])) for m in methods}
    return {
        "title": title,
        "tasks": task_names,
        "mean_final": mean_final,
        "ranks": ranks,
        "avg": avg,
    }


def print_block(res):
    print(f"\n===== {res['title']} =====")
    print(f"{'task':17s} " + " ".join(f"{m:>19s}" for m in res["mean_final"]))
    for t in res["tasks"]:
        print(f"{t:17s} " + " ".join(f"{res['mean_final'][m][t]:19.5g}" for m in res["mean_final"]))
    print("\navg rank (lower better):")
    for m in sorted(res["avg"], key=lambda k: res["avg"][k]):
        print(f"  {res['avg'][m]:.2f}  {m:20s}  {[round(x,1) for x in res['ranks'][m]]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids for GP math; empty = CPU")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    print(f"workers={args.workers} gpus={gpus} seeds={args.seeds} q={args.q}\n", flush=True)

    comp = run_block(
        COMPARISON, TASKS, args.seeds, args.q, args.workers, gpus, "COMPARISON vs prior work"
    )
    print_block(comp)
    abl = run_block(ABLATION, TASKS, args.seeds, args.q, args.workers, gpus, "ABLATION: mask axis")
    print_block(abl)
    scout = run_block(
        SCOUT_ABLATION, SCOUT_TASK, args.seeds, args.q, args.workers, gpus, "ABLATION: scout axis"
    )
    print_block(scout)

    with open(args.out, "w") as f:
        json.dump(
            {
                "comparison": comp,
                "ablation": abl,
                "scout_ablation": scout,
                "config": {"seeds": args.seeds, "q": args.q, "tasks": [t[0] for t in TASKS]},
            },
            f,
            indent=2,
        )
    print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
