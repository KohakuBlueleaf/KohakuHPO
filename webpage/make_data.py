"""Generate data/bench_data.js for the interactive benchmark explorer on the project page.

Emits best-so-far curves (mean over seeds) for:
  - academic[task][method]: the 8 general tasks, methods = S3-TuRBO (default), S3-TuRBO (best scout),
    TuRBO, CMA-ES, GP-BO, HEBO, Sobol. The "best scout" curve is, per task, whichever scout
    (none/random/sidecar/switch) has the best final value on that task.
  - many_basin[method]: the barrier-separated escape task, methods = the 4 scouts + the baselines,
    so the reader can watch only `switch` cross the plateau.

Run: python webpage/make_data.py --workers 14 --gpus 2,3
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bench_data.js")

GEN_TASKS = [
    ("hartmann6", 6, 120), ("ackley", 25, 200), ("griewank", 25, 200), ("powell", 25, 200),
    ("rastrigin", 25, 200), ("levy", 25, 200), ("rosenbrock", 25, 200), ("styblinski_tang", 25, 200),
]
MANY = ("many_basin", 20, 300)
SEEDS = 6


def _s3(scout):
    return {"name": "s3turbo", "mask_distribution": "adaptive", "scout_strategy": scout, "tr_update": "batch"}


GEN_METHODS = {
    "s3turbo": _s3("random"), "s3_none": _s3("none"), "s3_sidecar": _s3("sidecar"), "s3_switch": _s3("switch"),
    "turbo": "turbo", "cmaes": "cmaes", "gpbo": "gpbo", "hebo": "hebo", "sobol": "sobol",
}
MANY_METHODS = {
    "s3_none": _s3("none"), "s3_random": _s3("random"), "s3_sidecar": _s3("sidecar"), "s3_switch": _s3("switch"),
    "turbo": "turbo", "cmaes": "cmaes", "gpbo": "gpbo", "hebo": "hebo", "sobol": "sobol",
}


def _init(gpus):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import torch

    torch.set_num_threads(1)
    import kohakuhpo as khpo

    if gpus and torch.cuda.is_available():
        khpo.use_device(f"cuda:{gpus[os.getpid() % len(gpus)]}")


def _job(a):
    import warnings

    warnings.filterwarnings("ignore")
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import kohakuhpo as khpo
    from kohakuhpo import OBJECTIVE, build
    from kohakuhpo.study import Result

    group, label, spec, okey, dim, budget, seed = a
    prob = build(okey, OBJECTIVE, dim=dim, seed=seed) if okey == "many_basin" else build(okey, OBJECTIVE, dim=dim)
    sp = {**spec, "budget": budget} if isinstance(spec, dict) else spec
    x0 = prob.x0 if okey == "many_basin" else None
    study = khpo.Study(prob.space, sp, seed=seed, x0=x0)
    for b in study.loop(budget=budget, q=4):
        b.report([prob(c) for c in b.configs])
    return group, label, okey, Result.from_study(study).best_so_far


def sig(x, n=4):
    if x == 0 or not np.isfinite(x):
        return float(x)
    from math import floor, log10

    return round(float(x), -int(floor(log10(abs(x)))) + (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--gpus", default="2,3")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x]
    print(f"workers={args.workers} gpus={gpus} seeds={SEEDS}", flush=True)

    jobs = []
    for okey, dim, bud in GEN_TASKS:
        for lb, sp in GEN_METHODS.items():
            for s in range(SEEDS):
                jobs.append(("gen", lb, sp, okey, dim, bud, s))
    for lb, sp in MANY_METHODS.items():
        for s in range(SEEDS):
            jobs.append(("many", lb, sp, MANY[0], MANY[1], MANY[2], s))

    raw = {}
    ctx = get_context("forkserver")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init, initargs=(gpus,)) as ex:
        for fut in as_completed([ex.submit(_job, j) for j in jobs]):
            group, label, okey, curve = fut.result()
            raw.setdefault((group, okey, label), []).append(curve)
            done += 1
            if done % 40 == 0:
                print(f"  [{done}/{len(jobs)}]", flush=True)

    def mean_curve(curves):
        L = min(len(c) for c in curves)
        return [sig(v) for v in np.mean([c[:L] for c in curves], axis=0)]

    academic = {}
    best_scout = {}  # task -> the actual scout name chosen for the second S3-TuRBO line
    for okey, _, _ in GEN_TASKS:
        m = {"s3turbo": mean_curve(raw[("gen", okey, "s3turbo")])}
        scout_finals = {
            sc: np.mean([c[-1] for c in raw[("gen", okey, f"s3_{sc}")]])
            if ("gen", okey, f"s3_{sc}") in raw
            else np.mean([c[-1] for c in raw[("gen", okey, "s3turbo")]])
            for sc in ("none", "random", "sidecar", "switch")
        }
        best = min(scout_finals, key=lambda k: scout_finals[k])
        best_scout[okey] = best
        best_key = "s3turbo" if best == "random" else f"s3_{best}"
        m["s3turbo_best"] = mean_curve(raw[("gen", okey, best_key)])
        for b in ("turbo", "cmaes", "gpbo", "hebo", "sobol"):
            m[b] = mean_curve(raw[("gen", okey, b)])
        academic[okey] = m

    many = {lb: mean_curve(raw[("many", MANY[0], lb)]) for lb in MANY_METHODS}

    payload = {
        "gen_methods": ["s3turbo", "s3turbo_best", "turbo", "cmaes", "gpbo", "hebo", "sobol"],
        "many_methods": ["s3_switch", "s3_sidecar", "s3_random", "s3_none",
                         "turbo", "cmaes", "gpbo", "hebo", "sobol"],
        "problems": [t[0] for t in GEN_TASKS],
        "academic": academic,
        "best_scout": best_scout,
        "many_basin": many,
        "seeds": SEEDS,
        "meta": {"dim": 25, "budget": 200, "q": 4, "many_dim": MANY[1], "many_budget": MANY[2]},
    }
    with open(DEST, "w") as f:
        f.write("window.BENCH = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"[wrote {DEST}]  {os.path.getsize(DEST) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
