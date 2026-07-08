"""Compare every built-in optimizer on one benchmark objective; prints a ranked table.

Add --plot to draw best-so-far curves.
"""

import argparse

import matplotlib.pyplot as plt

import kohakuhpo as khpo
from kohakuhpo import OBJECTIVE, build

METHODS = ["s3turbo", "turbo", "gpbo", "hebo", "cmaes", "sobol", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="hartmann6")
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    curves = {}
    for name in METHODS:
        traces = []
        for seed in range(args.seeds):
            problem = build(args.objective, OBJECTIVE)
            res = khpo.minimize(
                problem, problem.space, name, budget=args.budget, q=args.q, seed=seed
            )
            traces.append(res.best_so_far)
        curves[name] = [sum(v) / len(v) for v in zip(*traces, strict=True)]

    print(f"\n{args.objective} (d={build(args.objective, OBJECTIVE).dim}), "
          f"budget={args.budget}, q={args.q}, mean of {args.seeds} seeds\n")  # fmt: skip
    for rank, (name, curve) in enumerate(sorted(curves.items(), key=lambda kv: kv[1][-1]), 1):
        print(f"  #{rank}  {name:12s} final={curve[-1]:.5f}")

    if args.plot:
        for name, curve in curves.items():
            plt.plot(range(1, len(curve) + 1), curve, label=name)
        plt.xlabel("evaluations")
        plt.ylabel("best-so-far (mean)")
        plt.title(args.objective)
        plt.legend()
        plt.savefig("compare_optimizers.png", dpi=150)
        print("\n[wrote compare_optimizers.png]")


if __name__ == "__main__":
    main()
