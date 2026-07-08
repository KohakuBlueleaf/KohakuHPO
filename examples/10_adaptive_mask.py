"""Watch the adaptive soft mask learn: credit concentrates and the concentration c0 sharpens.

The objective is anisotropic (only `k` of `d` coordinates change the value) and unknown to the
optimizer. The adaptive mask (mask_distribution="adaptive") runs the soft mask but learns its
concentration online from per-coordinate improvement credit (attributed by realized displacement),
so it sharpens from soft toward hard as the active coordinates separate. This prints the credit
concentration and the learned c0 over the run; for the full benchmark see 11_benchmark_suite.py.

Usage:
    python examples/10_adaptive_mask.py --dim 25 --active 3 --budget 300
"""

import argparse

import numpy as np

import kohakuhpo as khpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=25)
    ap.add_argument("--active", type=int, default=3, help="coordinates that actually matter")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    active = sorted(rng.choice(args.dim, size=args.active, replace=False).tolist())
    targets = {j: rng.uniform(0.2, 0.8) for j in active}
    space = khpo.SearchSpace.from_dim(args.dim)

    def objective(cfg: dict) -> float:
        u = space.to_unit(cfg)
        return float(sum((i + 1) * (u[j] - t) ** 2 for i, (j, t) in enumerate(targets.items())))

    study = khpo.Study(
        space,
        {
            "name": "s3turbo",
            "mask_distribution": "adaptive",
            "scout_strategy": "none",
            "budget": args.budget,
        },
        seed=args.seed,
    )
    opt = study.optimizer
    print(f"d={args.dim}, active coords={active} (unknown to the optimizer)\n")
    print(
        f"{'evals':>6} {'best':>11} {'confidence':>11} {'c0 (learned)':>13} {'credit active/inert':>20}"
    )

    # checkpoints snapped to multiples of q so they always land on a reported step
    checkpoints = {round(f * args.budget / args.q) * args.q for f in (0.1, 0.25, 0.5, 0.75, 1.0)}
    for batch in study.loop(budget=args.budget, q=args.q):
        batch.report([objective(c) for c in batch.configs])
        if study.n_evals in checkpoints:
            credit = opt._mask_credit
            act = credit[active].mean()
            inert = np.delete(credit, active).mean()
            ratio = act / max(inert, 1e-12)
            print(
                f"{study.n_evals:>6} {study.best_value:>11.4g} {opt.mask_confidence:>11.2f} "
                f"{opt.mask_concentration_current:>13.3f} {ratio:>20.1f}"
            )

    print(f"\nbest = {study.best_value:.4g}")
    print("credit ratio > 1 means the mask concentrated on the coordinates that actually matter;")
    print(
        "c0 falling toward mask_min_concentration means the mask sharpened from soft toward hard."
    )


if __name__ == "__main__":
    main()
