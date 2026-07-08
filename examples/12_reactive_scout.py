"""The adaptive scout (reactive): escape a trapped basin without paying for it on smooth problems.

The escape axis has a dilemma. A pure local search (`scout="none"`) is the best all-round choice, but
it can never leave the basin it starts in; on a landscape with far-apart basins it gets stuck. The
`switch` scout escapes reliably, but it fires a focus burst on every landscape, so it wastes budget
(and loses badly) on smooth problems where there is nothing to escape.

`reactive` is the adaptive resolution. It does not try to predict whether a far basin exists (a local
search's data can't reveal that; the surrogate has never sampled there). Instead it keeps a small
always-on base scout rate and tracks an escape value E: a planted candidate region that proves
spatially distinct from the incumbent and competitive raises E, one that drifts back to the same basin
lowers it. Scout rate and focus burst scale with E, so on a smooth funnel candidates prove redundant,
E decays, and reactive behaves like `none`; on a multi-basin landscape a distinct find raises E and the
escape ramps up.

This runs `none`, `switch`, and `reactive` on both a smooth task (where the scout should cost nothing)
and a barrier-separated many-basin task (where only a real escape reaches the deep core), printing the
honest trade-off.

    python examples/12_reactive_scout.py
"""

import numpy as np

import kohakuhpo as khpo
from kohakuhpo import OBJECTIVE, build
from kohakuhpo.benchmarks import ManyBasin

SCOUTS = ["none", "switch", "reactive"]
SEEDS = 6


def run_smooth(scout):
    """Ackley d=25: a scout should NOT hurt here (nothing to escape)."""
    vals = []
    for s in range(SEEDS):
        prob = build("ackley", OBJECTIVE, dim=25)
        res = khpo.minimize(
            prob,
            prob.space,
            {
                "name": "s3turbo",
                "mask_distribution": "adaptive",
                "scout_strategy": scout,
                "budget": 200,
            },
            budget=200,
            q=4,
            seed=s,
        )
        vals.append(res.best_value)
    return float(np.mean(vals))


def run_escape(scout):
    """Barrier-separated many-basin: only a real escape reaches a deep core."""
    regrets, hits = [], 0
    for s in range(SEEDS):
        prob = ManyBasin(dim=20, seed=s, sigma=0.30)  # medium difficulty
        study = khpo.Study(
            prob.space,
            {
                "name": "s3turbo",
                "mask_distribution": "adaptive",
                "scout_strategy": scout,
                "budget": 300,
            },
            seed=s,
            x0=prob.x0,
        )
        for batch in study.loop(budget=300, q=4):
            batch.report([prob(c) for c in batch.configs])
        regrets.append(study.best_value)
        hits += bool(prob.in_core(prob.space.to_unit(study.best_config)[None])[0])
    return float(np.mean(regrets)), hits / SEEDS


def main():
    print(
        f"{SEEDS} seeds. Smooth: lower regret is better. Escape: core-hit is the fraction reaching a deep core.\n"
    )
    print(
        f"{'scout':8s} {'smooth Ackley (regret)':>24s} {'escape many-basin (regret / core-hit)':>40s}"
    )
    for sc in SCOUTS:
        smooth = run_smooth(sc)
        reg, hit = run_escape(sc)
        print(f"{sc:8s} {smooth:24.3f} {f'{reg:.3f} / {hit:.0%}':>40s}", flush=True)
    print(
        "\nRead the rows: `none` is cheap on the smooth task but never escapes; `switch` escapes but"
        "\nwrecks the smooth task; `reactive` stays close to `none` on smooth AND escapes on many-basin."
    )


if __name__ == "__main__":
    main()
