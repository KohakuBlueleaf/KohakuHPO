# The optimizer catalogue

All optimizers implement the same cube-level ask/tell contract and are selected through the
`OPTIMIZER` registry: a name string, or a `{"name": ..., **kwargs}` dict for options.

| key | method | character |
|---|---|---|
| `s3turbo` | **S3-TuRBO** (Soft-Sparse Scout TuRBO), the flagship | trust-region batch Thompson sampling + a coordinate-mask axis + a scout/escape axis; constants derived from the problem |
| `turbo` | TuRBO (Eriksson et al. 2019) with its default batch-TS acquisition | strong local exploiter; single trust region |
| `warped_turbo` | `turbo` + HEBO's input/output warps | for non-stationary or skewed objectives |
| `gpbo` | global exact-GP BO, acquisition over a Sobol pool (default `log_ei`), constant-liar batching | the textbook baseline; strong at low dim, collapses at high dim |
| `hebo` | warped-GP + rank-combined EI/PI/UCB (our from-scratch HEBO) | robust general-purpose BO |
| `cmaes` | CMA-ES via the `cmaes` library | strong non-GP baseline; cheap per ask |
| `sobol` | scrambled Sobol | quasirandom coverage floor |
| `random` | uniform random | the mandatory sanity floor |

`budget=<total evals>` should be passed to `s3turbo` when known: it sharpens the budget-derived
constants (scout cadence, focus window, region count).

## S3-TuRBO in one paragraph

The base is a trust region with **batch Thompson sampling**: a Sobol candidate pool inside the
box, `q` joint GP-posterior draws, the pool-minimizer of each draw is the batch (no acquisition
constants, batch diversity for free). Two orthogonal axes extend it. The **mask** decides which
coordinates a local step moves: `dense` (all), `hard` (a Bernoulli subset, others frozen), `soft`
(a polarized Beta weight per coordinate, containing the other two as limits), or `adaptive` (the
soft mask with its density and polarization *learned online* from which coordinates pay off; see
[below](#the-adaptive-soft-mask-learn-the-mask-from-the-run)). The **scout** decides how far basins
are reached, with four headline strategies: `none`, `random` (periodic far probes), `switch` (a
protected main path plus a bounded *focus burst* that concentrates `q-1` slots on a promising
candidate basin, the mechanism that reaches hidden narrow cores), or `reactive` (the *adaptive
escape*: it does not predict whether a far basin exists, since a local search's data cannot reveal
that, but reacts to evidence, keeping a small always-on base scout rate set by the `escape_k` dial and
tracking an escape value `E` that rises when a planted candidate region proves
spatially distinct from the incumbent and competitive, and decays when candidates prove redundant,
so scout rate and focus-burst commitment scale with `E`). A third knob `tr_update` counts trust-region success/failure per `batch` (standard, safe)
or per `point` (faster box collapse, wins smooth funnels). Everything else is derived from
`(d, q, budget, x0)` or the observed value scale.

The default mask is `adaptive`, which runs the soft mask and learns its concentration online from
which coordinates pay off. It is the best-ranked mask in the ablation below and the sensible choice
when you do not know the regime; a fixed mask is fine when you do.

Full math, propositions and derivations: [s3turbo-method.md](s3turbo-method.md), or the
interactive [project page](../webpage/index.html).

## Configuring: which scout to use

The default is `adaptive · reactive · batch` with `escape_k=0.75`: the adaptive mask self-selects (it
learns its concentration online), the trust-region update stays `batch`, and the scout is the
evidence-gated `reactive` escape. The escape axis reduces to one derived dial, `escape_k`, which sets a
base scout rate `rho_0 = 1/(escape_k * sqrt(d))`: large `escape_k` approaches pure-local `none`, small
`escape_k` scouts aggressively toward `switch`. Benchmarked against CMA-ES / GP-BO / HEBO / TuRBO / Sobol,
the default ranks *first* on the real sklearn HPO suite (avg rank 1.80 vs 2.20 for pure-local `none`) and
second only to `none` on the smooth synthetic suite, and in the mask ablation the adaptive mask ranks best.
The two axes are separate contributions: the soft-sparse mask carries the unimodal and simple regime on its
own (with the scout off), and the scout is the opt-in addition for landscapes that are, or might be,
multi-basin.

| scout | use when |
|---|---|
| `reactive` (default, `escape_k=0.75`) | the regime is unknown; a light evidence-gated escape that stays near pure-local on smooth landscapes and ramps up where structure appears. Ranks first on real sklearn HPO, second only to `none` on the smooth synthetic suite. Aliases `adaptive`, `evidence` |
| `none` | you know the problem is unimodal or only mildly multimodal; the mask alone is the contribution and the scout has nothing to escape, so it is a small drag. Best on the smooth synthetic suite (avg rank 2.00 vs 2.25 for reactive there) |
| `switch` | genuine multi-basin escape: many far basins, one plausibly hiding a better narrow core; the strongest raw escape and the only scout that survives the hard escape case, but it hurts on smooth tasks so it is not a good default |
| `random` | the most general escape but weak: a periodic uniform far probe is little more than restarting a fresh search elsewhere; a naive baseline, so it only sometimes helps and does not justify being on by default |

`reactive` is the adaptive escape axis, the escape-axis analogue of the adaptive mask: it reads the
run and spends escape budget in proportion to evidence. It does *not* try to predict
whether a far basin exists (a local search's surrogate never samples the far region, so its data
cannot reveal that); instead it always spends the derived base rate `rho_0` set by `escape_k` and tracks
an escape value `E` in [0,1] that a spatially-distinct-and-competitive candidate raises and a
redundant-or-worse one lowers, scaling scout rate and focus-burst commitment by `E`. Its one non-derived
input is `escape_k` (the base-rate dial); the distinctness threshold reuses the already-derived novelty
radius, not a tuned threshold. On the smooth synthetic suite the escape costs almost nothing (reactive
sits second only to `none`); on the many-basin core-hit test a smaller `escape_k` reaches narrower far
cores (`escape_k=0.1` matches `switch` on the hardest); and on *real* sklearn HPO it ranks *first*
overall, which is why it is the default. Set `escape_k` in `0.5`-`1.0` (smaller for spaces likely to
hide separated basins, larger toward pure-local for near-unimodal problems).

```python
khpo.minimize(f, space, {"name": "s3turbo", "budget": 300}, budget=300, q=4)  # reactive, escape_k=0.75
khpo.minimize(f, space, {"name": "s3turbo", "scout_strategy": "none", "budget": 300}, budget=300, q=4)  # known-simple: mask only
```

The scout only matters when a local search would be trapped. On a barrier-separated many-basin task
(start in a shallow near basin, deep basins far across a high plateau), an ablation over 16 seeds shows
`switch` reaches a deep core 100% of the time (regret 0.007) while `none`, `random`, and external methods
like TuRBO stay stuck in the near basin (regret 0.50); only CMA-ES also escapes, more slowly. On smoother
landscapes the scout contributes little, so setting `scout_strategy="none"` (mask only) is the right call
when the problem is known to be simple; `switch` is reserved for the hidden-core case.

The fixed masks (`dense`, `hard`, `soft`) and the named `preset`s (`balanced`, `rugged`, `smooth`,
`soft_smooth`, `heterogeneous`, `multibasin`, each a fixed `(mask, scout, tr_update)` triple) are
kept as static reference configurations for the ablation; pick one only when the regime is known and
a fully fixed setup is wanted.

## Manual control of the three axes

Every axis is a plain constructor argument; a preset only fills the axes you did not set, and an
explicit axis always wins (with nothing set, the defaults are `adaptive · reactive · batch` with
`escape_k=0.75`):

```python
# fully manual
khpo.minimize(f, space, {
    "name": "s3turbo",
    "mask_distribution": "soft",   # dense | hard | soft | adaptive
    "scout_strategy": "switch",    # none | random | switch | reactive
    "tr_update": "point",          # batch | point
    "budget": 300,
}, budget=300, q=4)

# preset as base, one axis overridden -> soft . switch . batch
khpo.minimize(f, space, {"name": "s3turbo", "preset": "heterogeneous",
                         "scout_strategy": "switch", "budget": 300}, budget=300, q=4)
```

Accepted aliases: mask `bernoulli`/`sparse` -> `hard`, `beta`/`soft_beta` -> `soft`, `none` ->
`dense`; scout `off` -> `none`, `probe` -> `random`, `focus` -> `switch`, `adaptive`/`evidence` ->
`reactive`. A user-registered
`MASK` or `SCOUT` entry (see [extending.md](extending.md)) is selectable by its own key through
the same arguments.

The remaining knobs: `budget` (total expected evaluations; sharpens the budget-derived constants),
`risk` (`"conservative" | "balanced" | "aggressive"`, how eagerly scouted basins are accepted),
and `value_scale` / `noise_scale` (seed the robust value scale when known).

## The adaptive soft mask (learn the mask from the run)

The fixed masks assume you *know* the regime: `hard` for rugged, `soft` for anisotropic, `dense`
for pure convex. The adaptive mask removes that assumption. It runs the soft Beta family, but
learns its two shape parameters online (the target active density `rho` and the polarization
`c0`) from which coordinates have actually paid off so far. Turn it on by asking for the
`adaptive` mask:

```python
khpo.minimize(f, space, {"name": "s3turbo", "mask_distribution": "adaptive",
                         "budget": 200}, budget=200, q=4)
```

Aliases: `adaptive`, `adaptive_soft`, `auto_soft` all select the soft mask with adaptation on.

### How it learns

The soft mask has two shape parameters: the active fraction `rho` (how many coordinates move) and
the concentration `c0` (how polarized the move is, hard vs uniform). The adaptive mask leaves `rho`
at its derived value `1/sqrt(d)` (section 3.2) and learns only `c0` from observation, because `rho`
already has a good prior value from the active-set assumption while `c0` has no such prior.

Every coordinate carries a **credit** `s_j`, an exponential-memory estimate of how much moving it
has improved the incumbent. When a batch is told, each improving point casts its normalized
improvement distributed over coordinates by its **squared realized displacement** from the region
center (not by the mask weight `alpha`, which is coordinate-symmetric for the soft mask and carries
no per-coordinate signal). Writing $\delta_{ij}=|u_{ij}-c_j|$ for how far coordinate $j$ actually
moved in point $i$:

$$
\Delta_i=\frac{\max(0,\ y_{\text{best}}-y_i)}{S_y},\qquad
s_j\leftarrow\lambda\,s_j+\sum_i \Delta_i\,\frac{\delta_{ij}^2}{\sum_k \delta_{ik}^2},
$$

with $\lambda$ = `mask_credit_decay` the memory factor and $S_y$ the robust value scale. From the
credit the mask reads its concentration, and sets `c0` by it:

$$
p_j=\frac{s_j}{\sum_k s_k},\qquad
C=1-\frac{H(p)}{\log d},\quad H(p)=-\sum_j p_j\log p_j,\qquad
c_{0,t}=\exp\!\big((1-C)\log c_{\max}+C\log c_{\min}\big).
$$

$C\in[0,1]$ is a **confidence** that rises from 0 (credit spread evenly, nothing learned) toward 1
(credit concentrated). When $C\to0$ the mask stays soft and wide ($c_{\max}$), still probing which
coordinates matter; when $C\to1$ it sharpens toward a hard, polarized mask ($c_{\min}$) on the
learned active set, walking the `soft` <-> `hard` edge of the family (Proposition 2) as evidence
accrues.

> **Algorithm A: adaptive mask update.**
> ```
> on_tell(batch U, values y, previous best y_best):
>   s ← λ·s                                       # decay coordinate credit
>   for each improving non-scout point i:
>     Δ   ← (y_best − y_i) / S_y                   # normalized improvement (>0)
>     δ²  ← (u_i − center)²                         # realized squared displacement
>     s   ← s + Δ · δ² / sum(δ²)                    # attribute by where it actually moved
>
> mask_shape (per soft draw):
>   ρ_t  ← 1/sqrt(d)                                # derived active fraction, NOT learned
>   if sum(s) ≈ 0:  c0_t ← c0_default;  return
>   p    ← s / sum(s)
>   C    ← clip(1 − entropy(p)/log d, 0, 1)
>   c0_t ← exp((1−C)·log c_max + C·log c_min)
> ```

### Adaptive-mask knobs

| argument | default | meaning |
|---|---|---|
| `mask_distribution="adaptive"` | (none) | enable adaptive soft masking (aliases `adaptive_soft`, `auto_soft`) |
| `mask_concentration` | `0.4` | initial `c0` before any credit accrues |
| `mask_min_concentration` | `0.03` | hard-like `c0` floor, reached at full confidence |
| `mask_max_concentration` | `1.2` | soft/wide `c0` ceiling, used at zero confidence |
| `mask_credit_decay` | `0.92` | credit memory `lambda`: higher = longer memory, lower = adapts faster but noisier |

Only the concentration is learned; the active fraction rho stays at its derived value `1/sqrt(d)`
(section 3.2). An ablation confirmed that deriving rho online (from k_eff) is worse than the fixed
`1/sqrt(d)`, so rho is left at its derived value and only c0 is adaptive. The credit defaults are
validated; the credit signal
(incumbent-improvement, attributed by realized displacement) is the main surface for future work
(rank-weighted batch credit, GP-lengthscale priors, per-region credit).

### What the benchmarks show

Two questions, both on a 9-task synthetic suite (Hartmann6 d=6; Ackley, Griewank, Powell,
Rastrigin, Levy, Rosenbrock, Styblinski-Tang and the many-basin family at d=25; budget 120-250,
q=4, 6 seeds; lower is better, ranked per task). Reproduce both with
`python examples/11_benchmark_suite.py --workers 16 --seeds 6`.

**Against prior work** (scout = random, adaptive mask), S3-TuRBO ranks first:

| method | avg rank |
|---|---:|
| **s3turbo (adaptive)** | **1.56** |
| turbo | 2.00 |
| CMA-ES | 2.83 |
| GP-BO | 4.39 |
| HEBO | 4.39 |
| Sobol | 5.83 |

**Mask-axis ablation** (scout = random, tr_update = batch, only the mask varies), the adaptive mask
is the best mask and every mask beats the no-mask (dense) baseline:

| mask | avg rank |
|---|---:|
| **adaptive** | **1.78** |
| hard | 2.11 |
| soft (fixed) | 2.11 |
| dense (no mask) | 4.00 |

The mask axis clearly helps (all three masks beat dense), and the adaptive mask edges out the fixed
soft and hard masks without being told the regime, a narrow but real win, and the reason it is the
default. It does not dominate the fixed masks by a wide margin: on a landscape whose regime you know,
the matching fixed mask is a fine choice.

### Recommended settings

```python
# default: adaptive mask + reactive escape (regime unknown)
{"name": "s3turbo", "mask_distribution": "adaptive", "scout_strategy": "reactive", "escape_k": 0.75, "budget": B}
# known simple / unimodal: mask only, scout off (mask is the contribution; escape has nothing to do)
{"name": "s3turbo", "mask_distribution": "adaptive", "scout_strategy": "none", "budget": B}
# leaning multi-basin / multi-family: scout harder with a smaller dial
{"name": "s3turbo", "mask_distribution": "adaptive", "scout_strategy": "reactive", "escape_k": 0.5, "budget": B}
# hidden far cores near-certain: the strongest raw escape
{"name": "s3turbo", "mask_distribution": "adaptive", "scout_strategy": "switch", "budget": B}
```

Keep `tr_update="batch"` with the adaptive mask. The default `reactive` (`escape_k=0.75`) is the best
all-round choice when the regime is unknown; set `scout_strategy="none"` when you know the problem is
simple (the mask alone wins there), lower `escape_k` toward `0.5` when separated basins are likely, and
use `switch` only when you strongly expect hidden narrow cores and have the budget for the focus burst.
`random` is the most general escape but weak, so it only sometimes helps.

## Choosing q (batch size)

`q` trades surrogate freshness (small q refits more often) against per-fit diversity and
wall-clock parallelism. For plain local search, q=1..2 converges fastest per evaluation; for the
`multibasin` preset q>=4 is required, because the focus burst spends `q-1` slots on the candidate
basin and one on the protected main path.

## When NOT to use this library

Cheap functions with huge eval budgets (use CMA-ES directly or gradient methods), differentiable
objectives (use gradients), heavily categorical/conditional spaces, dimensions in the many
hundreds, or multi-objective problems (not yet supported).
