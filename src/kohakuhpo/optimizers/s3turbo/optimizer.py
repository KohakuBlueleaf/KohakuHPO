"""S3-TuRBO (Soft-Sparse Scout TuRBO): trust-region Thompson sampling with two orthogonal axes.

A single-incumbent trust region with batch Thompson sampling (as ``turbo``) extended by two
independent plug-in choices:

* ``mask_distribution`` (the local-move axis, :data:`~kohakuhpo.registry.MASK`): how a local
  candidate perturbs coordinates: ``dense`` moves everything, ``hard`` freezes all but a
  Bernoulli subset, ``soft`` mixes each coordinate by a polarized Beta weight, and ``adaptive``
  (the default) runs the soft law but LEARNS its concentration online from which coordinates have
  paid off (the active fraction stays the derived schedule), so no regime needs to be declared.
* ``scout_strategy`` (the escape axis, :data:`~kohakuhpo.registry.SCOUT`): how far basins are
  reached: ``none``, ``random`` (periodic far probes + mining), ``sidecar`` (protected main +
  bounded side channel), ``switch`` (sidecar + a bounded dense focus burst on a promising basin),
  and ``reactive`` (the adaptive escape: spends escape budget in proportion to evidence, where
  candidates that prove spatially distinct from the incumbent raise it and redundant ones lower it,
  so it stays near ``none`` on single-basin problems and ramps up on multi-basin ones).

Because the adaptive mask self-selects and the ``reactive`` scout is the default, the one knob a user
typically sets is ``escape_k``: the base scout rate ``rho_0 = 1/(escape_k*sqrt d)`` interpolates from
near-``none`` behavior (large ``escape_k``, near-unimodal landscapes) to near-``switch`` behavior
(small ``escape_k``, multi-basin landscapes), so a single continuous dial replaces the discrete escape
choice. The default ``escape_k=0.75`` (recommended range 0.5--1.0) ranks first against the baselines on
real HPO, second only to pure-local on the smooth synthetic suite, and retains real multi-basin escape. A
reference knob ``tr_update`` sets trust-region success/failure counting per ``batch`` (the
proposed default, standard TuRBO) or per ``point`` (faster box collapse, retained only for
ablation; a smaller effective ``q`` gives the same effect). Every other internal constant is
derived in :meth:`S3Turbo._derive` from the problem itself (``d``, ``q``, ``budget``, presence of
``x0``) or from the observed value scale. The named ``preset`` s are legacy fixed-mask
configurations kept only as ablation references against the adaptive mask (see
``docs/optimizers.md`` and the method note).
"""

import numpy as np
import torch
from scipy.stats import qmc

from kohakuhpo.device import tensor_kw
from kohakuhpo.optimizer import Optimizer
from kohakuhpo.optimizers.s3turbo.masks import MASK_ALIASES
from kohakuhpo.optimizers.s3turbo.regions import Region
from kohakuhpo.optimizers.s3turbo.scouts import SCOUT_ALIASES
from kohakuhpo.registry import MASK, OPTIMIZER, SCOUT
from kohakuhpo.surrogate import GP

# Legacy fixed (mask, scout, tr_update) triples, kept as ablation references only; prefer setting
# mask_distribution + scout_strategy directly.
PRESETS = {
    "balanced": ("hard", "random", "batch"),
    "rugged": ("hard", "sidecar", "batch"),
    "smooth": ("dense", "none", "point"),
    "soft_smooth": ("soft", "none", "point"),
    "heterogeneous": ("soft", "none", "batch"),
    "multibasin": ("hard", "switch", "batch"),
}


DEFAULT_MASK = "adaptive"
DEFAULT_SCOUT = "reactive"


@OPTIMIZER.register("s3turbo")
class S3Turbo(Optimizer):
    """Trust-region batch Thompson sampler; adaptive soft mask by default.

    With no arguments the optimizer uses the **adaptive soft mask** (it learns its concentration
    online; the active fraction stays the derived ``1/sqrt(d)``) and the **reactive scout** at the
    all-round-best dial ``escape_k=0.75`` (recommended range 0.5--1.0), first against the baselines on
    real HPO, second only to pure-local on the smooth synthetic suite (§7). The single escape knob is
    ``escape_k``: the base scout rate is ``rho_0 = 1/(escape_k*sqrt d)``,
    so a large ``escape_k`` (little scouting) suits near-unimodal landscapes and a small one
    (aggressive scouting) suits multi-basin landscapes, one continuous dial spanning the behavior of
    ``none`` (no escape) through ``switch`` (strong escape). ``scout_strategy``
    (``none`` / ``random`` / ``sidecar`` / ``switch`` / ``reactive``), ``mask_distribution``
    (``dense`` / ``hard`` / ``soft`` / ``adaptive``) and ``tr_update`` (``batch`` default;
    ``point`` retained as an ablation reference) remain overridable. The six named ``preset`` s are
    legacy fixed-mask ablation references, not the recommended path. ``budget`` sharpens the
    derived schedule; ``risk`` sets the acceptance margin; ``value_scale`` / ``noise_scale`` seed
    the robust value scale.
    """

    def __init__(
        self,
        space,
        seed: int = 0,
        preset: str | None = None,
        mask_distribution: str | None = None,
        scout_strategy: str | None = None,
        tr_update: str | None = None,
        budget: int | None = None,
        risk: str = "balanced",
        escape_k: float = 0.75,
        value_scale: float | None = None,
        noise_scale: float | None = None,
        adaptive_mask: bool = False,
        mask_concentration: float = 0.4,
        mask_min_concentration: float = 0.03,
        mask_max_concentration: float = 1.2,
        mask_credit_decay: float = 0.92,
    ):
        super().__init__(space, seed)
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"preset {preset!r}; choices {sorted(PRESETS)}")
        p_mask, p_scout, p_tr = (
            PRESETS[preset] if preset else (DEFAULT_MASK, DEFAULT_SCOUT, "batch")
        )
        mask_distribution = mask_distribution if mask_distribution is not None else p_mask
        scout_strategy = scout_strategy if scout_strategy is not None else p_scout
        tr_update = tr_update if tr_update is not None else p_tr
        if mask_distribution in {"adaptive", "adaptive_soft", "auto_soft"}:
            adaptive_mask = True
            mask_distribution = "soft"
        # Aliases first, then any user-registered MASK/SCOUT entry by its own key.
        mask_key = MASK_ALIASES.get(mask_distribution, mask_distribution)
        scout_key = SCOUT_ALIASES.get(scout_strategy, scout_strategy)
        if mask_key not in MASK:
            raise ValueError(f"mask_distribution {mask_distribution!r}; choices {MASK.keys()}")
        if scout_key not in SCOUT:
            raise ValueError(f"scout_strategy {scout_strategy!r}; choices {SCOUT.keys()}")
        if tr_update not in {"batch", "point"}:
            raise ValueError(f"tr_update {tr_update!r}; choices ['batch', 'point']")
        self.tr_update = tr_update
        self.mask = mask_key
        self.scout_name = scout_key
        self.scout = SCOUT.get(self.scout_name)()
        self.budget = int(budget) if budget and budget > 0 else None
        self.c_accept = {"conservative": 0.10, "balanced": 0.25, "aggressive": 0.40}.get(risk, 0.25)
        self.escape_k = float(escape_k) if escape_k and escape_k > 0 else 0.75
        self.value_scale = value_scale
        self.noise_scale = noise_scale
        self.adaptive_mask = bool(adaptive_mask and mask_key == "soft")
        self.mask_concentration = float(mask_concentration)
        self.mask_min_concentration = float(mask_min_concentration)
        self.mask_max_concentration = float(mask_max_concentration)
        self.mask_credit_decay = float(mask_credit_decay)
        self.mask_rho_current = 0.0
        self.mask_concentration_current = self.mask_concentration
        self.mask_effective_dim = float(self.dim)
        self.mask_confidence = 0.0
        self._mask_credit = np.zeros(self.dim)
        self._mask_event_score = 0.0
        self._ask_alpha_buffer: list[np.ndarray] = []
        self._last_alpha: list[np.ndarray] = []
        self._sobol = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        self._rng = np.random.default_rng(seed + 991)
        torch.manual_seed(seed)  # seeds every device; posterior draws use the global RNG
        self._regions: list[Region] = []
        self._mined: set[tuple] = set()
        self._ask_count = 0
        self._global_fail = 0
        self._improved_this_tell = False
        self._last_ridx: list[int] = []
        self._last_kind: list[str] = []
        self._focus_idx: int | None = None
        self._focus_left = 0
        self._escape_value = 0.0
        self._q: int | None = None
        self._derive(1)

    # ---- derived constants (from d, q, budget, x0; see the method note, section 5) ----------- #
    def _derive(self, q):
        q = max(1, int(q))
        if self._q == q:
            return
        self._q = q
        d = max(1, self.dim)
        batches = (self.budget / q) if self.budget else 75.0
        self.n_init = 2 * q
        self.pool = max(256, 64 * q)
        self.max_data = int(np.clip(4 * d, 64, 128))
        self.succ_tol = (
            3  # grow after 3 consecutive improving batches (standard TuRBO success streak)
        )
        self.rho = float(1.0 / np.sqrt(d))
        d_eff = self.rho * d if self.mask != "dense" else d
        self.fail_tol = max(4, int(np.ceil(d_eff / q)))
        self.novel_dist = float(0.6 / np.sqrt(6.0))
        self.l_max = 1.0
        self.l_init = 1.0
        self.l_min = float(self.novel_dist / 8.0)
        self.scout_far_pool = max(2048, 64 * d)
        self.max_regions = (
            1 if self.scout_name in {"none", "random"} else 1 + min(3, max(1, int(batches // 20)))
        )
        f_scout, f_focus = (
            0.06,
            0.15,
        )  # scout / focus cadence fractions: strategy defaults, not derived (§5.5)
        self.scout_period = 0 if self.scout_name == "none" else max(1, int(np.ceil(0.25 / f_scout)))
        self.stagnation_after = max(4, int(np.ceil(1.5 * self.fail_tol)))
        self.cand_radius = float(max(self.l_init * 0.9, 1.5 * self.novel_dist))
        self.cand_warmup = max(4, int(np.ceil(1.5 * q)))
        self.focus_slots = max(1, q - 1)
        self.focus_batches = max(2, int(np.ceil(f_focus * batches)))
        self.beta = 1.0
        self.eta = 1.0
        self.escape_base = 1.0 / (self.escape_k * np.sqrt(d))
        self.escape_decay = float(1.0 - 1.0 / max(2, self.max_regions))

    def _score_scale(self):
        """Robust value scale ``S_y = max(IQR, 1.4826 MAD, noise, value_scale, floor)``."""
        pieces = []
        if self.value_scale:
            pieces.append(float(self.value_scale))
        if self.noise_scale:
            pieces.append(float(self.noise_scale))
        if len(self.y) >= 4:
            q25, q75 = np.quantile(self.y, [0.25, 0.75])
            med = float(np.median(self.y))
            pieces += [float(q75 - q25), float(1.4826 * np.median(np.abs(self.y - med)))]
        if len(self.y):
            pieces.append(0.02 * max(1.0, abs(float(self.y.min()))))
        return max([p for p in pieces if np.isfinite(p) and p > 0] or [1.0])

    def _rho(self, region):
        if not self.adaptive_mask:
            return self.rho
        self._refresh_adaptive_mask()
        return self.mask_rho_current

    def _refresh_adaptive_mask(self):
        """Set the mask shape parameters. rho (the active fraction) is the derived 1/sqrt(d); only
        the Beta concentration c0 is learned from observation: confidence C (credit concentration)
        sharpens it toward c_min (hard) when the active coordinates separate, and keeps it near
        c_max (soft) while they have not.
        """
        self.mask_rho_current = float(self.rho)
        total = float(self._mask_credit.sum())
        if total <= 1e-12:
            self.mask_concentration_current = self.mask_concentration
            self.mask_effective_dim = float(self.dim)
            self.mask_confidence = 0.0
            return
        probs = (self._mask_credit + 1e-12) / (total + 1e-12 * self.dim)
        entropy = float(-(probs * np.log(probs)).sum())
        confidence = float(np.clip(1.0 - entropy / np.log(max(self.dim, 2)), 0.0, 1.0))
        k_eff = float(np.clip(1.0 / np.square(probs).sum(), 1.0, self.dim))
        lo = np.log(max(self.mask_min_concentration, 1e-3))
        hi = np.log(max(self.mask_max_concentration, self.mask_min_concentration + 1e-3))
        self.mask_concentration_current = float(np.exp((1.0 - confidence) * hi + confidence * lo))
        self.mask_effective_dim = k_eff
        self.mask_confidence = confidence

    # ---- masked local Thompson step ----------------------------------------------------------- #
    def _apply_mask(self, region, box, dense_override=False):
        name = "dense" if dense_override else self.mask
        rho = self._rho(region)
        if name == "soft":
            concentration = (
                self.mask_concentration_current if self.adaptive_mask else self.mask_concentration
            )
            alpha = MASK.get(name)(self._rng, len(box), self.dim, rho, concentration=concentration)
        else:
            alpha = MASK.get(name)(self._rng, len(box), self.dim, rho)
        moved = np.clip(region.center[None] + alpha * (box - region.center[None]), 0.0, 1.0)
        return moved, np.abs(moved - region.center[None])

    def _local_ts(self, region, n, dense_override=False):
        if n <= 0:
            return np.empty((0, self.dim))
        cand_region = region.kind == "candidate"
        if cand_region and region.visits <= self.cand_warmup:
            raw = region.center[None] + self._rng.uniform(-0.5, 0.5, (n, self.dim)) * region.radius
            box, disp = self._apply_mask(region, np.clip(raw, 0.0, 1.0), dense_override)
            self._ask_alpha_buffer.extend(disp.copy())
            return box
        tu, ty = self._train_near(
            region.center, region.radius if cand_region else None, cand_region
        )
        gp = GP(torch.tensor(tu, **tensor_kw()), torch.tensor(ty, **tensor_kw()))
        raw = self._sobol.random(self.pool)
        lo = np.clip(region.center - region.radius / 2, 0.0, 1.0)
        hi = np.clip(region.center + region.radius / 2, 0.0, 1.0)
        box, disp = self._apply_mask(region, lo + raw * (hi - lo), dense_override)
        draws = gp.sample(torch.tensor(box, **tensor_kw()), n)
        idx = torch.argmin(draws, dim=1).cpu().numpy()
        self._ask_alpha_buffer.extend(disp[idx].copy())
        return box[idx]

    def _train_near(self, center, radius=None, local_only=False):
        """Training set for a region GP: region-local points when enough exist, else nearest+best."""
        if len(self.y) == 0:
            return self.U, self.y
        d2 = ((self.U - center[None]) ** 2).sum(1)
        if radius is not None and local_only:
            loc = np.where(np.sqrt(d2 / self.dim) <= radius)[0]
            if len(loc) >= 4:
                idx = loc[np.argsort(self.y[loc])[: self.max_data]]
                return self.U[idx], self.y[idx]
        if len(self.y) <= self.max_data:
            return self.U, self.y
        near = np.argsort(d2)[: max(1, self.max_data // 2)]
        best = np.argsort(self.y)[: self.max_data - len(near)]
        idx = np.unique(np.concatenate([near, best]))[: self.max_data]
        return self.U[idx], self.y[idx]

    def _farthest_point(self):
        """Far probe: the pool point maximizing the distance to all observed points."""
        cand = self._sobol.random(self.scout_far_pool)
        if len(self.U) == 0:
            return cand[0]
        d2 = ((cand[:, None, :] - self.U[None]) ** 2).sum(2).min(1)
        return cand[int(np.argmax(d2))]

    # ---- regions ------------------------------------------------------------------------------ #
    def _ensure_main(self):
        if self._regions or len(self.y) == 0:
            return
        c, y = self.best
        r = Region(c, self.l_init, "main")
        r.best_y, r.best_u = float(y), c.copy()
        self._regions.append(r)

    def _main_index(self):
        self._ensure_main()
        for i, r in enumerate(self._regions):
            if r.kind == "main":
                return i
        return int(np.argmin([r.best_y for r in self._regions])) if self._regions else None

    def _candidate_index(self):
        cands = [
            (self._importance(r), i) for i, r in enumerate(self._regions) if r.kind == "candidate"
        ]
        return int(min(cands)[1]) if cands else None

    def _importance(self, region):
        """Keep-score: best value minus S_y-scaled uncertainty and novelty bonuses (lower keeps)."""
        s = self._score_scale()
        unc = np.sqrt(np.log(len(self.y) + 2.0) / (region.visits + 1.0))
        others = [r.center for r in self._regions if r is not region]
        nov = min(
            (np.linalg.norm(region.center - o) / np.sqrt(self.dim) for o in others), default=0.0
        )
        return region.best_y - self.beta * s * unc - self.eta * s * nov

    def _rank_regions(self):
        self._ensure_main()
        warm = [i for i, r in enumerate(self._regions) if r.warmup > 0]
        warm.sort(key=lambda i: (-self._regions[i].warmup, self._regions[i].best_y))
        cold = sorted(
            (i for i, r in enumerate(self._regions) if r.warmup <= 0),
            key=lambda i: self._importance(self._regions[i]),
        )
        return warm + cold

    def _novel(self, u):
        if not self._regions:
            return True
        return (
            min(np.linalg.norm(u - r.center) / np.sqrt(self.dim) for r in self._regions)
            >= self.novel_dist
        )

    def _accept(self, y, quantile, margin_k):
        """Scout acceptance gate: below the value quantile, or within a scaled margin of the best."""
        if len(self.y) < self.n_init:
            return False
        thr = float(np.quantile(self.y, quantile))
        return bool(y <= thr or y <= float(self.y.min()) + margin_k * self._score_scale())

    def _focus_gate(self, y, quantile):
        if len(self.y) < self.n_init:
            return False
        thr = float(np.quantile(self.y, quantile))
        return bool(y <= thr or y <= float(self.y.min()) + self.c_accept * self._score_scale())

    def _add_candidate(self, u, y):
        if self.max_regions <= 1:
            return False
        key = tuple(np.round(u, 6))
        if key in self._mined or not self._novel(u):
            return False
        r = Region(u, self.cand_radius, "candidate", warmup=self.cand_warmup)
        r.best_y, r.best_u, r.visits = float(y), u.copy(), 1
        self._regions.append(r)
        self._mined.add(key)
        self._drop()
        return True

    def _mine_archive(self, quantile, margin_k):
        if len(self.y) < self.n_init or len(self._regions) >= self.max_regions:
            return
        for idx in np.argsort(self.y):
            if len(self._regions) >= self.max_regions:
                break
            if self._accept(float(self.y[idx]), quantile, margin_k):
                self._add_candidate(self.U[idx], float(self.y[idx]))

    def _drop(self):
        while len(self._regions) > self.max_regions:
            protected = {i for i, r in enumerate(self._regions) if r.kind == "main" or r.warmup > 0}
            order = np.argsort([self._importance(r) for r in self._regions])[::-1]
            drop = next((int(i) for i in order if int(i) not in protected), None)
            if drop is None:
                drop = next(
                    (int(i) for i in order if self._regions[int(i)].kind != "main"), int(order[0])
                )
            self._regions.pop(drop)
            if self._focus_idx is not None and drop == self._focus_idx:
                self._focus_idx, self._focus_left = None, 0
            elif self._focus_idx is not None and drop < self._focus_idx:
                self._focus_idx -= 1

    @property
    def arm_gate(self):
        """Escape-value threshold to arm a reactive focus burst: half the base rate. Low enough that a
        genuinely distinct find commits a burst, high enough that below-base noise cannot."""
        return 0.5 * self.escape_base

    def _valid_focus(self):
        return (
            self.scout_name in {"switch", "reactive"} and self._focus_idx is not None and self._focus_left > 0
            and 0 <= self._focus_idx < len(self._regions)
            and self._regions[self._focus_idx].kind == "candidate"
        )  # fmt: skip

    def _update_escape_value(self):
        """Reactive escape signal in [0,1]. Each worked candidate region is judged against the main
        region: a candidate that is spatially *distinct* (center separated by more than the novelty
        radius) and *competitive* (not clearly worse) is evidence that separated basins exist and
        pushes the value toward 1; a candidate that has drifted back near the main region, or is
        clearly worse, pushes it toward 0. On a single-basin landscape planted candidates prove
        redundant and the value decays; on a multi-basin one a distinct competitive candidate raises
        it. No landscape is predicted; only the outcome of candidates already planted is measured.
        """
        mi = self._main_index()
        if mi is None:
            return
        main = self._regions[mi]
        scale = self._score_scale()
        decay = self.escape_decay
        for r in self._regions:
            if r.kind != "candidate" or r.visits < self.cand_warmup + 1:
                continue
            sep = float(np.sqrt(((r.center - main.center) ** 2).mean()))
            distinct = sep >= self.novel_dist
            not_much_worse = r.best_y <= main.best_y + 3.0 * scale
            target = 1.0 if (distinct and not_much_worse) else 0.0
            self._escape_value = decay * self._escape_value + (1.0 - decay) * target

    # ---- batch helpers shared by the scout strategies ------------------------------------------ #
    def _local_one(self, ridx, dense=False):
        if ridx is None:
            return self._sobol.random(1)[0], -1, "sobol"
        return self._local_ts(self._regions[ridx], 1, dense)[0], ridx, "local"

    def _region_batch(self, ridx, n, dense=False):
        """``n`` local picks in one region from a SINGLE GP fit + pool (q joint posterior draws,
        the TS_q of the method note); one fit per slot would be q-times slower for nothing."""
        if n <= 0:
            return [], [], []
        if ridx is None:
            return [self._sobol.random(1)[0] for _ in range(n)], [-1] * n, ["sobol"] * n
        picks = self._local_ts(self._regions[ridx], n, dense)
        return list(picks), [ridx] * n, ["local"] * n

    def _main_batch(self, n):
        """``n`` local picks on the protected main region (sidecar/switch allocation)."""
        return self._region_batch(self._main_index(), max(0, n))

    def _local_batch(self, n):
        """``n`` local picks round-robin over ranked regions (none/random allocation), grouped so
        each region fits its GP once per ask."""
        ranked = self._rank_regions()
        counts: dict[int | None, int] = {}
        for slot in range(max(0, n)):
            ri = ranked[slot % len(ranked)] if ranked else None
            counts[ri] = counts.get(ri, 0) + 1
        picks, ridx, kinds = [], [], []
        for ri, c in counts.items():
            p, r, k = self._region_batch(ri, c)
            picks.extend(p)
            ridx.extend(r)
            kinds.extend(k)
        return picks, ridx, kinds

    def _kind_at(self, i):
        return self._last_kind[i] if i < len(self._last_kind) else "?"

    # ---- ask / tell ----------------------------------------------------------------------------- #
    def _ask(self, q):
        self._derive(q)
        self._ask_alpha_buffer = []
        self._last_alpha = []
        if len(self.y) < self.n_init:
            self._last_ridx, self._last_kind = [-1] * q, ["init"] * q
            self._last_alpha = [np.zeros(self.dim) for _ in range(q)]
            return self._sobol.random(q)
        self._ensure_main()
        self._ask_count += 1
        picks, ridx, kinds = self.scout.select(self, q)
        self._last_ridx, self._last_kind = ridx[:q], kinds[:q]
        self._last_alpha = self._ask_alpha_buffer[:q]
        return np.array(picks[:q])

    def tell(self, U, y):
        ub = np.atleast_2d(np.asarray(U, float))
        yb = np.asarray(y, float).reshape(-1)
        before = float(self.y.min()) if len(self.y) else float("inf")
        super().tell(ub, yb)
        self._ensure_main()
        if self.tr_update == "batch":
            touched = {}
            for i, (u, val) in enumerate(zip(ub, yb, strict=False)):
                ridx = self._last_ridx[i] if i < len(self._last_ridx) else -1
                if 0 <= ridx < len(self._regions) and self._kind_at(i) != "scout":
                    rec = touched.get(ridx)
                    if rec is None:
                        touched[ridx] = [u, float(val), 1]
                    else:
                        rec[2] += 1
                        if val < rec[1]:
                            rec[0], rec[1] = u, float(val)
            for ridx, (u, val, count) in touched.items():
                self._update_region(self._regions[ridx], u, float(val), count)
        else:
            for i, (u, val) in enumerate(zip(ub, yb, strict=False)):
                ridx = self._last_ridx[i] if i < len(self._last_ridx) else -1
                if 0 <= ridx < len(self._regions) and self._kind_at(i) != "scout":
                    self._update_region(self._regions[ridx], u, float(val), 1)
        self._improved_this_tell = float(self.y.min()) < before - 1e-9
        self._update_adaptive_mask(ub, yb, before)
        self._global_fail = 0 if self._improved_this_tell else self._global_fail + 1
        self._update_escape_value()
        self.scout.on_tell(self, ub, yb)
        self._drop()

    def _update_adaptive_mask(self, ub, yb, before):
        """Accrue per-coordinate credit from improving points.

        Each improving point casts credit equal to its normalized improvement, distributed over
        coordinates by its squared realized displacement (normalized to sum 1 per point). A move
        that improved by displacing coordinate j far credits j; a symmetric no-signal move spreads
        credit uniformly and cancels out under the participation ratio.
        """
        if not self.adaptive_mask or len(self._last_alpha) == 0 or not np.isfinite(before):
            return
        self._mask_credit *= float(np.clip(self.mask_credit_decay, 0.0, 1.0))
        scale = self._score_scale()
        gain = 0.0
        for i, val in enumerate(yb):
            if i >= len(self._last_alpha) or self._kind_at(i) == "scout":
                continue
            improvement = max(0.0, before - float(val)) / scale
            if improvement <= 0.0:
                continue
            disp = np.asarray(self._last_alpha[i], dtype=float)
            if disp.shape != (self.dim,):
                continue
            sq = disp * disp
            denom = float(sq.sum())
            if denom <= 1e-18:
                continue
            self._mask_credit += improvement * (sq / denom)
            gain += improvement
        self._mask_event_score = gain

    def _update_region(self, region, u, y, count):
        region.visits += count
        region.warmup = max(0, region.warmup - count)
        if y < region.best_y - 1e-9:
            region.best_y, region.best_u, region.center = y, u.copy(), u.copy()
            region.succ += 1
            region.fail = 0
            if region.succ >= self.succ_tol:
                region.radius = min(self.l_max, region.radius * 1.5)
                region.succ = 0
        else:
            region.fail += 1
            region.succ = 0
            if region.fail >= self.fail_tol:
                region.radius = max(self.l_min, region.radius / 2.0)
                region.fail = 0


OPTIMIZER.register("soft_sparse_scout_turbo")(S3Turbo)
