"""The escape axis: scout strategies in the :data:`~kohakuhpo.registry.SCOUT` registry.

A strategy owns three decisions: when to spend a scout slot (``want_scout``), how to allocate the
batch across regions and moves (``select``), and whether an observation becomes a candidate region
(``on_tell``). Behavioral constants (acceptance quantile, focus quantile, candidate probability,
slot count) belong to the strategy as class attributes. Subclass :class:`ScoutStrategy` and
register it to extend S3-TuRBO's escape axis without touching the optimizer.
"""

import numpy as np

from kohakuhpo.registry import SCOUT


class ScoutStrategy:
    """Base strategy: no escape channel (a single region, pure local search).

    ``opt`` is the owning :class:`~kohakuhpo.optimizers.s3turbo.optimizer.S3Turbo`, which provides
    the shared machinery strategies call into: ``_local_batch`` / ``_main_batch`` / ``_local_one``
    (masked local Thompson picks), ``_farthest_point`` (far probe), ``_accept`` / ``_focus_gate``
    (value gates), ``_add_candidate`` / ``_mine_archive`` (region creation), and the focus fields.
    """

    mines = False
    accept_q = 0.0
    scout_slots = 0

    def want_scout(self, opt) -> bool:
        return False

    def select(self, opt, q):
        return opt._local_batch(q)

    def on_tell(self, opt, ub, yb) -> None:
        return None


@SCOUT.register("none")
class NoScout(ScoutStrategy):
    pass


@SCOUT.register("random")
class RandomScout(ScoutStrategy):
    """Periodic far probes + archive mining; accepted finds are polished as candidates."""

    mines = True
    accept_q = 0.65
    accept_margin_k = 2.0
    scout_slots = 1

    def want_scout(self, opt) -> bool:
        return opt.scout_period > 0 and opt._ask_count % opt.scout_period == 0

    def select(self, opt, q):
        n_scout = self.scout_slots if self.want_scout(opt) else 0
        picks, ridx, kinds = opt._local_batch(q - n_scout)
        for _ in range(n_scout):
            picks.append(opt._farthest_point())
            ridx.append(-1)
            kinds.append("scout")
        return picks, ridx, kinds

    def on_tell(self, opt, ub, yb) -> None:
        for i, (u, val) in enumerate(zip(ub, yb, strict=False)):
            if opt._kind_at(i) == "scout" and opt._accept(
                float(val), self.accept_q, self.accept_margin_k
            ):
                opt._add_candidate(u, float(val))
        opt._mine_archive(self.accept_q, self.accept_margin_k)


@SCOUT.register("sidecar")
class SidecarScout(ScoutStrategy):
    """Protected main path + a bounded side channel; promotes only explicit scout finds."""

    mines = False
    accept_q = 0.55
    accept_margin_k = 1.0
    candidate_prob = 0.75
    scout_slots = 1

    def want_scout(self, opt) -> bool:
        periodic = opt.scout_period > 0 and opt._ask_count % opt.scout_period == 0
        return periodic or opt._global_fail >= opt.stagnation_after

    def select(self, opt, q):
        n_side = min(self.scout_slots if self.want_scout(opt) else 0, max(0, q - 1))
        picks, ridx, kinds = opt._main_batch(q - n_side)
        for _ in range(n_side):
            ci = opt._candidate_index()
            if ci is not None and opt._rng.random() < self.candidate_prob:
                u, ri, k = opt._local_one(ci)
            else:
                u, ri, k = opt._farthest_point(), -1, "scout"
            picks.append(u)
            ridx.append(ri)
            kinds.append(k)
        return picks, ridx, kinds

    def on_tell(self, opt, ub, yb) -> None:
        for i, (u, val) in enumerate(zip(ub, yb, strict=False)):
            if opt._kind_at(i) == "scout" and opt._accept(
                float(val), self.accept_q, self.accept_margin_k
            ):
                opt._add_candidate(u, float(val))


@SCOUT.register("switch")
class SwitchScout(SidecarScout):
    """Sidecar + a bounded dense focus burst on a promising candidate (hidden-core hunter)."""

    focus_q = 0.35

    def select(self, opt, q):
        if opt._valid_focus():
            k = min(opt.focus_slots, max(1, q - 1))
            picks, ridx, kinds = opt._region_batch(int(opt._focus_idx), k, dense=True)
            mp, mr, mk = opt._main_batch(q - len(picks))
            opt._focus_left -= 1
            return picks + mp, ridx + mr, kinds + mk
        return super().select(opt, q)

    def _arm_focus(self, opt, y) -> None:
        if opt._focus_gate(float(y), self.focus_q):
            cands = [i for i, r in enumerate(opt._regions) if r.kind == "candidate"]
            if cands:
                opt._focus_idx = min(cands, key=lambda i: opt._regions[i].best_y)
                opt._focus_left = opt.focus_batches

    def on_tell(self, opt, ub, yb) -> None:
        for i, (u, val) in enumerate(zip(ub, yb, strict=False)):
            if opt._kind_at(i) == "scout" and opt._accept(
                float(val), self.accept_q, self.accept_margin_k
            ):
                if opt._add_candidate(u, float(val)):
                    self._arm_focus(opt, val)
        # mine at most one archive candidate per scout tick
        if opt._focus_idx is None and self.want_scout(opt) and len(opt._regions) < opt.max_regions:
            for idx in np.argsort(opt.y):
                if opt._focus_gate(float(opt.y[idx]), self.focus_q):
                    if opt._add_candidate(opt.U[idx], float(opt.y[idx])):
                        self._arm_focus(opt, float(opt.y[idx]))
                        break
        if opt._focus_idx is not None and opt._improved_this_tell:
            opt._focus_left = max(opt._focus_left, opt.focus_batches // 2)


@SCOUT.register("reactive")
class ReactiveScout(SwitchScout):
    """Reactive escape: a derived, dial-controlled dose of speculative escape, reallocated by evidence.

    No score can tell from a local search's data whether a far basin exists (the surrogate has
    never sampled there), so ``reactive`` does not try to classify the landscape. It always spends a
    small always-on base rate of far probes ``rho_0 = 1/(escape_k*sqrt d)`` (the escape dial: large
    ``escape_k`` scouts little, small ``escape_k`` scouts aggressively), and lets the *outcome* of the
    candidate regions it has already planted modulate that spend. A running escape value
    ``opt._escape_value`` in [0,1] rises with each recently-worked candidate that is spatially distinct
    from the incumbent (by the derived novelty radius) and competitive, and falls otherwise; the
    far-probe rate interpolates from ``rho_0`` at no evidence up to 1 as it saturates, and the focus
    burst, the expensive commitment, is armed only once the value clears half the base rate, so a
    single below-base blip cannot trigger one. The base rate is the load-bearing term (a principled,
    k-controlled dose); the evidence value is a bounded reallocation on top of it. ``rho_0``, the
    value's memory ``escape_decay``, and the distinctness radius are all derived (§5); nothing tuned.
    """

    def want_scout(self, opt) -> bool:
        if opt._valid_focus():
            return True
        rho0 = opt.escape_base
        rate = rho0 + (1.0 - rho0) * opt._escape_value
        return bool(opt._rng.random() < rate)

    def _arm_focus(self, opt, y) -> None:
        if opt._escape_value > opt.arm_gate:
            super()._arm_focus(opt, y)

    def on_tell(self, opt, ub, yb) -> None:
        super().on_tell(opt, ub, yb)
        if not opt._valid_focus() and opt._escape_value > opt.arm_gate:
            cands = [i for i, r in enumerate(opt._regions) if r.kind == "candidate"]
            if cands:
                best = min(cands, key=lambda i: opt._regions[i].best_y)
                r = opt._regions[best]
                if opt._focus_gate(r.best_y, self.focus_q) and r.radius > opt.l_min * 2:
                    opt._focus_idx = best
                    opt._focus_left = opt.focus_batches


SCOUT_ALIASES = {"none": "none", "off": "none", "random": "random", "probe": "random",
                 "sidecar": "sidecar", "switch": "switch", "focus": "switch",
                 "reactive": "reactive", "adaptive": "reactive", "evidence": "reactive"}  # fmt: skip
