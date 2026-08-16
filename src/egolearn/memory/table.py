"""Memory tables.

Control-plane only: a table holds event items (refs + small metadata), never
tensors.  Admission/update/sampling decisions are single-threaded and
deterministic given the run seed (named RNGs).

Two routing topologies, both code-verified:
  * BabyCL   : new -> short only; FIFO overflow spills into long
               (`admit_to: [short], overflow: {short: long}`);
  * Storyboard: new -> short AND long independently, no spill
               (`admit_to: [short, long], overflow: {}`).
"""

from __future__ import annotations

import random

from egolearn.config.registry import SAMPLERS, UPDATE_POLICIES
from egolearn.core.context import Context


class Table:
    def __init__(
        self,
        name: str,
        capacity: int,
        update: str = "fifo",
        sampler: str = "uniform",
        admit_stride: int = 1,
    ):
        """`admit_stride`: accept every k-th offered entry (default: all).
        Memory Storyboard's short FIFO subsamples the stream (--subsample:
        every 8th frame) while its long reservoir ingests every frame --
        per-tier admission striding is a standard knob, not method logic."""
        if capacity <= 0:
            raise ValueError(f"Table {name!r} needs positive capacity")
        self.name = name
        self.capacity = int(capacity)
        self.update_policy = UPDATE_POLICIES.get(update)()
        self.sampler = SAMPLERS.get(sampler)()
        self.admit_stride = max(1, int(admit_stride))
        self.entries: list[dict] = []
        self.total_admitted = 0
        self.total_offered = 0

    def admit(self, entry: dict, rng: random.Random) -> dict | None:
        self.total_offered += 1
        if (self.total_offered - 1) % self.admit_stride != 0:
            return None
        self.total_admitted += 1
        return self.update_policy.admit(self.entries, entry, self.capacity, rng)

    def sample(self, n: int, rng: random.Random) -> list[dict]:
        return self.sampler.sample(self.entries, n, rng)

    @property
    def full(self) -> bool:
        return len(self.entries) >= self.capacity

    def __len__(self):
        return len(self.entries)

    def state_dict(self) -> dict:
        return {
            "entries": list(self.entries),
            "total_admitted": self.total_admitted,
            "total_offered": self.total_offered,
            "update": self.update_policy.state_dict(),
        }

    def load_state_dict(self, st: dict) -> None:
        self.entries = list(st["entries"])
        self.total_admitted = st["total_admitted"]
        self.total_offered = st.get("total_offered", st["total_admitted"])
        self.update_policy.load_state_dict(st.get("update", st.get("eviction", {})))


class TieredMemory:
    def __init__(
        self,
        name: str,
        tiers: dict[str, dict],
        admit_to: list[str],
        overflow: dict[str, str] | None = None,
    ):
        self.name = name
        self.tiers: dict[str, Table] = {
            tname: Table(f"{name}.{tname}", **spec) for tname, spec in tiers.items()
        }
        for t in admit_to:
            self._check_tier(t)
        self.admit_to = list(admit_to)
        self.overflow = dict(overflow or {})
        for src, dst in self.overflow.items():
            self._check_tier(src)
            self._check_tier(dst)

    def _check_tier(self, t: str) -> None:
        if t not in self.tiers:
            raise KeyError(f"memory {self.name!r} has no tier {t!r} (tiers: {sorted(self.tiers)})")

    def _rng(self, ctx: Context, tier: str) -> random.Random:
        return ctx.rng(f"memory.{self.name}.{tier}")

    def admit(self, event: dict, ctx: Context) -> None:
        for tname in self.admit_to:
            evicted = self.tiers[tname].admit(event, self._rng(ctx, tname))
            # cascade spill (BabyCL: short FIFO overflow -> long reservoir)
            src = tname
            while evicted is not None and src in self.overflow:
                dst = self.overflow[src]
                evicted = self.tiers[dst].admit(evicted, self._rng(ctx, dst))
                src = dst

    # ---- sampling -----------------------------------------------------------
    def sample_tier(self, tier: str, n: int, ctx: Context) -> list[dict]:
        self._check_tier(tier)
        return self.tiers[tier].sample(n, self._rng(ctx, tier + ".sample"))

    def sample_mix(self, n: int, fractions: dict[str, float], ctx: Context) -> list[dict]:
        """Quota per tier with deficit backfill (BabyCL TwoTierBuffer.sample):
        integer quotas from fractions; any shortfall is refilled from the
        other tiers' remaining entries."""
        quotas = {t: int(n * f) for t, f in fractions.items()}
        # give rounding leftovers to the largest-fraction tier
        leftover = n - sum(quotas.values())
        if leftover > 0 and quotas:
            top = max(fractions, key=lambda t: fractions[t])
            quotas[top] += leftover

        picked: list[dict] = []
        picked_ids = set()
        for t, q in quotas.items():
            self._check_tier(t)
            for e in self.tiers[t].sample(q, self._rng(ctx, t + ".sample")):
                if id(e) not in picked_ids:
                    picked.append(e)
                    picked_ids.add(id(e))
        if len(picked) < n:  # deficit backfill from all tiers
            pool = [e for t in self.tiers for e in self.tiers[t].entries if id(e) not in picked_ids]
            rng = self._rng(ctx, "backfill")
            take = min(n - len(picked), len(pool))
            for e in rng.sample(pool, take) if take < len(pool) else pool:
                picked.append(e)
                picked_ids.add(id(e))
        return picked[:n]

    def tier_full(self, tier: str) -> bool:
        self._check_tier(tier)
        return self.tiers[tier].full

    def stats(self) -> str:
        return "{}[{}]".format(
            self.name, ", ".join(f"{t}={len(tab)}/{tab.capacity}" for t, tab in self.tiers.items())
        )

    def state_dict(self) -> dict:
        return {t: tab.state_dict() for t, tab in self.tiers.items()}

    def load_state_dict(self, st: dict) -> None:
        for t, s in st.items():
            self.tiers[t].load_state_dict(s)
