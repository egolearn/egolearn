"""Update policies (what a full buffer does when a new item arrives) and
sampling policies (how items are drawn from a buffer to build a batch).

These are standard, method-agnostic building blocks, which is why they live
in core: the abstract UpdatePolicy/Sampler classes plus the common variants,
all in this one file.  Where each variant comes from:
  * `reservoir` is the count-based reservoir sampling used by BabyCL's
    TwoTierBuffer (a reservoir keeps a uniform random subset of the stream);
  * `reservoir_weighted` is Memory Storyboard's random-priority reservoir
    (Buffer.update): its admission is statistically equivalent to
    `reservoir`, but it is kept as its own policy for faithful reproduction;
  * `fifo` (first in, first out) and `random_replace` complete BabyCL's
    replay ablation set.
Policies invented by a specific method (MinRed, semantic samplers) live in
that method's project and register into these same registries.
"""

from __future__ import annotations

import random

from egolearn.config.registry import SAMPLERS, UPDATE_POLICIES


class UpdatePolicy:
    """Decides how a new item is inserted into a table's entry list.  `admit`
    may modify the list in place and returns the item it pushed out, if any."""

    def reset(self) -> None:
        pass

    def admit(
        self, entries: list[dict], entry: dict, capacity: int, rng: random.Random
    ) -> dict | None:
        raise NotImplementedError  # pragma: no cover

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, st: dict) -> None:
        pass


@UPDATE_POLICIES.register("fifo")
class Fifo(UpdatePolicy):
    def admit(self, entries, entry, capacity, rng):
        entries.append(entry)
        if len(entries) > capacity:
            return entries.pop(0)
        return None


@UPDATE_POLICIES.register("reservoir")
class Reservoir(UpdatePolicy):
    """Standard count-based reservoir sampling: keeps a uniform random subset
    of every item seen so far (used by BabyCL's long tier)."""

    def __init__(self):
        self.seen = 0

    def reset(self):
        self.seen = 0

    def admit(self, entries, entry, capacity, rng):
        self.seen += 1
        if len(entries) < capacity:
            entries.append(entry)
            return None
        j = rng.randint(0, self.seen - 1)
        if j < capacity:
            displaced = entries[j]
            entries[j] = entry
            return displaced
        return entry  # not admitted: return the new item as "displaced" so overflow routing sees it

    def state_dict(self):
        return {"seen": self.seen}

    def load_state_dict(self, st):
        self.seen = st["seen"]


@UPDATE_POLICIES.register("reservoir_weighted")
class ReservoirWeighted(UpdatePolicy):
    """Random-priority reservoir (Memory Storyboard's Buffer): every entry
    draws a uniform random weight, and the table keeps the `capacity` entries
    with the largest weights.  The result is a uniform random subset, same as
    `reservoir`."""

    def admit(self, entries, entry, capacity, rng):
        entry = dict(entry)
        entry["_weight"] = rng.random()
        entries.append(entry)
        if len(entries) <= capacity:
            return None
        worst = min(range(len(entries)), key=lambda i: entries[i].get("_weight", 0.0))
        return entries.pop(worst)


@UPDATE_POLICIES.register("random_replace")
class RandomReplace(UpdatePolicy):
    def admit(self, entries, entry, capacity, rng):
        if len(entries) < capacity:
            entries.append(entry)
            return None
        j = rng.randint(0, capacity - 1)
        displaced = entries[j]
        entries[j] = entry
        return displaced


class Sampler:
    def sample(self, entries: list[dict], n: int, rng: random.Random) -> list[dict]:
        raise NotImplementedError  # pragma: no cover


@SAMPLERS.register("uniform")
class Uniform(Sampler):
    """Sample uniformly without replacement; returns fewer than n items when
    the table holds fewer than n."""

    def sample(self, entries, n, rng):
        if n >= len(entries):
            return list(entries)
        return rng.sample(entries, n)
