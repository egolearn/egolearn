"""Buffer update policies (what happens when a full buffer receives a new
item) and sampling policies -- composable, Reverb-style atoms.

These are standard tools, so they live in core: the abstract
UpdatePolicy/Sampler plus the family of standard variants in this one file.
Provenance notes:
  * `reservoir` is the count-based reservoir used by BabyCL's TwoTierBuffer;
  * `reservoir_weighted` is Memory Storyboard's random-priority reservoir
    (Buffer.update): statistically equivalent admission, kept for faithful
    reproduction;
  * `fifo`, `random_replace` complete BabyCL's replay ablation set.
Method-novel policies (MinRed, semantic samplers) live in the project that
researches them and register into the same registries.
"""

from __future__ import annotations

import random

from egolearn.config.registry import SAMPLERS, UPDATE_POLICIES


class UpdatePolicy:
    """Mutates the entry list on admission; returns the displaced entry, if any."""

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
    """Standard count-based reservoir sampling (BabyCL long tier)."""

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
        return entry  # rejected at the door counts as "displaced" for routing

    def state_dict(self):
        return {"seen": self.seen}

    def load_state_dict(self, st):
        self.seen = st["seen"]


@UPDATE_POLICIES.register("reservoir_weighted")
class ReservoirWeighted(UpdatePolicy):
    """Random-priority reservoir (Memory Storyboard's Buffer): every entry
    draws a uniform weight; the table keeps the `capacity` largest weights."""

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
    """Sample without replacement; fewer than n if the table is small."""

    def sample(self, entries, n, rng):
        if n >= len(entries):
            return list(entries)
        return rng.sample(entries, n)
