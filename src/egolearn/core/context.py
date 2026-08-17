"""Run context: clocks and named deterministic RNGs.

Every component gets its notion of time and randomness from a shared Context
object instead of from globals.  Rules this module enforces:
  * No component uses global randomness or global step counters --
    everything is handed in through the Context.
  * There are exactly two clocks in core: `progress` (fraction of the stream
    consumed so far) and `step` (number of gradient steps taken).  Media time
    travels inside items (time.t0/t1).  There are no epochs anywhere.
  * RNGs are looked up by name, and each one is seeded from sha256 of
    (seed, name), so a run produces the same random numbers no matter how
    threads are scheduled or dicts are ordered.  (This deliberately replaces
    `hash(thread_id)`-style seeding.)
"""

from __future__ import annotations

import hashlib
import random

import torch


def _stable_seed(base_seed: int, name: str) -> int:
    h = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "little")


class Clock:
    """Counts progress through a run: `step` is the number of gradient steps
    taken; `admitted` is the number of stream observations consumed.

    `progress` is in [0, 1] when the stream announces a total, and None when
    it does not -- LR schedules must handle both (see solver.ProgressCosine).
    """

    def __init__(self, total_observations: int | None = None):
        self.step = 0
        self.admitted = 0
        self.events_emitted = 0
        self.total_observations = total_observations

    @property
    def progress(self) -> float | None:
        if not self.total_observations:
            return None
        return min(1.0, self.admitted / float(self.total_observations))


class Context:
    def __init__(
        self, seed: int = 1337, device: str = "cuda", total_observations: int | None = None
    ):
        self.seed = int(seed)
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.clock = Clock(total_observations)
        self._rngs: dict[str, random.Random] = {}
        self._torch_gens: dict[str, torch.Generator] = {}

    def rng(self, name: str) -> random.Random:
        """Named python RNG, e.g. ctx.rng('memory.segments.sampler')."""
        if name not in self._rngs:
            self._rngs[name] = random.Random(_stable_seed(self.seed, name))
        return self._rngs[name]

    def torch_gen(self, name: str) -> torch.Generator:
        if name not in self._torch_gens:
            g = torch.Generator()
            g.manual_seed(_stable_seed(self.seed, name))
            self._torch_gens[name] = g
        return self._torch_gens[name]

    # ---- snapshot support (plain save/restore of state; see utils.snapshot)
    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "clock": {
                "step": self.clock.step,
                "admitted": self.clock.admitted,
                "events_emitted": self.clock.events_emitted,
                "total_observations": self.clock.total_observations,
            },
            "rngs": {k: v.getstate() for k, v in self._rngs.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed = state["seed"]
        c = state["clock"]
        self.clock.step = c["step"]
        self.clock.admitted = c["admitted"]
        self.clock.events_emitted = c["events_emitted"]
        self.clock.total_observations = c["total_observations"]
        for k, s in state["rngs"].items():
            self.rng(k).setstate(s)
