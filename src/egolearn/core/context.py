"""Run context: clocks and named deterministic RNGs.

Every component gets its notion of time and randomness from a shared Context
object instead of from globals.  Rules this module enforces:
  * Framework components take their randomness from named RNGs handed in
    through the Context.  The unavoidable exception is library code that
    only knows global state -- torch initializes weights and torchvision
    draws augmentation parameters from torch's global generator, and a few
    ported transforms use python's global `random` -- so the Context seeds
    those globals from the configured seed at construction and saves and
    restores their state in snapshots.  Same seed, same run; a resumed run
    continues the exact random sequence.
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
        # Global RNGs: weight init and torchvision transforms draw from
        # torch's global generator, ported PIL transforms from python's
        # global `random`.  Seeding them here is what makes `seed` actually
        # pin the run.  (torch.manual_seed also seeds every CUDA device.)
        random.seed(_stable_seed(self.seed, "global.python"))
        torch.manual_seed(_stable_seed(self.seed, "global.torch"))

    def rng(self, name: str) -> random.Random:
        """Named python RNG, e.g. ctx.rng('memory.segments.sampler').
        Cached: the same name returns the same stateful generator, and its
        state is saved in snapshots."""
        if name not in self._rngs:
            self._rngs[name] = random.Random(_stable_seed(self.seed, name))
        return self._rngs[name]

    def fresh_rng(self, name: str) -> random.Random:
        """A NEW generator seeded from (seed, name) -- not cached, not
        snapshotted.  For one-shot derived randomness that must regenerate
        identically on resume, like an epoch's shuffle order: a cached
        generator would be restored in its already-consumed state and
        shuffle differently the second time."""
        return random.Random(_stable_seed(self.seed, name))

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
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed = state["seed"]
        self.load_clock(state)
        for k, s in state["rngs"].items():
            self.rng(k).setstate(s)
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def load_clock(self, state: dict) -> None:
        """Restore only the clock.  The resume path needs the clock first
        (the replay must know how far to go) but the RNG states last -- see
        utils/snapshot.py for the ordering and why."""
        c = state["clock"]
        self.clock.step = c["step"]
        self.clock.admitted = c["admitted"]
        self.clock.events_emitted = c["events_emitted"]
        self.clock.total_observations = c["total_observations"]
