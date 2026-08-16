"""BatchRecipe: the declarative description of one training minibatch.

Slot source kinds (all three code-verified as necessary):
  * tier        -- sample n items from one tier of one memory
                   (Memory Storyboard: curr=short, replay=long);
  * mix         -- sample n items from one memory with per-tier fractions and
                   deficit backfill (BabyCL: pairs 25% short / 75% long);
  * stream_head -- the most recently admitted events of a schema, bypassing
                   memory (the CL-CVCL baseline's include_pushed_new_in_batch).

`strict` slots that cannot be filled fail the recipe; the composer then falls
back to `fallback` (BabyCL: multimodal slot starved early in the stream ->
all-visual batch of 4 segments).  This cross-slot conditional is deliberately
explicit in config instead of hidden in dataloader logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlotSpec:
    name: str
    source: dict  # {"kind": "tier"|"mix"|"stream_head", ...}
    n: int
    unit: str = "frame"  # "frame" | "segment" | "pair" | "clip"
    frames_per_item: int = 1  # frames sampled per item (segment unit)
    clip_len: int = 16  # frames per clip (clip unit)
    clip_stride: int = 1  # take every k-th frame (clip unit)
    views: list[str] = field(default_factory=list)
    text: bool = False  # tokenize text.utterance into the slot
    labels_from: str | None = None  # LabelStore name; resolves seg.id
    batch_label: bool = False  # derive seg.batch_label = item position
    strict: bool = True

    def __post_init__(self):
        kinds = ("tier", "mix", "stream_head")
        if self.source.get("kind") not in kinds:
            raise ValueError(f"slot {self.name!r}: source.kind must be one of {kinds}")
        if self.unit not in ("frame", "segment", "pair", "clip"):
            raise ValueError(f"slot {self.name!r}: bad unit {self.unit!r}")
        if self.unit != "segment" and self.frames_per_item != 1:
            raise ValueError(f"slot {self.name!r}: frames_per_item>1 only for unit=segment")
        if self.unit == "clip" and (self.clip_len < 1 or self.clip_stride < 1):
            raise ValueError(f"slot {self.name!r}: clip_len/clip_stride must be >= 1")
        if not self.views:
            raise ValueError(f"slot {self.name!r} declares no views")


@dataclass
class Recipe:
    slots: list[SlotSpec]
    fallback: Recipe | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> Recipe:
        slots = [SlotSpec(**s) for s in cfg["slots"]]
        fb = cfg.get("fallback")
        return cls(slots=slots, fallback=cls.from_config(fb) if fb else None)

    def slot_names(self) -> list[str]:
        return [s.name for s in self.slots]
