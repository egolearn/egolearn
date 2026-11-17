"""BatchRecipe: a config-level description of what one training minibatch
contains.  A recipe is a list of slots; each slot (one named part of the
batch) says where its items come from and how they should be loaded.

Slot source kinds (all three code-verified as necessary):
  * tier        -- sample n items from one tier (one age band) of one memory
                   (Memory Storyboard: curr=short, replay=long);
  * mix         -- sample n items from one memory with per-tier fractions;
                   if one tier has too few items, the remainder is drawn from
                   the other tiers (BabyCL: pairs 25% short / 75% long);
  * stream_head -- the most recently admitted events of a schema, taken
                   directly rather than sampled from memory (the CL-CVCL
                   baseline's include_pushed_new_in_batch).

If a `strict` slot cannot be filled, the whole recipe fails and the composer
retries with `fallback` instead (BabyCL: early in the stream the multimodal
slot has no items yet, so it falls back to an all-visual batch of 4
segments).  This dependency between slots is deliberately spelled out in
config instead of being hidden inside dataloader logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def modality_for_unit(unit: str) -> str:
    """What a unit's views are: clips are video, everything else loads
    frames.  The single home of this rule -- the offline driver and the
    slot default both call it."""
    return "video" if unit == "clip" else "image"


@dataclass
class SlotSpec:
    name: str
    source: dict  # {"kind": "tier"|"mix"|"stream_head", ...}
    n: int
    unit: str = "frame"  # "frame" | "segment" | "pair" | "clip"
    modality: str | None = None  # of the views; defaults from unit (clip -> video, else image)
    frames_per_item: int = 1  # frames sampled per item (segment unit)
    clip_len: int = 16  # frames per clip (clip unit)
    clip_stride: int = 1  # take every k-th frame (clip unit)
    views: list[str] = field(default_factory=list)
    text: bool = False  # tokenize text.utterance into the slot
    labels_from: str | None = None  # LabelStore name; seg.id is looked up there at batch time
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
        if self.text and self.frames_per_item != 1:
            raise ValueError(
                f"slot {self.name!r}: text=true with frames_per_item>1 would "
                "collate one utterance against several frames; not supported"
            )
        if self.modality is None:
            self.modality = modality_for_unit(self.unit)


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
