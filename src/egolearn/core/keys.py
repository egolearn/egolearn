"""Key registry: the shared vocabulary of sample-record keys.

Every piece of data flowing between EgoLearn modules is a plain dict
(a "sample record") whose keys are namespaced strings registered here.
This file exists so that all modules agree on what those keys are called
and what they mean.

Design rules (see README, "What flows between modules"):
  * Structure is CLOSED: components communicate only through registered keys
    and named schemas (`egolearn.core.events`).  A typo'd or unregistered key
    raises an error the first time it is used, instead of silently producing
    wrong data tens of thousands of steps into a run.
  * Extension is OPEN: projects may register new keys (e.g. a per-sample
    importance weight) without touching core code.  Downstream components
    ignore keys they did not declare.
  * Components treat incoming items as read-only; they extend by adding new
    keys or emitting new items.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class KeyFlowError(RuntimeError):
    """A component required a key that is absent (or unregistered)."""


@dataclass(frozen=True)
class KeySpec:
    name: str
    kind: str  # "ref" | "meta" | "tensor" | "derived"
    doc: str = ""


_REGISTRY: dict[str, KeySpec] = {}


def register_key(name: str, kind: str, doc: str = "") -> str:
    """Register a key. Idempotent if (name, kind) match; error on conflict."""
    if kind not in ("ref", "meta", "tensor", "derived"):
        raise ValueError(f"kind must be ref|meta|tensor|derived, got {kind!r}")
    prev = _REGISTRY.get(name)
    if prev is not None:
        if prev.kind != kind:
            raise KeyFlowError(
                f"key {name!r} already registered with kind={prev.kind!r} (attempted {kind!r})"
            )
        return name
    _REGISTRY[name] = KeySpec(name, kind, doc)
    return name


def key_spec(name: str) -> KeySpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyFlowError(f"key {name!r} is not registered (see egolearn.core.keys)") from None


def registered_keys() -> dict[str, KeySpec]:
    return dict(_REGISTRY)


def require_keys(item: dict, keys: Iterable[str], component: str) -> None:
    """Check at runtime that `item` carries all of `keys`.

    Components declare which keys they consume and call this on their first
    input, so a missing key raises a clear error at the very start of the
    run instead of surfacing much later as wrong results.
    """
    missing = [k for k in keys if k not in item]
    if missing:
        raise KeyFlowError(
            f"{component} requires keys {missing} but item only has {sorted(item.keys())}. "
            "Check the batch recipe's key lists and upstream "
            "admission components' `emits`."
        )


# --------------------------------------------------------------------------
# Core keys.  Namespaces: time.* video.* text.* seg.* weight.* event.*
# --------------------------------------------------------------------------

# identity / time
TIME_SESSION = register_key("time.session", "meta", "Session (video) id the item came from.")
TIME_T0 = register_key("time.t0", "meta", "Interval start, seconds of media time.")
TIME_T1 = register_key("time.t1", "meta", "Interval end (>= t0). Point items have t1 == t0.")

# vision
VIDEO_REF = register_key("video.ref", "ref", "ItemRef into the video channel (no pixels yet).")
VIDEO_FRAMES = register_key("video.frames", "tensor", "[K,C,H,W] decoded frames (actual pixels).")
VIDEO_VIEWS = register_key("video.views", "tensor", "[V,C,H,W] augmented views of one frame.")
VIDEO_CLIP = register_key(
    "video.clip",
    "tensor",
    "[T,C,H,W] contiguous clip with temporally consistent augmentation "
    "(recipe unit=clip; consumed by video models).",
)

# language
TEXT_UTTERANCE = register_key("text.utterance", "meta", "Raw utterance string.")
TEXT_TOKEN_IDS = register_key("text.token_ids", "tensor", "[T] token ids (vocab tokenizer).")
TEXT_LENGTH = register_key("text.length", "meta", "Token count incl. sos/eos.")
UTTERANCE_NUM = register_key("text.utterance_num", "meta", "Source utterance number (provenance).")
TEXT_TIMESTAMPS = register_key(
    "text.timestamps",
    "meta",
    "Raw per-utterance timestamp list (seconds). BabyCL's AN chunk emitter "
    "splits non-contiguous timestamp runs into separate chunks, so the full "
    "list must flow with the utterance, not just [t0, t1].",
)

# segment labels
SEG_ID = register_key(
    "seg.id",
    "derived",
    "Temporal segment / storyboard class id. Resolved through a LabelStore at "
    "batch-composition time; the stored value may be revised by the admission component "
    "(Memory Storyboard reclusters and merges retroactively).",
)
SEG_BATCH_LABEL = register_key(
    "seg.batch_label",
    "derived",
    "Batch-local label (0..S-1) derived by the composer from item grouping. "
    "BabyCL's TCL uses this instead of global seg.id.",
)

# weights / annotations
WEIGHT_SAMPLE = register_key(
    "weight.sample",
    "meta",
    "Optional per-sample loss weight (e.g. clip_importance). Losses opt in "
    "via their `weight_key` argument; all other components pass it through.",
)

# event bookkeeping
EVENT_SCHEMA = register_key(
    "event.schema", "meta", "Name of the EventSchema this item conforms to."
)

# --------------------------------------------------------------------------
# Batch slot fields.  A composed batch is dict[slot_name -> slot dict]; the
# slot dict's fields are the vocabulary below.  Code refers to these
# constants instead of retyping the strings, so a typo fails at import time
# rather than producing a silently empty lookup.  (These are batch-level
# names, distinct from the item-record keys above: an item key travels with
# one sample through admission and memory, a slot field describes one
# collated group inside a batch.)
# --------------------------------------------------------------------------

SLOT_VIEWS = "views"  # FloatTensor [B, V, C, H, W]; clip slots [B, V, T, C, H, W]
SLOT_MODALITY = "modality"  # what the views are ("image"/"video"); routes to an encoder
SLOT_FEATS = "feats"  # FloatTensor [B, V, D] pooled features (model.precompute_feats)
SLOT_LABELS = "labels"  # LongTensor [B]
SLOT_TEXT_IDS = "text_ids"  # LongTensor [B, T]
SLOT_TEXT_LEN = "text_len"  # LongTensor [B]
SLOT_WEIGHTS = "weights"  # FloatTensor [B] or None
SLOT_META = "meta"  # list of source items (refs + payload; read-only)

# --------------------------------------------------------------------------
# Capability names.  A composed model derives its capability set from its
# parts (model.capabilities()); probes declare what they require and the
# pipeline compares the two at build time.  Modality-derived names follow
# the patterns "<modality>_representation" (an encoder of that modality) and
# "<modality>_generation" (a generative component producing it), so new
# modalities mint their capability without touching this file; the constants
# below are the ones framework code spells out.
# --------------------------------------------------------------------------

CAP_IMAGE_REPRESENTATION = "image_representation"
CAP_VIDEO_REPRESENTATION = "video_representation"
CAP_TEXT_REPRESENTATION = "text_representation"
CAP_MULTIMODAL = "multimodal"
CAP_PREDICTION = "prediction"
CAP_TEXT_GENERATION = "text_generation"

KNOWN_CAPABILITIES = frozenset(
    {
        CAP_IMAGE_REPRESENTATION,
        CAP_VIDEO_REPRESENTATION,
        CAP_TEXT_REPRESENTATION,
        CAP_MULTIMODAL,
        CAP_PREDICTION,
        CAP_TEXT_GENERATION,
    }
)
