"""Key registry: the shared vocabulary of sample-record keys.

Every piece of data flowing between EgoLearn modules is a plain dict
(a "sample record") whose keys are namespaced strings registered here.

Design rules (see README, "What flows between modules"):
  * Structure is CLOSED: components communicate through registered keys and
    named schemas (`egolearn.core.events`).  A typo'd or unregistered key fails
    loudly at first use, not silently at step 40k.
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
    """Runtime first-use check: assert `item` carries all `keys`.

    This is the cheap 90% of static key-flow analysis: components declare
    what they consume and this raises a one-line human error at step 0.
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
VIDEO_REF = register_key("video.ref", "ref", "ItemRef into the video channel (control plane).")
VIDEO_FRAMES = register_key("video.frames", "tensor", "[K,C,H,W] loaded frames (data plane).")
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
