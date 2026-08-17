"""Event schemas: named sets of required keys.

A schema is the agreement between admission components (the pieces that
watch the stream and decide what enters memory) and the memory tables that
store what they emit.  An event "type" is deliberately simple: just a named
set of required keys, so checking whether two components fit together is a
plain comparison of key sets (see pipeline.validate).

Governance rule (README, "Adding a schema"): try to express new research with
existing schemas plus extra keys first; add a new schema only as a last
resort -- every additional schema shrinks the set of component combinations
that can work together.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import keys as K


@dataclass(frozen=True)
class EventSchema:
    name: str
    required: frozenset[str]
    doc: str = ""


_SCHEMAS: dict[str, EventSchema] = {}


def register_schema(name: str, required, doc: str = "") -> EventSchema:
    for k in required:
        K.key_spec(k)  # must be registered keys
    schema = EventSchema(name, frozenset(required), doc)
    prev = _SCHEMAS.get(name)
    if prev is not None and prev.required != schema.required:
        raise K.KeyFlowError(f"schema {name!r} re-registered with different keys")
    _SCHEMAS[name] = schema
    return schema


def schema(name: str) -> EventSchema:
    try:
        return _SCHEMAS[name]
    except KeyError:
        raise K.KeyFlowError(f"unknown event schema {name!r} (known: {sorted(_SCHEMAS)})") from None


def make_event(schema_name: str, **fields) -> dict:
    """Build a validated event item. Extra keys are allowed and preserved."""
    sc = schema(schema_name)
    item = dict(fields)
    item[K.EVENT_SCHEMA] = schema_name
    K.require_keys(item, sc.required, f"make_event({schema_name})")
    return item


# --------------------------------------------------------------------------
# Core schemas: the small set of event shapes shared by all projects.
# Method-specific schemas may still be registered from projects;
# register_schema raises on conflicting re-registration.
# --------------------------------------------------------------------------

RAW_FRAME = register_schema(
    "RawFrame",
    [K.TIME_SESSION, K.TIME_T0, K.TIME_T1, K.VIDEO_REF],
    "A single frame flowing straight into memory; labels may be attached lazily via a LabelStore.",
)

VISUAL_SEGMENT = register_schema(
    "VisualSegment",
    [K.TIME_SESSION, K.TIME_T0, K.TIME_T1, K.VIDEO_REF],
    "A finalized temporal event segment. Frame-based methods sample K frames "
    "per segment at load time; clip-based methods load the same "
    "schema as clips.",
)

ALIGNED_PAIR = register_schema(
    "AlignedPair",
    [K.TIME_SESSION, K.TIME_T0, K.TIME_T1, K.VIDEO_REF, K.TEXT_UTTERANCE],
    "A cross-channel aligned tuple: a video interval paired with an utterance "
    "(CVCL/BabyCL multimodal unit; Ego4D narrations fit unchanged).",
)
