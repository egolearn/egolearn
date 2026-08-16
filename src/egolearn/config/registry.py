"""Per-kind component registries.

A variant of a protocol registers itself under a kind ("stream", "update",
"loss", ...).  Configs address components as "kind:name".  Where the variant
physically lives (a core subpackage or a project directory) is invisible to
the config -- this is what makes swapping variants seamless.

Promotion rule (README, "Where components live"): protocols live in core; a
variant enters a core subpackage once >= 2 projects use it; experimental
variants live inside their project and register locally.
"""

from __future__ import annotations

from collections.abc import Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Callable] = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items and self._items[name] is not obj:
                raise KeyError(f"{self.kind}:{name} already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str):
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown component {self.kind}:{name} (known: {sorted(self._items)})"
            ) from None

    def names(self):
        return sorted(self._items)


# data
STREAMS = Registry("stream")  # dataset sources (data/sources)
DECODERS = Registry("decoder")  # video decode backends (data/decoding)
VIEWS = Registry("view")  # augmentation pipelines (data/views)
# memory
ADMISSION = Registry("admission")  # what enters memory (memory/admission)
UPDATE_POLICIES = Registry("update")  # buffer update policies (memory/update)
SAMPLERS = Registry("sampler")  # buffer sampling policies
EMBEDDERS = Registry("embedder")  # ref -> embedding providers
# model
TRUNKS = Registry("trunk")
HEADS = Registry("head")
LOSSES = Registry("loss")  # model/losses
# evaluation
PROBES = Registry("probe")

_ALL = {
    r.kind: r
    for r in [
        STREAMS,
        DECODERS,
        VIEWS,
        ADMISSION,
        UPDATE_POLICIES,
        SAMPLERS,
        EMBEDDERS,
        TRUNKS,
        HEADS,
        LOSSES,
        PROBES,
    ]
}


def lookup(target: str):
    """Resolve 'kind:name' to a registered component."""
    if ":" not in target:
        raise KeyError(f"component target must be 'kind:name', got {target!r}")
    kind, name = target.split(":", 1)
    if kind not in _ALL:
        raise KeyError(f"unknown registry kind {kind!r} (known: {sorted(_ALL)})")
    return _ALL[kind].get(name)
