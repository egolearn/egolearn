"""Per-kind component registries (name -> class/function lookup tables).

Each component kind ("stream", "update", "loss", ...) has its own registry.
An implementation registers itself under a short name, and configs refer to
it as "kind:name".  The config never needs to know which file the code lives
in (a core subpackage or a project directory), so one implementation can be
swapped for another by editing only the config.

Promotion rule (README, "Where components live"): protocols live in core; an
implementation moves into a core subpackage once >= 2 projects use it;
experimental ones live inside their project and register locally.
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
ENCODERS = Registry("encoder")  # modality -> representation (model/encoders)
HEADS = Registry("head")
PREDICTORS = Registry("predictor")  # representation(s) -> predicted representation
GENERATIVE = Registry("generative")  # generative/autoregressive components (interface)
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
        ENCODERS,
        HEADS,
        PREDICTORS,
        GENERATIVE,
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
