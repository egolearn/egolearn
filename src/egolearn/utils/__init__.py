"""Utility helpers; training dependencies load only when a snapshot helper is used."""

__all__ = ["load_snapshot", "save_snapshot"]


def __getattr__(name: str):
    if name in __all__:
        from egolearn.utils import snapshot

        return getattr(snapshot, name)
    raise AttributeError(f"module 'egolearn.utils' has no attribute {name!r}")
