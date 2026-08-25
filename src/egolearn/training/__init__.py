"""Training: two parallel loops -- streaming (driver) and traditional IID
(offline) -- plus optimizer/schedule policy (solver), periodic actions
(hooks), and stage-latency measurement (profiling)."""

from .driver import LockstepDriver
from .hooks import CheckpointHook, Hook, LogHook, ProbeHook, SnapshotHook
from .offline import OfflineDriver
from .profiling import StageTimer
from .solver import ProgressCosine, apply_lr, build_optimizer

__all__ = [
    "CheckpointHook",
    "Hook",
    "LockstepDriver",
    "LogHook",
    "OfflineDriver",
    "ProbeHook",
    "ProgressCosine",
    "SnapshotHook",
    "StageTimer",
    "apply_lr",
    "build_optimizer",
]
