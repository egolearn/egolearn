"""GRASP Level 2 catalog: the 16 tests, their concepts, and metric labels.

Importable without the video or training extras; `prepare.py` writes the pair
manifest and `probe.py` reads it.  Test names are the folder names of the
upstream release without their ``P_``/``IP_`` prefix.  Concepts follow the
benchmark's own eight (Jassim et al. 2024); a test that combines two
concepts counts toward both, as in Garrido et al. (2025).
"""

from __future__ import annotations

import re

SOURCE = "https://github.com/i-machine-think/grasp"
POSSIBLE_PREFIX = "P_"
IMPOSSIBLE_PREFIX = "IP_"

TESTS: dict[str, tuple[str, ...]] = {
    "Collision": ("collision",),
    "Continuity": ("continuity",),
    "Gravity": ("gravity",),
    "GravityContinuity": ("gravity", "continuity"),
    "GravityInertia": ("gravity", "inertia"),
    "GravityInertia2": ("gravity", "inertia"),
    "GravitySupport": ("gravity", "support"),
    "Inertia": ("inertia",),
    "Inertia2": ("inertia",),
    "ObjectPermanence": ("object_permanence",),
    "ObjectPermanence2": ("object_permanence",),
    "ObjectPermanence3": ("object_permanence",),
    "SolidityContinuity": ("solidity", "continuity"),
    "SolidityContinuity2": ("solidity", "continuity"),
    "Unchangeableness": ("unchangeableness",),
    "Unchangeableness2": ("unchangeableness",),
}
CONCEPTS: tuple[str, ...] = tuple(sorted({c for concepts in TESTS.values() for c in concepts}))


def metric_label(test: str) -> str:
    """CamelCase test name -> metric label: ``GravityInertia2`` -> ``gravity_inertia2``."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", test).lower()


def pair_groups(test: str) -> tuple[str, ...]:
    """The group labels a pair of `test` contributes to: its test and each concept."""
    return (f"test_{metric_label(test)}", *(f"concept_{c}" for c in TESTS[test]))
