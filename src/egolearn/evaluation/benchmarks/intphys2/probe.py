"""IntPhys 2 probe backed by the generic VoE scorer.

``metadata.csv`` lists every video of the Main split with its ``SceneIndex``,
``type`` and dataset-relative ``file_name``.  This adapter verifies it against
the official release by SHA-256, pairs each ``N_Possible`` video with the
``N_Impossible`` video of the same scene, and resolves both paths against a
caller-provided root.  docs/evaluation/intphys2.md documents the pinned download.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from egolearn.config.registry import PROBES
from egolearn.evaluation.voe import VoEProbe

# Main/metadata.csv, identical in IntPhys2.zip and at Hugging Face revision
# a077a2f94e25889016fc6e5983cf21e2ddc25fb2
METADATA_SHA256 = "0db123911e595e263e88dde19ac52bc2684d23d19d7c1f2dc49c67c916075422"


@PROBES.register("intphys2")
class IntPhys2Probe(VoEProbe):
    """VoE accuracy on the 506 matched pairs of the IntPhys 2 Main split."""

    def load_pairs(self) -> list[tuple[str, str]]:
        root = Path(self.data_root)
        raw = (root / "metadata.csv").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != METADATA_SHA256:
            raise ValueError(
                "IntPhys 2 metadata.csv has an unexpected SHA-256: "
                f"expected {METADATA_SHA256}, got {digest}"
            )
        videos = {
            (row["SceneIndex"], row["type"]): str(root / row["file_name"])
            for row in csv.DictReader(raw.decode("utf-8").splitlines())
        }
        missing = [path for path in videos.values() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(videos)} IntPhys 2 videos are missing, e.g. {missing[:3]}"
            )
        return [
            (path, videos[scene, kind.replace("Possible", "Impossible")])
            for (scene, kind), path in videos.items()
            if kind.endswith("_Possible")
        ]

    def metadata(self) -> dict:
        return {**super().metadata(), "metadata_sha256": METADATA_SHA256}
