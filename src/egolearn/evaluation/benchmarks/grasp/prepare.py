"""Build the GRASP Level 2 pair manifest from the upstream videos.

The release ships no metadata: ``level2/P_<Test>/<n>.mp4`` is a plausible
video and ``level2/IP_<Test>/<n>.mp4`` its implausible twin, rendered from
the same seed (the generator seeds each video with its index).  This walks
those folders, pairs twins by file name, hashes every video, and records
whether each pair opens on an identical first frame -- the evidence that the
index pairing is a matched pair and not two unrelated renders.  A video whose
container reports no frame count is an error, because the scorer samples
frames by that count.  The output is never overwritten.
docs/evaluation/grasp.md documents the download.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from egolearn.commands.prepare_parsers import preparation_parser
from egolearn.evaluation.benchmarks.grasp.manifest import (
    IMPOSSIBLE_PREFIX,
    POSSIBLE_PREFIX,
    SOURCE,
    TESTS,
)
from egolearn.hub import sha256_file

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _natural(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _videos(folder: Path) -> list[str]:
    return sorted(
        (
            p.name
            for p in folder.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=_natural,
    )


def discover_pairs(level_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """{test: [(possible, impossible)]} relative paths for every ``P_``/``IP_`` folder pair."""
    possible_dirs = sorted(
        p for p in level_dir.iterdir() if p.is_dir() and p.name.startswith(POSSIBLE_PREFIX)
    )
    if not possible_dirs:
        raise FileNotFoundError(f"no {POSSIBLE_PREFIX}<Test> folders under {level_dir}")
    names = [p.name[len(POSSIBLE_PREFIX) :] for p in possible_dirs]
    unknown = [test for test in names if test not in TESTS]
    if unknown:
        raise ValueError(f"unknown GRASP tests {unknown}; known tests: {sorted(TESTS)}")
    tests: dict[str, list[tuple[str, str]]] = {}
    for possible_dir in possible_dirs:
        test = possible_dir.name[len(POSSIBLE_PREFIX) :]
        impossible_dir = level_dir / f"{IMPOSSIBLE_PREFIX}{test}"
        if not impossible_dir.is_dir():
            raise FileNotFoundError(f"{possible_dir.name} has no {impossible_dir.name} twin folder")
        possible, impossible = _videos(possible_dir), _videos(impossible_dir)
        if possible != impossible:
            odd = sorted(set(possible) ^ set(impossible), key=_natural)
            raise ValueError(f"{test}: P_/IP_ video names differ, e.g. {odd[:3]}")
        if not possible:
            raise FileNotFoundError(f"{possible_dir} holds no videos")
        tests[test] = [
            (f"{possible_dir.name}/{name}", f"{impossible_dir.name}/{name}") for name in possible
        ]
    return tests


def first_frame(path: Path) -> tuple[str, int]:
    """(SHA-256 of the first decoded RGB frame, the frame count the container reports)."""
    import av  # optional dependency (video / train extras)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        digest = hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest()
        return digest, int(stream.frames or 0)


def build_manifest(level_dir: Path, *, log=None) -> dict:
    """Pair, hash and inspect every Level 2 video under `level_dir`."""
    level_dir = Path(level_dir).expanduser()
    tests = discover_pairs(level_dir)
    videos: dict[str, dict] = {}
    pairs = []
    for test, twins in tests.items():
        for possible, impossible in twins:
            starts = []
            for rel in (possible, impossible):
                digest, frames = first_frame(level_dir / rel)
                if not frames:
                    raise ValueError(
                        f"{rel}: the container reports no frame count, which the scorer "
                        "needs to sample frames; re-mux the video (e.g. to MP4)"
                    )
                videos[rel] = {"sha256": sha256_file(level_dir / rel), "frames": frames}
                starts.append(digest)
            pairs.append(
                {
                    "test": test,
                    "scene": Path(possible).stem,
                    "possible": possible,
                    "impossible": impossible,
                    "first_frame_match": starts[0] == starts[1],
                }
            )
        if log is not None:
            log(f"{test}: {len(twins)} pairs")
    return {
        "benchmark": "grasp",
        "level": level_dir.name,
        "source": SOURCE,
        "tests": {
            test: {"concepts": list(TESTS[test]), "n_pairs": len(twins)}
            for test, twins in tests.items()
        },
        "pairs": pairs,
        "videos": videos,
    }


def write_manifest(manifest: dict, output: Path) -> None:
    """Write `manifest` as JSON; an existing file is never overwritten."""
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = preparation_parser("grasp", description=__doc__)
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.data_root, log=print)
        write_manifest(manifest, args.output)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.error(str(error))
    pairs = manifest["pairs"]
    mismatched = sum(not pair["first_frame_match"] for pair in pairs)
    print(
        f"Wrote {len(pairs)} pairs over {len(manifest['tests'])} tests to {args.output}; "
        f"{mismatched} pairs differ on their first frame."
    )


if __name__ == "__main__":
    main()
