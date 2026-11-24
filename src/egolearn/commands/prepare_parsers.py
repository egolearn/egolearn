"""Shared preparation arguments, importable without optional benchmark dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path


def preparation_parser(
    name: str, *, description: str | None = None, prog: str | None = None
) -> argparse.ArgumentParser:
    """Build the same parser for CLI help and the legacy preparation entry point."""
    parser = argparse.ArgumentParser(description=description, prog=prog)
    if name == "shape-bias":
        parser.add_argument("--asset-dir", default="data/shape_bias/source")
        parser.add_argument("--output-dir", default="data/shape_bias/80_unaligned")
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--force-download", action="store_true")
    elif name == "spatial-relations":
        parser.add_argument("--output-dir", type=Path, default=Path("data/spatial_relations/v1"))
        parser.add_argument("--exp3-image-root", type=Path)
        parser.add_argument(
            "--generator-root",
            type=Path,
            help="external simple-relational-reasoning checkout; never modified",
        )
        parser.add_argument("--experiment", choices=("exp1", "exp3", "both"), default="both")
        parser.add_argument("--seed", type=int, default=33)
        parser.add_argument("--n-per-condition", type=int, default=1024)
        parser.add_argument("--validate-only", action="store_true")
    elif name == "spatial-monitoring":
        parser.add_argument("--full-root", type=Path, default=Path("data/spatial_relations/v1"))
        parser.add_argument(
            "--output-dir", type=Path, default=Path("data/spatial_relations/monitoring_v1")
        )
        parser.add_argument("--seed", type=int, default=33)
        parser.add_argument("--exp1-per-condition", type=int, default=64)
        parser.add_argument("--exp3-configurations", type=int, default=16)
    elif name == "babyview-4afc":
        parser.add_argument("--data_root", type=Path, required=True)
        parser.add_argument("--vocab_json", type=Path)
        parser.add_argument("--out_dir", type=Path, required=True)
        parser.add_argument("--download", action="store_true")
        parser.add_argument(
            "--download_only", action="store_true", help="fetch the crops; write no manifests"
        )
        parser.add_argument("--n_trials", type=int, default=4150)
        parser.add_argument("--n_foils", type=int, default=3)
        parser.add_argument("--trials_seed", type=int, default=0)
        parser.add_argument("--split_seed", type=int, default=0)
        parser.add_argument("--test_frac", type=float, default=0.2)
    elif name == "labeled-s-4afc":
        parser.add_argument("--data_root", type=Path, required=True)
        parser.add_argument("--trials_json", type=Path, required=True)
        parser.add_argument("--vocab_json", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
    elif name == "konkle-4afc":
        parser.add_argument(
            "--data_root",
            type=Path,
            required=True,
            help="directory holding one folder of photographs per object category",
        )
        parser.add_argument(
            "--vocab_json", type=Path, required=True, help="the model's training vocab.json"
        )
        parser.add_argument("--output", type=Path, required=True, help="trial manifest to write")
        parser.add_argument("--n_trials", type=int, default=100, help="trials per category")
        parser.add_argument("--seed", type=int, default=0)
    elif name == "grasp":
        parser.add_argument(
            "--data_root",
            type=Path,
            required=True,
            help="the unpacked level2 directory holding the P_<Test>/ and IP_<Test>/ folders",
        )
        parser.add_argument("--output", type=Path, required=True, help="manifest.json to write")
    else:
        raise ValueError(f"unknown preparer {name!r}")
    return parser
