"""Dataset catalog and preparation commands, independent of the training stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from egolearn import hub


@dataclass(frozen=True)
class Dataset:
    name: str
    access: str
    description: str
    source: str
    collection: str | None = None
    prepare_module: str | None = None
    prepare_defaults: tuple[tuple[str, str], ...] = ()


_BENCHMARKS = "egolearn.evaluation.benchmarks."
DATASETS = (
    Dataset(
        "shape-bias",
        "public",
        "Prepared shape-bias stimuli",
        "https://github.com/alexatartaglini/developmental-shape-bias",
        "shape-bias",
        _BENCHMARKS + "shape_bias.prepare",
        (("--asset-dir", ".sources/shape-bias"), ("--output-dir", "shape-bias")),
    ),
    Dataset(
        "spatial-relation-categorization",
        "public",
        "Hosted spatial relation monitoring subsets (monitoring/)",
        "https://doi.org/10.1016/j.cognition.2023.105690",
        "spatial-relation-categorization",
    ),
    Dataset(
        "spatial-relations",
        "public",
        "Prepared spatial relation trials",
        "https://doi.org/10.1016/j.cognition.2023.105690",
        None,
        _BENCHMARKS + "spatial_relations.prepare",
        (("--output-dir", "spatial-relations"),),
    ),
    Dataset(
        "spatial-monitoring",
        "public",
        "Balanced spatial monitoring subsets",
        "https://doi.org/10.1016/j.cognition.2023.105690",
        None,
        _BENCHMARKS + "spatial_relations.subset",
        (
            ("--full-root", "spatial-relations"),
            ("--output-dir", "spatial-monitoring"),
        ),
    ),
    Dataset(
        "babyview-4afc",
        "public",
        "BabyView trial/split metadata; Hub payload excludes pixels",
        "https://github.com/babyview-project/ccn-2026-object-variability",
        "babyview-4afc",
        _BENCHMARKS + "babyview_4afc.prepare",
        (("--data_root", "babyview-public-crops"), ("--out_dir", "babyview-4afc")),
    ),
    Dataset(
        "labeled-s-4afc",
        "IRB",
        "Local trials referencing restricted Labeled-S images",
        "https://nyu.databrary.org/volume/564/",
        None,
        _BENCHMARKS + "labeled_s_4afc.prepare",
        (("--data_root", "labeled-s"), ("--output", "labeled-s-4afc/manifest.json")),
    ),
    Dataset(
        "labeled-s-splits",
        "public",
        "Split identifiers only; images require separate access",
        "https://github.com/wkvong/multimodal-baby",
        "labeled-s-splits",
    ),
    Dataset(
        "saycam",
        "IRB",
        "SAYCam videos and extracted frames",
        "https://nyu.databrary.org/volume/564/",
    ),
    Dataset("labeled-s", "IRB", "Labeled-S images", "https://nyu.databrary.org/volume/564/"),
    Dataset(
        "babyview", "IRB", "Restricted BabyView recordings", "https://babyview-project.github.io/"
    ),
    Dataset(
        "vtwt",
        "IRB",
        "Public labels; restricted SAYCam-S images",
        "https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/vtwt",
    ),
    Dataset(
        "baby-winoground",
        "IRB",
        "Public labels; restricted original and synthetic images",
        "https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/baby_winoground",
    ),
    Dataset(
        "intphys2",
        "upstream-licensed",
        "IntPhys 2 Main split videos; download from the upstream release",
        "https://github.com/facebookresearch/IntPhys2",
    ),
    Dataset(
        "grasp",
        "upstream-licensed",
        "GRASP Level 2 videos; download the upstream release, then prepare the pair manifest",
        "https://github.com/i-machine-think/grasp",
        None,
        _BENCHMARKS + "grasp.prepare",
        (("--data_root", "grasp/level2"), ("--output", "grasp/manifest.json")),
    ),
    Dataset(
        "imagenet",
        "upstream-licensed",
        "Obtain images from the licensed upstream release",
        "https://www.image-net.org/",
    ),
    Dataset(
        "ego4d-nlq",
        "upstream-licensed",
        "Ego4D v2 NLQ annotations and clips; download with the Ego4D CLI",
        "https://ego4d-data.org/docs/start-here/",
    ),
    Dataset(
        "egoschema",
        "upstream-licensed",
        "EgoSchema questions, public subset answers and 3-minute videos",
        "https://github.com/egoschema/EgoSchema",
    ),
)


def dataset(name: str) -> Dataset:
    for item in DATASETS:
        if item.name == name:
            return item
    raise ValueError(f"unknown dataset {name!r}; use 'egl data list'")


def preparation_command(item: Dataset, root: Path, arguments: list[str]) -> list[str]:
    if item.prepare_module is None:
        raise ValueError(f"{item.name} has no preparer; see {item.source}")
    read_only = (
        "--help" in arguments
        or "-h" in arguments
        or (item.name == "spatial-relations" and "--validate-only" in arguments)
    )
    for flag, path in item.prepare_defaults:
        if flag not in {"--output-dir", "--out_dir", "--output"}:
            continue
        explicit = any(arg == flag or arg.startswith(flag + "=") for arg in arguments)
        if not read_only and not explicit and (root / path).exists():
            raise FileExistsError(
                f"preparation destination exists: {root / path}; "
                f"use {flag} to explicitly choose the preparation destination"
            )
    defaults = [value for flag, path in item.prepare_defaults for value in (flag, str(root / path))]
    # argparse's last occurrence wins; explicit legacy options override our defaults.
    return [sys.executable, "-m", item.prepare_module, *defaults, *arguments]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("EGOLEARN_DATA") or "data")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="show sources, access tiers and available operations"
    )
    listing.add_argument("--json", action="store_true")
    fetching = commands.add_parser("fetch", help="fetch and verify a registered Hub collection")
    fetching.add_argument("name")
    fetching.add_argument("--revision", default="main")
    fetching.add_argument("--repo", help="explicit published dataset repo, org/name")
    fetching.add_argument("--allow-unverified", action="store_true")
    fetching.add_argument("--replace", action="store_true", help="replace an existing collection")
    preparing = commands.add_parser("prepare", help="run an existing benchmark preparer")
    preparing.add_argument("name")
    preparing.add_argument("arguments", nargs=argparse.REMAINDER)
    verifying = commands.add_parser("verify", help="verify checksums.sha256 without changing files")
    verifying.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            if args.json:
                print(json.dumps([asdict(item) for item in DATASETS], indent=2))
            else:
                print("NAME                 ACCESS             FETCH  PREPARE  DESCRIPTION")
                for item in DATASETS:
                    print(
                        f"{item.name:20} {item.access:18} "
                        f"{'yes' if item.collection else '-':6} "
                        f"{'yes' if item.prepare_module else '-':8} {item.description}"
                    )
                print("Hub entries are publication candidates; availability depends on the owner.")
                print("Preparation requires egolearn[train] and the benchmark source assets.")
        elif args.command == "verify":
            problems = hub.verify_checksums(args.directory.expanduser())
            print("\n".join(problems) if problems else "ok: all checksums match")
            return int(bool(problems))
        else:
            item = dataset(args.name)
            root = args.root.expanduser()
            if args.command == "fetch":
                if item.collection is None:
                    raise ValueError(
                        f"{item.name} ({item.access}) has no Hub collection; see {item.source}"
                    )
                if (root / item.collection).exists() and not args.replace:
                    raise FileExistsError("destination exists; pass --replace to replace it")
                print(
                    hub.fetch(
                        item.collection,
                        root,
                        revision=args.revision,
                        repo=args.repo,
                        allow_unverified=args.allow_unverified,
                        replace=args.replace,
                    )
                )
            else:
                arguments = args.arguments
                if arguments[:1] == ["--"]:
                    arguments = arguments[1:]
                command = preparation_command(item, root, arguments)
                if "--help" in arguments or "-h" in arguments:
                    from egolearn.commands.prepare_parsers import preparation_parser

                    preparation_parser(
                        item.name,
                        description=item.description,
                        prog=f"egl data prepare {item.name}",
                    ).parse_args(command[3:])
                    return 0
                if importlib.util.find_spec("torch") is None:
                    raise ValueError("benchmark preparation requires egolearn[train]")
                os.execv(sys.executable, command)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
