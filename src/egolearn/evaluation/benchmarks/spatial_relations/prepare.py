"""Prepare frozen, portable evaluation trials without redistributing source assets.

Experiment 1 calls an explicitly supplied external renderer; Experiment 3 copies
existing renders. Neither model outputs nor their ordering determine trial selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from egolearn.evaluation.benchmarks.shape_bias.stimuli import sha256_file, write_json

PAPER = "https://doi.org/10.1016/j.cognition.2023.105690"
RENDERERS = ("different_shapes", "split_text", "random_color")
RELATIONS = ("above_below", "between_outside")
TARGET_CONDITIONS = ("identical", "different")
ROLES = ("anchor", "same", "other")


def _rng(seed: int, key: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big"))


def _trial(
    trial_id: str,
    group_id: str,
    relation: str,
    target_condition: str,
    renderer: str,
    paths: list[str],
    correct_index: int,
    **metadata: Any,
) -> dict[str, Any]:
    candidates = paths[1:] if correct_index == 0 else paths[:0:-1]
    return {
        "id": trial_id,
        "group_id": group_id,
        "split": "test",
        "relation": relation,
        "target_condition": target_condition,
        "renderer": renderer,
        "anchor": paths[0],
        "candidates": candidates,
        "correct_index": correct_index,
        **metadata,
    }


def audit_trials(trials: list[dict[str, Any]], images: list[dict[str, Any]]) -> dict:
    """Identify exact repeated draws irrespective of candidate presentation order."""
    hashes = {record["path"]: record["sha256"] for record in images}
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in trials:
        correct = row["correct_index"]
        key = (
            row["renderer"],
            row["relation"],
            row["target_condition"],
            hashes[row["anchor"]],
            hashes[row["candidates"][correct]],
            hashes[row["candidates"][1 - correct]],
        )
        groups[key].append(row["id"])
    repeats = sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)
    return {
        "n_unique_triplets_within_conditions": len(groups),
        "n_repeated_draws": len(trials) - len(groups),
        "repeated_trial_ids": repeats,
        "correct_index_counts": {
            str(i): sum(row["correct_index"] == i for row in trials) for i in (0, 1)
        },
        "repeat_policy": (
            "retain seeded draws; merge shared-image groups before supervised splitting"
        ),
    }


def _finish(root: Path, experiment: str, trials: list[dict[str, Any]], source: dict) -> dict:
    if not trials:
        raise ValueError("no trials")
    trials = sorted(trials, key=lambda row: row["id"])
    trials_path = root / "trials.jsonl"
    with trials_path.open("w") as handle:
        for row in trials:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    paths = sorted({p for row in trials for p in [row["anchor"], *row["candidates"]]})
    images = []
    for path in paths:
        with Image.open(root / path) as image:
            image.load()
            images.append(
                {
                    "path": path,
                    "sha256": sha256_file(root / path),
                    "size": list(image.size),
                    "mode": image.mode,
                }
            )
    counts = Counter(f"{r['renderer']}/{r['relation']}/{r['target_condition']}" for r in trials)
    write_json(root / "audit.json", audit_trials(trials, images))
    manifest = {
        "schema_version": 1,
        "benchmark": "spatial_relations",
        "experiment": experiment,
        "paper": PAPER,
        "protocol": "egolearn_balanced_v1",
        "source": source,
        "task": "Choose the candidate preserving the anchor's spatial relation.",
        "split": "test",
        "n_trials": len(trials),
        "n_images": len(images),
        "condition_counts": dict(sorted(counts.items())),
        "trials": {"path": "trials.jsonl", "sha256": sha256_file(trials_path)},
        "audit": {"path": "audit.json", "sha256": sha256_file(root / "audit.json")},
        "images": images,
    }
    write_json(root / "manifest.json", manifest)
    validate_dataset(root)
    return manifest


def validate_dataset(root: str | Path) -> dict:
    """Fail on broken labels, condition imbalance, path escapes, or altered files."""
    root = Path(root).resolve()

    def local(path: str) -> Path:
        target = (root / path).resolve()
        if Path(path).is_absolute() or not target.is_relative_to(root):
            raise ValueError(f"nonlocal dataset path: {path}")
        return target

    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["benchmark"] != "spatial_relations"
        or manifest["experiment"] not in ("exp1", "exp3")
    ):
        raise ValueError("unsupported dataset schema")
    trial_file = local(manifest["trials"]["path"])
    if sha256_file(trial_file) != manifest["trials"]["sha256"]:
        raise ValueError("trial checksum mismatch")
    trials = [json.loads(line) for line in trial_file.read_text().splitlines()]
    if len(trials) != manifest["n_trials"] or len({r["id"] for r in trials}) != len(trials):
        raise ValueError("incorrect trial count or duplicate trial IDs")
    records = {r["path"]: r for r in manifest["images"]}
    if len(records) != manifest["n_images"] or len(records) != len(manifest["images"]):
        raise ValueError("incorrect image inventory")
    used = set()
    conditions: Counter = Counter()
    answers: dict[str, Counter] = defaultdict(Counter)
    for row in trials:
        if (
            row["correct_index"] not in (0, 1)
            or len(row["candidates"]) != 2
            or len(set(row["candidates"])) != 2
            or row["anchor"] in row["candidates"]
        ):
            raise ValueError("invalid two-choice trial")
        if (
            row["relation"] not in RELATIONS
            or row["target_condition"] not in TARGET_CONDITIONS
            or row["split"] != "test"
            or not row["group_id"]
        ):
            raise ValueError("invalid trial condition")
        key = f"{row['renderer']}/{row['relation']}/{row['target_condition']}"
        conditions[key] += 1
        answers[key][row["correct_index"]] += 1
        used.update([row["anchor"], *row["candidates"]])
    expected_renderers = RENDERERS if manifest["experiment"] == "exp1" else ("blender",)
    expected = {
        f"{s}/{r}/{t}" for s in expected_renderers for r in RELATIONS for t in TARGET_CONDITIONS
    }
    if (
        set(conditions) != expected
        or len(set(conditions.values())) != 1
        or dict(conditions) != manifest["condition_counts"]
    ):
        raise ValueError("incomplete or unbalanced conditions")
    if any(abs(c[0] - c[1]) > 1 for c in answers.values()):
        raise ValueError("unbalanced candidate order")
    if used != set(records):
        raise ValueError("image inventory does not match trial paths")
    audit_path = local(manifest["audit"]["path"])
    if sha256_file(audit_path) != manifest["audit"]["sha256"] or json.loads(
        audit_path.read_text()
    ) != audit_trials(trials, manifest["images"]):
        raise ValueError("trial audit mismatch")
    for path, record in records.items():
        target = local(path)
        if sha256_file(target) != record["sha256"]:
            raise ValueError(f"image checksum mismatch: {path}")
        with Image.open(target) as image:
            image.load()
            if list(image.size) != record["size"] or image.mode != record["mode"]:
                raise ValueError(f"image metadata mismatch: {path}")
    return manifest


def prepare_exp3(
    image_root: str | Path,
    output: Path,
    *,
    seed: int = 33,
    expected_objects: int = 8,
    expected_configurations: int = 256,
) -> dict:
    """Balance target changes within every existing reference/configuration block.

    One seeded cyclic derangement pairs each anchor object with one different
    test object. All objects appear equally often in both roles. All available
    configuration blocks are retained; legacy model-result CSVs are not inputs.
    """
    image_root = Path(image_root)
    blocks: dict[tuple[str, str, int], dict[str, dict[str, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for relation, folder in (
        ("above_below", "quinn_above_below"),
        ("between_outside", "quinn_between"),
    ):
        for path in sorted((image_root / folder).glob("*.png")):
            reference, object_name, index_text, role = path.stem.rsplit("_", 3)
            if role not in ("habituation", "same", "different"):
                continue
            block = blocks[relation, reference, int(index_text)][object_name]
            if role in block:
                raise ValueError(f"duplicate scene role: {path.name}")
            block[role] = path
    if not blocks:
        raise ValueError("no Experiment 3 renders found")
    object_sets = {tuple(sorted(objects)) for objects in blocks.values()}
    if len(object_sets) != 1 or len(next(iter(object_sets))) != expected_objects:
        raise ValueError("incomplete or inconsistent object coverage across scene blocks")
    if expected_objects < 2 or expected_objects % 2:
        raise ValueError("expected_objects must be an even integer >= 2")
    config_sets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for relation, reference, index in blocks:
        config_sets[relation, reference].add(index)
    if (
        {r for r, _ in config_sets} != set(RELATIONS)
        or len(config_sets) != 2
        or any(v != set(range(expected_configurations)) for v in config_sets.values())
    ):
        raise ValueError("incomplete configuration coverage across relations/references")
    trials = []
    source_images = []
    for (relation, reference, index), objects in sorted(blocks.items()):
        names = sorted(objects)
        group = f"exp3/{relation}/{reference}/{index:04d}"
        shuffled = names.copy()
        rng = _rng(seed, group)
        rng.shuffle(shuffled)
        offset = rng.randrange(1, len(shuffled))
        changed = {name: shuffled[(i + offset) % len(shuffled)] for i, name in enumerate(shuffled)}
        copied = {}
        for name in names:
            if set(objects[name]) != {"habituation", "same", "different"}:
                raise ValueError(f"incomplete image triplet: {group}/{name}")
            for role, path in sorted(objects[name].items()):
                relative = f"images/{relation}/{path.name}"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied[name, role] = relative
                source_images.append(
                    {"path": path.relative_to(image_root).as_posix(), "sha256": sha256_file(path)}
                )
        for condition in TARGET_CONDITIONS:
            positions = [i % 2 for i in range(len(names))]
            _rng(seed, f"{group}/{condition}/order").shuffle(positions)
            for name, correct_index in zip(names, positions, strict=True):
                probe = name if condition == "identical" else changed[name]
                paths = [
                    copied[name, "habituation"],
                    copied[probe, "same"],
                    copied[probe, "different"],
                ]
                trials.append(
                    _trial(
                        f"{group}/{condition}/{name}",
                        group,
                        relation,
                        condition,
                        "blender",
                        paths,
                        correct_index,
                        reference_object=reference,
                        anchor_object=name,
                        candidate_object=probe,
                        configuration=index,
                    )
                )
    return _finish(
        output,
        "exp3",
        trials,
        {
            "kind": "existing_renders",
            "seed": seed,
            "selection": "all scene blocks; identical plus one cyclic derangement per block",
            "limitations": [
                "reference-object type is confounded with relation",
                "balanced trial selection is an adaptation, not the original trial list",
                "image hashes identify local assets; original render provenance unverified",
            ],
            "images": source_images,
        },
    )


def prepare_exp1(output: Path, *, seed: int = 33, n_per_condition: int = 1024) -> dict:
    """Stream uncached individual trials from an external historical generator."""
    import matplotlib
    import numpy as np
    import torchvision.transforms as transforms
    from torchvision.transforms import _functional_tensor

    matplotlib.use("Agg")
    # The legacy renderer imports the old public module, removed in torchvision.
    # Expose its maintained implementation without modifying the external checkout.
    transforms.functional_tensor = _functional_tensor
    from simple_relational_reasoning.embeddings import stimuli, triplets

    if n_per_condition < 1:
        raise ValueError("n_per_condition must be positive")
    if stimuli.cv2 is None or stimuli.cc is None:
        raise ValueError("install opencv-python-headless and colorcet; no rendering fallbacks")
    source_files = {Path(m.__file__).name: sha256_file(m.__file__) for m in (stimuli, triplets)}
    trials = []
    for renderer in RENDERERS:
        for relation in RELATIONS:
            for condition in TARGET_CONDITIONS:
                cell = f"{renderer}/{relation}/{condition}"
                stimulus = stimuli.STIMULUS_GENERATORS[renderer](
                    rng=np.random.default_rng(seed), rotate_angle=0
                )
                generator = triplets.QuinnTripletGenerator(
                    stimulus,
                    (30, 80) if relation == "above_below" else (50, 80),
                    relation="above_below" if relation == "above_below" else "between",
                    two_reference_objects=relation == "between_outside",
                    adjacent_reference_objects=False,
                    n_target_types=1 if condition == "identical" else 2,
                    n_habituation_stimuli=1,
                    same_relation_target_distance_ratio=1.0,
                    same_relation_at_reference_ends=False,
                    seed=seed,
                )
                positions = [i % 2 for i in range(n_per_condition)]
                _rng(seed, f"{cell}/order").shuffle(positions)
                for i, correct_index in enumerate(positions):
                    # __call__ resets RNG and caches equal-sized chunks. Bypass it.
                    tensor = generator.generate_single_triplet(normalize=False)
                    arrays = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
                    if arrays.shape != (3, 224, 224, 3) or not np.isfinite(arrays).all():
                        raise ValueError("invalid renderer output")
                    if arrays.min() < 0 or arrays.max() > 1:
                        raise ValueError("renderer output is not unnormalized RGB in [0,1]")
                    paths = []
                    for role, array in zip(ROLES, arrays, strict=True):
                        relative = f"images/{cell}/{i:05d}_{role}.png"
                        path = output / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path)
                        paths.append(relative)
                    trials.append(
                        _trial(
                            f"exp1/{cell}/{i:05d}",
                            f"exp1/{i:05d}",
                            relation,
                            condition,
                            renderer,
                            paths,
                            correct_index,
                            configuration=i,
                        )
                    )
                print(f"Prepared exp1/{cell}: {n_per_condition} trials", flush=True)
    versions = {
        p: importlib.metadata.version(p)
        for p in (
            "numpy",
            "torch",
            "torchvision",
            "matplotlib",
            "scipy",
            "colorcet",
            "opencv-python-headless",
            "pillow",
        )
    }
    return _finish(
        output,
        "exp1",
        trials,
        {
            "kind": "external_generator",
            "seed": seed,
            "n_per_condition": n_per_condition,
            "generator_sha256": source_files,
            "versions": versions,
            "rendering": {
                "rotation": 0,
                "canvas": [224, 224],
                "quantization": "round uint8",
                "distance_ratio": 1.0,
                "habituation_images": 1,
            },
            "limitations": ["regenerated stimuli, not a pixel-exact recovery of the paper"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    root = args.output_dir.resolve()
    experiments = ("exp1", "exp3") if args.experiment == "both" else (args.experiment,)
    if args.validate_only:
        for experiment in experiments:
            manifest = validate_dataset(root / experiment)
            print(f"Validated {experiment}: {manifest['n_trials']} trials")
        return
    if args.generator_root is not None and "exp1" in experiments:
        # Python's normal working-directory import discovers the external package.
        # Run in a child so library imports never depend on changing the caller's cwd.
        command = [
            sys.executable,
            "-c",
            "from egolearn.evaluation.benchmarks.spatial_relations.prepare import main; main()",
            "--output-dir",
            str(root),
            "--experiment",
            args.experiment,
            "--seed",
            str(args.seed),
            "--n-per-condition",
            str(args.n_per_condition),
        ]
        if args.exp3_image_root is not None:
            command.extend(["--exp3-image-root", str(args.exp3_image_root.resolve())])
        subprocess.run(command, cwd=args.generator_root.resolve(), check=True)
        return
    if "exp3" in experiments and args.exp3_image_root is None:
        parser.error("--exp3-image-root is required for Experiment 3")
    root.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        if (root / experiment).exists():
            raise FileExistsError(
                f"refusing to overwrite {root / experiment}; use a new output dir"
            )
    for experiment in experiments:
        with tempfile.TemporaryDirectory(prefix=f".{experiment}-", dir=root) as temporary:
            output = Path(temporary)
            if experiment == "exp1":
                import torch

                # Tiny CPU raster operations become slower with large thread pools.
                torch.set_num_threads(1)
                manifest = prepare_exp1(
                    output, seed=args.seed, n_per_condition=args.n_per_condition
                )
            else:
                manifest = prepare_exp3(args.exp3_image_root, output, seed=args.seed)
            output.rename(root / experiment)
        print(
            f"Prepared {experiment}: {manifest['n_trials']} trials, "
            f"{manifest['n_images']} images at {root / experiment}",
            flush=True,
        )


if __name__ == "__main__":
    main()
