"""Fixed monitoring subsets selected from the frozen full spatial datasets."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from egolearn.commands.prepare_parsers import preparation_parser
from egolearn.evaluation.benchmarks.shape_bias.stimuli import sha256_file
from egolearn.evaluation.benchmarks.spatial_relations.prepare import (
    _finish,
    _rng,
    validate_dataset,
)


def select_trials(
    trials: list[dict[str, Any]],
    experiment: str,
    *,
    seed: int = 33,
    exp1_per_condition: int = 64,
    exp3_configurations: int = 16,
) -> list[dict[str, Any]]:
    """Preserve full-set trial rows, including labels and candidate presentation."""
    if experiment == "exp1":
        if exp1_per_condition < 2 or exp1_per_condition % 2:
            raise ValueError("Exp. 1 trials per condition must be positive and even")
        cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in trials:
            key = (row["renderer"], row["relation"], row["target_condition"], row["correct_index"])
            cells[key].append(row)
        selected = []
        for key, rows in sorted(cells.items()):
            ordered = sorted(rows, key=lambda row: row["id"])
            count = exp1_per_condition // 2
            if len(ordered) < count:
                raise ValueError(f"insufficient Exp. 1 trials in {key}")
            selected.extend(_rng(seed, f"monitoring/exp1/{key}").sample(ordered, count))
    elif experiment == "exp3":
        if exp3_configurations < 1:
            raise ValueError("Exp. 3 configuration count must be positive")
        cells_by_config: dict[tuple, set[int]] = defaultdict(set)
        for row in trials:
            key = (
                row["relation"],
                row["target_condition"],
                row["reference_object"],
                row["anchor_object"],
            )
            cells_by_config[key].add(row["configuration"])
        if not cells_by_config or len({tuple(sorted(c)) for c in cells_by_config.values()}) != 1:
            raise ValueError(
                "Exp. 3 configurations must be shared across all condition/object cells"
            )
        configurations = sorted(next(iter(cells_by_config.values())))
        if len(configurations) < exp3_configurations:
            raise ValueError("insufficient shared Exp. 3 configurations")
        chosen = set(
            _rng(seed, "monitoring/exp3/configurations").sample(configurations, exp3_configurations)
        )
        selected = [row for row in trials if row["configuration"] in chosen]
    else:
        raise ValueError(f"unknown experiment: {experiment}")
    return sorted(selected, key=lambda row: row["id"])


def prepare_subset(
    full_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 33,
    exp1_per_condition: int = 64,
    exp3_configurations: int = 16,
) -> dict[str, dict]:
    """Validate parents, copy selected images, and publish both subsets atomically."""
    full_root = Path(full_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    parents = {}
    selections = {}
    for experiment in ("exp1", "exp3"):
        parent = full_root / experiment
        parents[experiment] = validate_dataset(parent)
        trials = [json.loads(line) for line in (parent / "trials.jsonl").read_text().splitlines()]
        selections[experiment] = select_trials(
            trials,
            experiment,
            seed=seed,
            exp1_per_condition=exp1_per_condition,
            exp3_configurations=exp3_configurations,
        )
        print(f"Selected {experiment}: {len(selections[experiment])} trials", flush=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    manifests = {}
    with tempfile.TemporaryDirectory(prefix=".monitoring-", dir=output_root.parent) as temporary:
        stage = Path(temporary) / "dataset"
        for experiment, trials in selections.items():
            parent = full_root / experiment
            output = stage / experiment
            output.mkdir(parents=True)
            images = {r["path"]: r["sha256"] for r in parents[experiment]["images"]}
            paths = sorted({p for row in trials for p in [row["anchor"], *row["candidates"]]})
            for relative in paths:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent / relative, target)
                if sha256_file(target) != images[relative]:
                    raise ValueError(f"copied image differs from parent: {relative}")
            source = {
                "kind": "monitoring_subset",
                "selection_protocol": "monitoring_v1",
                "seed": seed,
                "parent_experiment": experiment,
                "parent_manifest_sha256": sha256_file(parent / "manifest.json"),
                "parent_trials_sha256": parents[experiment]["trials"]["sha256"],
                "parent_n_trials": parents[experiment]["n_trials"],
                "selection": (
                    "equal samples per renderer/relation/target condition and correct index"
                    if experiment == "exp1"
                    else "shared configuration sample; retain all objects and conditions"
                ),
                "exp1_per_condition": exp1_per_condition,
                "exp3_configurations": exp3_configurations,
                "selected_configurations": sorted({row["configuration"] for row in trials}),
                "usage": "training-time monitoring; overlaps full final-evaluation dataset",
            }
            manifests[experiment] = _finish(output, experiment, trials, source)
        stage.rename(output_root)
    return manifests


def main() -> None:
    parser = preparation_parser("spatial-monitoring", description=__doc__)
    args = parser.parse_args()
    manifests = prepare_subset(
        args.full_root,
        args.output_dir,
        seed=args.seed,
        exp1_per_condition=args.exp1_per_condition,
        exp3_configurations=args.exp3_configurations,
    )
    for experiment, manifest in manifests.items():
        print(
            f"Prepared {experiment}: {manifest['n_trials']} trials, "
            f"{manifest['n_images']} images at {args.output_dir / experiment}"
        )


if __name__ == "__main__":
    main()
