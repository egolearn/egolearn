"""Compare full and monitoring scores using identical frozen-model embeddings.

External DINO implementations are imported only in worker processes launched
from their source directories. Checkpoints and source configuration are explicit
inputs; neither model code nor weights are redistributed.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from egolearn.evaluation.benchmarks.shape_bias.stimuli import sha256_file, write_json
from egolearn.evaluation.benchmarks.spatial_relations.prepare import validate_dataset
from egolearn.evaluation.benchmarks.spatial_relations.preprocessing import spatial_transform
from egolearn.evaluation.benchmarks.spatial_relations.score import score_trials


class Images(Dataset):
    def __init__(self, root, records):
        self.root = root
        self.records = records
        self.transform = spatial_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.root / self.records[index]["path"]) as image:
            return self.transform(image.convert("RGB"))


def _resolve(root, path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def _load_model(root, spec):
    if spec["loader_type"] == "dinov2":
        from dinov2.models import build_model_from_cfg
        from omegaconf import OmegaConf

        model, _ = build_model_from_cfg(
            OmegaConf.load(_resolve(root, spec["config_file"])), only_teacher=True
        )
    elif spec["loader_type"] == "emin_vit":
        import vision_transformer

        model = getattr(vision_transformer, spec["arch"])(
            patch_size=int(spec["patch_size"]), num_classes=0
        )
    else:
        raise ValueError(f"unsupported loader {spec['loader_type']}")
    path = _resolve(root, spec["checkpoint_path"])
    # Legacy DINO checkpoints include argument and NumPy scalar metadata.
    with torch.serialization.safe_globals(
        [
            argparse.Namespace,
            np.dtype,
            (
                importlib.import_module("numpy._core.multiarray").scalar,
                "numpy.core.multiarray.scalar",
            ),
            np.dtypes.Float64DType,
        ]
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get(spec.get("checkpoint_key", "teacher"), checkpoint)
    state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
    expected = model.state_dict()
    # Training projection heads are not part of the image backbone.
    model.load_state_dict({k: v for k, v in state.items() if k in expected}, strict=True)
    return model.eval().cuda(), sha256_file(path)


def evaluate_model(args, spec):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint_hash = _load_model(args.source_root, spec)
    output = args.output_dir / args.worker_model
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model_id": args.worker_model,
        "display_name": spec["display_name"],
        "checkpoint_sha256": checkpoint_hash,
        "torch": torch.__version__,
        "precision": "float32; TF32 disabled; float64 cosine scoring",
        "transform": "RGB, shortest-side bilinear resize 224, ImageNet normalization",
        "feature": "final normalized CLS token",
        "tie_tolerance": args.tie_tolerance,
        "device": torch.cuda.get_device_name(),
        "datasets": {},
    }
    for experiment in ("exp1", "exp3"):
        root = args.full_root / experiment
        manifest = json.loads((root / "manifest.json").read_text())
        trials = list(map(json.loads, (root / "trials.jsonl").read_text().splitlines()))
        monitoring = list(
            map(
                json.loads,
                (args.subset_root / experiment / "trials.jsonl").read_text().splitlines(),
            )
        )
        parent = {row["id"]: row for row in trials}
        if any(row != parent[row["id"]] for row in monitoring):
            raise ValueError("monitoring trial differs from full trial")
        ids = {row["id"] for row in monitoring}
        selected = np.array([row["id"] in ids for row in trials])
        loader = DataLoader(
            Images(root, manifest["images"]),
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
        )
        batches = []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch_index, inputs in enumerate(loader):
                features = model(inputs.cuda(non_blocking=True))
                batches.append(features.float().cpu().numpy())
                if batch_index % 50 == 0:
                    print(
                        f"{args.worker_model}/{experiment}: batch {batch_index}/{len(loader)}",
                        flush=True,
                    )
        features = np.concatenate(batches)
        scores, margins = score_trials(features, manifest["images"], trials, args.tie_tolerance)
        np.savez_compressed(
            output / f"{experiment}.npz",
            trial_ids=np.array([r["id"] for r in trials]),
            scores=scores,
            margins=margins,
            selected=selected,
        )
        metadata["datasets"][experiment] = {
            "full_manifest_sha256": sha256_file(root / "manifest.json"),
            "subset_manifest_sha256": sha256_file(args.subset_root / experiment / "manifest.json"),
            "seconds": time.perf_counter() - started,
            "full_accuracy": float(scores.mean()),
            "subset_accuracy": float(scores[selected].mean()),
        }
        print(args.worker_model, experiment, metadata["datasets"][experiment], flush=True)
    write_json(output / "metadata.json", metadata)


def summarize(args, models):
    rows = []
    for experiment in ("exp1", "exp3"):
        trials = list(
            map(json.loads, (args.full_root / experiment / "trials.jsonl").read_text().splitlines())
        )
        cells = np.array(
            [f"{r['renderer']}/{r['relation']}/{r['target_condition']}" for r in trials]
        )
        for model_id in models:
            data = np.load(args.output_dir / model_id / f"{experiment}.npz")
            if data["trial_ids"].tolist() != [r["id"] for r in trials]:
                raise ValueError("score IDs do not match manifest")
            scores, margins, selected = data["scores"], data["margins"], data["selected"]
            for cell in ["overall", *sorted(set(cells))]:
                full = np.ones(len(trials), dtype=bool) if cell == "overall" else cells == cell
                small = full & selected
                rows.append(
                    {
                        "model": model_id,
                        "experiment": experiment,
                        "condition": cell,
                        "full_n": int(full.sum()),
                        "subset_n": int(small.sum()),
                        "full_accuracy": float(scores[full].mean()),
                        "subset_accuracy": float(scores[small].mean()),
                        "difference_pp": float(100 * (scores[small].mean() - scores[full].mean())),
                        "full_margin": float(margins[full].mean()),
                        "subset_margin": float(margins[small].mean()),
                        "full_tie_rate": float(
                            (np.abs(margins[full]) <= args.tie_tolerance).mean()
                        ),
                        "subset_tie_rate": float(
                            (np.abs(margins[small]) <= args.tie_tolerance).mean()
                        ),
                    }
                )
    with (args.output_dir / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "comparison.json", rows)


def sampling_variability(args, models, repetitions=2000):
    """Finite-population subset variation under the monitoring sampling design.

    These are ranges of sampling error relative to the fixed full dataset,
    not confidence intervals for performance on an unknown image population.
    """
    records = []
    rankings = []
    model_ids = list(models)
    rng = np.random.default_rng(2026)
    for experiment in ("exp1", "exp3"):
        trials = list(
            map(json.loads, (args.full_root / experiment / "trials.jsonl").read_text().splitlines())
        )
        archives = [np.load(args.output_dir / model / f"{experiment}.npz") for model in model_ids]
        scores = np.stack([a["scores"] for a in archives])
        selected = archives[0]["selected"]
        if not all(np.array_equal(a["selected"], selected) for a in archives):
            raise ValueError("models were evaluated on different subsets")
        full = scores.mean(axis=1)
        actual = scores[:, selected].mean(axis=1)
        if experiment == "exp1":
            strata: dict[tuple[str, str, str, int], list[int]] = {}
            for i, row in enumerate(trials):
                key = (
                    row["renderer"],
                    row["relation"],
                    row["target_condition"],
                    row["correct_index"],
                )
                strata.setdefault(key, []).append(i)
            draws = []
            for indices in strata.values():
                count = int(selected[indices].sum())
                draws.append(
                    np.array(
                        [rng.choice(indices, count, replace=False) for _ in range(repetitions)]
                    )
                )
            samples = scores[:, np.concatenate(draws, axis=1)].mean(axis=2)
        else:
            configurations = sorted({row["configuration"] for row in trials})
            group_scores = np.stack(
                [
                    scores[
                        :, [i for i, row in enumerate(trials) if row["configuration"] == c]
                    ].mean(axis=1)
                    for c in configurations
                ],
                axis=1,
            )
            count = len({row["configuration"] for i, row in enumerate(trials) if selected[i]})
            configuration_draws = np.array(
                [rng.choice(len(configurations), count, replace=False) for _ in range(repetitions)]
            )
            samples = group_scores[:, configuration_draws].mean(axis=2)
        errors = 100 * (samples - full[:, None])
        for i, model_id in enumerate(model_ids):
            lo, hi = np.quantile(errors[i], [0.025, 0.975])
            actual_error = 100 * (actual[i] - full[i])
            records.append(
                {
                    "model": model_id,
                    "experiment": experiment,
                    "actual_difference_pp": float(actual_error),
                    "random_subset_error_95_range_pp": [float(lo), float(hi)],
                    "absolute_error_percentile": float(
                        100 * (np.abs(errors[i]) <= abs(actual_error)).mean()
                    ),
                }
            )
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                full_sign = np.sign(full[i] - full[j])
                rankings.append(
                    {
                        "experiment": experiment,
                        "model_a": model_ids[i],
                        "model_b": model_ids[j],
                        "full_difference_pp": float(100 * (full[i] - full[j])),
                        "subset_difference_pp": float(100 * (actual[i] - actual[j])),
                        "fixed_subset_preserves_order": bool(
                            np.sign(actual[i] - actual[j]) == full_sign
                        ),
                        "random_subset_order_agreement": float(
                            (np.sign(samples[i] - samples[j]) == full_sign).mean()
                        ),
                    }
                )
    write_json(
        args.output_dir / "sampling_variability.json",
        {
            "seed": 2026,
            "repetitions": repetitions,
            "interpretation": "subset sampling error relative to the fixed full dataset",
            "models": records,
            "model_pairs": rankings,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--worker-model")
    args = parser.parse_args()
    for key in ("source_root", "comparison_config", "full_root", "subset_root", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    import yaml

    config = yaml.safe_load(args.comparison_config.read_text())
    models = {k: s for k, s in config["models"].items() if s.get("enabled", True)}
    if args.worker_model:
        evaluate_model(args, models[args.worker_model])
        return
    for experiment in ("exp1", "exp3"):
        validate_dataset(args.full_root / experiment)
        validate_dataset(args.subset_root / experiment)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for model_id, spec in models.items():
        cwd = (
            args.source_root
            if spec["loader_type"] == "dinov2"
            else _resolve(args.source_root, spec["emin_root"])
        )
        command = [
            sys.executable,
            "-c",
            "from egolearn.evaluation.benchmarks.spatial_relations.compare import main; main()",
        ]
        for key in (
            "source_root",
            "comparison_config",
            "full_root",
            "subset_root",
            "output_dir",
            "batch_size",
            "workers",
            "tie_tolerance",
        ):
            command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        command.extend(["--worker-model", model_id])
        subprocess.run(command, cwd=cwd, check=True)
    summarize(args, models)
    sampling_variability(args, models)


if __name__ == "__main__":
    main()
