"""Build the BabyView benchmark manifests from the public crops release.

Reads the release's manifest.csv (category, stem, jpeg_path) and writes:

  babyview_4afc_trials.json   4AFC trials in probe:four_afc's format
  babyview_linear.json        frozen stratified split for probe:linear
  babyview_report.json        provenance: counts, OOV list, seeds

The trial construction is a faithful port of the source generator
(mcl eval/babyview_4afc/generate_trials.py): identical RNG call order and
seed, identical category ordering (sorted, vocabulary-filtered), so the
emitted trials match the source repo's frozen 4150-trial json image for
image.  The linear-probe split uses ALL categories (no vocabulary filter)
and its own seed.

Usage (data already on disk, or --download to fetch the zip first):

    python -m egolearn.evaluation.benchmarks.babyview_4afc.prepare \\
        --data_root <dir with manifest.csv + crops/> \\
        --vocab_json <saycam vocab.json> --out_dir <dir> [--download]
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from egolearn.commands.prepare_parsers import preparation_parser

CROPS_URL = (
    "https://raw.githubusercontent.com/babyview-project/ccn-2026-object-variability/"
    "main/data/shared_data_ccn_2026/valid5921_public/crops/valid7018_public_crops.zip"
)


def download(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "valid7018_public_crops.zip"
    if not zip_path.is_file():
        print(f"[download] {CROPS_URL}")
        urllib.request.urlretrieve(CROPS_URL, zip_path)
    if not (data_root / "manifest.csv").is_file():
        zipfile.ZipFile(zip_path).extractall(data_root)
        print(f"[unzip] {data_root}")


def load_manifest(data_root: Path) -> dict[str, list[Path]]:
    """category -> sorted absolute crop paths (the source generator's
    ordering; the RNG-faithful port depends on it)."""
    man = data_root / "manifest.csv"
    if not man.is_file():
        raise SystemExit(f"missing {man}; run with --download or unzip the release first")
    by_cat: dict[str, list[Path]] = defaultdict(list)
    with man.open() as f:
        for row in csv.DictReader(f):
            path = (data_root / row["jpeg_path"]).resolve()
            if not path.is_file():
                raise SystemExit(f"missing crop {path}")
            by_cat[row["category"].strip()].append(path)
    return {k: sorted(v) for k, v in by_cat.items()}


def generate_trials(
    by_cat: dict[str, list[Path]], categories: list[str], n_trials: int, n_foils: int, seed: int
) -> list[dict]:
    """Faithful port of the source generator: same draws in the same order.
    Output is probe:four_afc's manifest entry format (target is images[0])."""
    rng = random.Random(seed)
    trials = []
    for _ in range(n_trials):
        target_category = rng.choice(categories)
        target_img = rng.choice(by_cat[target_category])
        foil_categories = rng.sample([c for c in categories if c != target_category], n_foils)
        foil_imgs = [str(rng.choice(by_cat[fc])) for fc in foil_categories]
        trials.append(
            {"text": target_category, "images": [str(target_img), *foil_imgs], "answer": 0}
        )
    return trials


def generate_split(by_cat: dict[str, list[Path]], test_frac: float, seed: int) -> dict:
    """Stratified train/test split over ALL categories, frozen by seed.
    Every class keeps at least one example on each side."""
    if not 0.0 < test_frac < 1.0:
        raise SystemExit(f"--test_frac must be in (0, 1), got {test_frac}")
    rng = random.Random(seed)
    classes = sorted(by_cat)
    train: list[dict] = []
    test: list[dict] = []
    for label, cat in enumerate(classes):
        files = [str(p) for p in by_cat[cat]]
        if len(files) < 2:
            raise SystemExit(f"category {cat!r} has {len(files)} crop(s); need >= 2 to split")
        rng.shuffle(files)
        n_test = min(max(1, round(test_frac * len(files))), len(files) - 1)
        test.extend({"image": p, "label": label} for p in files[:n_test])
        train.extend({"image": p, "label": label} for p in files[n_test:])
    return {"classes": classes, "train": train, "test": test}


def main() -> None:
    p = preparation_parser("babyview-4afc", description=__doc__)
    args = p.parse_args()

    if args.download:
        download(args.data_root)
    vocab = json.loads(args.vocab_json.read_text())
    by_cat = load_manifest(args.data_root)
    all_cats = sorted(by_cat)
    in_vocab = [c for c in all_cats if c in vocab]
    oov = [c for c in all_cats if c not in vocab]
    if len(in_vocab) < args.n_foils + 1:
        raise SystemExit(f"need >= {args.n_foils + 1} in-vocab categories, got {len(in_vocab)}")

    trials = generate_trials(by_cat, in_vocab, args.n_trials, args.n_foils, args.trials_seed)
    split = generate_split(by_cat, args.test_frac, args.split_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "babyview_4afc_trials.json").write_text(
        json.dumps({"trials": trials}, indent=1) + "\n"
    )
    (args.out_dir / "babyview_linear.json").write_text(json.dumps(split, indent=1) + "\n")
    report = {
        "source": "babyview-project/ccn-2026-object-variability valid5921_public",
        "four_afc": {
            "protocol": "Labeled-S eval_object_categories_4150: random target + 3 foils",
            "n_trials": args.n_trials,
            "n_foils": args.n_foils,
            "seed": args.trials_seed,
            "n_in_vocab_categories": len(in_vocab),
            "oov_categories_excluded": oov,
        },
        "linear": {
            "protocol": "stratified split over ALL categories, no vocabulary filter",
            "n_classes": len(all_cats),
            "n_train": len(split["train"]),
            "n_test": len(split["test"]),
            "test_frac": args.test_frac,
            "seed": args.split_seed,
        },
        "transform": "center_crop 224 (frozen 2026-08-30; see the package docstring)",
        "n_public_crops": sum(len(v) for v in by_cat.values()),
    }
    (args.out_dir / "babyview_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out_dir}: {len(trials)} trials, "
        f"linear split {len(split['train'])}/{len(split['test'])} over {len(all_cats)} classes, "
        f"oov excluded: {oov}"
    )


if __name__ == "__main__":
    main()
