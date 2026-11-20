"""Draw the Konkle 4AFC trials for one vocabulary and write a probe:four_afc manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from egolearn.commands.prepare_parsers import preparation_parser

# Test split of Vong et al. (2024), as used by arXiv:2610.31842.
TEST_CATEGORIES = [
    "airplane",
    "apple",
    "bagel",
    "ball",
    "balloon",
    "basket",
    "bed",
    "bell",
    "bike",
    "bill",
    "bird",
    "boot",
    "bottle",
    "bowl",
    "bucket",
    "butterfly",
    "button",
    "cake",
    "camera",
    "cat",
    "chair",
    "cheese",
    "clock",
    "cookie",
    "crib",
    "dog",
    "doll",
    "fan",
    "guitar",
    "hat",
    "jacket",
    "juice",
    "key",
    "knife",
    "leaves",
    "lock",
    "meat",
    "necklace",
    "pants",
    "pen",
    "phone",
    "pipe",
    "pizza",
    "ring",
    "rock",
    "rug",
    "sandwich",
    "shoe",
    "socks",
    "spoon",
    "stool",
    "tape",
    "tent",
    "train",
    "tree",
    "trumpet",
    "turtle",
    "tv",
    "umbrella",
    "watch",
]
IMAGES_PER_CATEGORY = 17
_SUFFIXES = {".jpg", ".jpeg"}


def category_images(directory: Path) -> list[Path]:
    """Return a category's photographs sorted by file name.

    The original distribution keeps 15 photographs in the category folder and
    two more in ``TestItems/``; a flattened copy keeps all 17 together. Both
    layouts are read. Hidden files and non-image files are ignored.
    """
    found = [
        path
        for folder in (directory, directory / "TestItems")
        if folder.is_dir()
        for path in folder.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in _SUFFIXES
    ]
    return sorted(found, key=lambda path: (path.name, str(path)))


def sample_trials(pools: dict[str, list[str]], n_trials: int = 100, seed: int = 0) -> list[dict]:
    """Draw ``n_trials`` trials per category in probe:four_afc's entry format.

    The order of random draws reproduces ``eval_4afc_region`` in
    mcfrank/vlm-headcam: categories are visited in sorted order, and each
    trial draws the target image, then three foil categories without
    replacement, then one image from each foil category. The target is
    ``images[0]``.
    """
    if len(pools) < 4:
        raise ValueError("4AFC needs at least four categories")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    rng = np.random.default_rng(seed)
    categories = sorted(pools)
    trials = []
    for category in categories:
        others = [c for c in categories if c != category]
        for _ in range(n_trials):
            images = [pools[category][int(rng.choice(len(pools[category])))]]
            foils = [others[int(i)] for i in rng.choice(len(others), 3, replace=False)]
            images += [pools[foil][int(rng.choice(len(pools[foil])))] for foil in foils]
            trials.append({"text": category, "images": images, "answer": 0})
    return trials


def prepare_manifest(
    data_root: Path, vocab_json: Path, output: Path, n_trials: int = 100, seed: int = 0
) -> dict:
    """Validate the 60 x 17 image set and write a new trial manifest.

    Categories whose label is not one token of the vocabulary get no trials and
    are left out of the foil pool. They are counted in ``chance_trials``, which
    probe:four_afc scores at 0.25, so the reported accuracy is the mean over
    all 60 categories. Existing output files are never overwritten; validation
    completes before writing. Returns the manifest's ``report``.
    """
    root = data_root.expanduser().resolve()
    vocab = json.loads(vocab_json.expanduser().read_text())
    pools = {}
    for category in TEST_CATEGORIES:
        images = category_images(root / category)
        if len(images) != IMAGES_PER_CATEGORY:
            raise ValueError(
                f"Expected {IMAGES_PER_CATEGORY} images for {category!r} under {root}, "
                f"found {len(images)}"
            )
        pools[category] = [str(path) for path in images]
    scored = sorted(c for c in pools if c in vocab)
    oov = sorted(set(pools) - set(scored))
    trials = sample_trials({c: pools[c] for c in scored}, n_trials, seed)
    report = {
        "protocol": "arXiv:2610.31842 Konkle 4AFC, Vong et al. (2024) test categories",
        "n_categories": len(pools),
        "n_trials_per_category": n_trials,
        "seed": seed,
        "oov_categories_at_chance": oov,
    }

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        manifest = {"trials": trials, "chance_trials": n_trials * len(oov), "report": report}
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = preparation_parser("konkle-4afc", description=__doc__)
    args = parser.parse_args()
    try:
        report = prepare_manifest(
            args.data_root, args.vocab_json, args.output, args.n_trials, args.seed
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    oov = report["oov_categories_at_chance"]
    scored = report["n_categories"] - len(oov)
    print(
        f"Wrote {scored * args.n_trials} trials over {scored} categories to {args.output}; "
        f"out of vocabulary, scored at chance: {oov}"
    )


if __name__ == "__main__":
    main()
