"""List the Konkle test-60 images (60 categories, 17 photographs each) in a manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def prepare_manifest(data_root: Path, output: Path) -> int:
    """Validate the 60 x 17 image set and write a new manifest.

    Existing output files are never overwritten; validation completes before
    writing. The manifest records absolute image paths and each file's SHA-256.
    """
    root = data_root.expanduser().resolve()
    categories = {}
    for category in TEST_CATEGORIES:
        images = category_images(root / category)
        if len(images) != IMAGES_PER_CATEGORY:
            raise ValueError(
                f"Expected {IMAGES_PER_CATEGORY} images for {category!r} under {root}, "
                f"found {len(images)}"
            )
        categories[category] = [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in images
        ]

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump({"split": "test60", "categories": categories}, handle, indent=2)
        handle.write("\n")
    return sum(len(images) for images in categories.values())


def main() -> None:
    parser = preparation_parser("konkle-4afc", description=__doc__)
    args = parser.parse_args()
    try:
        count = prepare_manifest(args.data_root, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {count} images in {len(TEST_CATEGORIES)} categories to {args.output}")


if __name__ == "__main__":
    main()
