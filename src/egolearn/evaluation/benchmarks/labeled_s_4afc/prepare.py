"""Convert frozen Labeled-S 4AFC trials without changing trial or candidate order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_manifest(data_root: Path, trials_json: Path, vocab_json: Path, output: Path) -> int:
    """Validate 4,150 trials over 23 categories and write a new local manifest.

    Source image paths are relocated under data_root/<category>/<filename>.
    Every target must be one token in the model's training vocabulary. Existing
    output files are never overwritten; validation completes before writing.
    """
    root = data_root.expanduser().resolve()
    source = json.loads(trials_json.expanduser().read_text())["data"]
    vocab = json.loads(vocab_json.expanduser().read_text())
    if len(source) != 4150 or len({row["target_category"] for row in source}) != 23:
        raise ValueError("Expected BabyCL's frozen 4,150 trials over 23 categories")

    trials = []
    for row in source:
        category = row["target_category"]
        if category not in vocab or len(category.split()) != 1:
            raise ValueError(f"Target must be one token in the model vocabulary: {category}")
        paths = [row["target_img_filename"], *row["foil_img_filenames"]]
        if len(paths) != 4:
            raise ValueError("Each trial must contain one target and three foils")
        images = []
        for raw in paths:
            original = Path(raw)
            path = root / original.parent.name / original.name
            if not path.is_file():
                raise FileNotFoundError(path)
            images.append(str(path))
        trials.append({"text": category, "images": images, "answer": 0})

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump({"trials": trials}, handle, indent=2)
        handle.write("\n")
    return len(trials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--trials_json", type=Path, required=True)
    parser.add_argument("--vocab_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = prepare_manifest(args.data_root, args.trials_json, args.vocab_json, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Wrote {count} trials to {args.output}")


if __name__ == "__main__":
    main()
