"""Labeled-S probe backed by the generic fixed-feature linear scorer.

The benchmark dataset contains one dataset-relative image
path and integer label per row.  The first 2,886 rows are training examples;
the remaining 2,901 are evaluation examples.  This adapter resolves those
paths against a caller-provided Labeled-S root and constructs the generic
``ClassificationSplit`` in memory -- no generated, machine-specific manifest.

The small split file is downloaded from an immutable upstream commit by
default and verified by SHA-256.  ``split_file`` supports offline clusters and
local mirrors.  Labeled-S itself is licensed separately and is never
downloaded here.

The benchmark was first proposed in "How Well Do Unsupervised Learning Algorithms
Model Human Real-time and Life-long Learning?" by Chengxu Zhuang et al.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from egolearn.config.registry import PROBES
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage, LinearProbe

SPLIT_COMMIT = "c084d826fdc85a72e78477bb53c88b7353a89563"
SPLIT_URL = (
    "https://raw.githubusercontent.com/agentic-learning-ai-lab/memory-storyboard/"
    f"{SPLIT_COMMIT}/metas/labeledS_train_val.txt"
)
SPLIT_SHA256 = "5e795939ba980779bb8e918847cd7df8f8d1310aac3a0dfa2ba0d6e1232cc349"
TRAIN_SIZE = 2886
TOTAL_SIZE = 5787
N_CLASSES = 26


def load_split_text(split_file: str | Path | None) -> tuple[str, str, str]:
    """Return metadata text, its source description, and SHA-256 digest."""
    if split_file is None:
        print(f"[download] {SPLIT_URL}")
        with urllib.request.urlopen(SPLIT_URL, timeout=60) as response:
            raw = response.read()
        source = SPLIT_URL
    else:
        path = Path(split_file)
        raw = path.read_bytes()
        source = str(path.resolve())

    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPLIT_SHA256:
        raise ValueError(
            f"Labeled-S split has an unexpected SHA-256: expected {SPLIT_SHA256}, got {digest}"
        )
    try:
        return raw.decode("utf-8"), source, digest
    except UnicodeDecodeError as exc:
        raise ValueError(f"Labeled-S split metadata is not UTF-8: {source}") from exc


def parse_split(text: str, data_root: str | Path, check_files: bool = True) -> ClassificationSplit:
    """Validate canonical metadata and construct the in-memory split."""
    lines = text.splitlines()
    if len(lines) != TOTAL_SIZE:
        raise ValueError(f"expected {TOTAL_SIZE} metadata rows, got {len(lines)}")

    root = Path(data_root).resolve()
    items: list[LabeledImage] = []
    class_for_label: dict[int, str] = {}
    label_for_class: dict[str, int] = {}
    seen_paths: set[str] = set()
    missing: list[Path] = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank metadata row at line {line_number}")
        try:
            path_text, label_text = line.rsplit(maxsplit=1)
            label = int(label_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid metadata row at line {line_number}: {line!r}") from exc
        if not 0 <= label < N_CLASSES:
            raise ValueError(
                f"label at line {line_number} must be in [0, {N_CLASSES - 1}], got {label}"
            )

        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError(
                f"image path at line {line_number} must be dataset-relative: {path_text!r}"
            )
        class_name = relative.parts[0]
        previous_class = class_for_label.setdefault(label, class_name)
        previous_label = label_for_class.setdefault(class_name, label)
        if previous_class != class_name or previous_label != label:
            raise ValueError(
                f"inconsistent class/label mapping at line {line_number}: {class_name!r} -> {label}"
            )

        image = (root / relative).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"image path escapes data root at line {line_number}: {path_text!r}")
        image_text = str(image)
        if image_text in seen_paths:
            raise ValueError(f"duplicate image path at line {line_number}: {path_text!r}")
        seen_paths.add(image_text)
        if check_files and not image.is_file() and len(missing) < 10:
            missing.append(image)
        items.append(LabeledImage(image=image_text, label=label))

    expected_labels = set(range(N_CLASSES))
    labels = set(class_for_label)
    if labels != expected_labels:
        raise ValueError(
            f"expected labels 0..{N_CLASSES - 1}, missing {sorted(expected_labels - labels)}, "
            f"unexpected {sorted(labels - expected_labels)}"
        )
    if missing:
        shown = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing Labeled-S images (showing up to 10):\n  {shown}")

    train = tuple(items[:TRAIN_SIZE])
    test = tuple(items[TRAIN_SIZE:])
    for split_name, split_items in (("train", train), ("evaluation", test)):
        split_labels = {item.label for item in split_items}
        if split_labels != expected_labels:
            raise ValueError(
                f"{split_name} split does not contain all labels; "
                f"missing {sorted(expected_labels - split_labels)}"
            )

    classes = tuple(class_for_label[label] for label in range(N_CLASSES))
    return ClassificationSplit(classes=classes, train=train, test=test)


@PROBES.register("labeled_s_linear")
class LabeledSLinearProbe(LinearProbe):
    """Labeled-S data adapter plus the shared deterministic linear scorer."""

    def __init__(
        self,
        name: str,
        data_root: str,
        split_file: str | None = None,
        check_files: bool = True,
        encoder: str = "image",
        head: str | None = None,
        view: str = "center_crop",
        size: int = 224,
        batch_size: int = 256,
        l2: float = 1e-3,
        max_iter: int = 500,
    ):
        text, self.split_source, self.split_sha256 = load_split_text(split_file)
        split = parse_split(text, data_root, check_files=check_files)
        super().__init__(
            name=name,
            split=split,
            encoder=encoder,
            head=head,
            view=view,
            size=size,
            batch_size=batch_size,
            l2=l2,
            max_iter=max_iter,
        )
