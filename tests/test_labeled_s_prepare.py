"""Frozen-trial conversion uses synthetic metadata and empty image placeholders."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from egolearn.evaluation.benchmarks.labeled_s_4afc.prepare import prepare_manifest


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "images"
    categories = [f"category_{i}" for i in range(23)]
    for category in categories:
        directory = root / category
        directory.mkdir(parents=True)
        (directory / "image.jpg").touch()
    rows = []
    for i in range(4150):
        candidates = [categories[(i + offset) % 23] for offset in (0, 3, 2, 1)]
        rows.append(
            {
                "target_category": candidates[0],
                "target_img_filename": f"source/{candidates[0]}/image.jpg",
                "foil_img_filenames": [f"source/{c}/image.jpg" for c in candidates[1:]],
            }
        )
    source = tmp_path / "trials.json"
    source.write_text(json.dumps({"data": rows}))
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({c: i + 10 for i, c in enumerate(categories)}))
    return root, source, vocab, tmp_path / "output/manifest.json"


def test_cli_preserves_trial_and_candidate_order(inputs, tmp_path):
    root, source, vocab, output = inputs
    original_vocab = vocab.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "egolearn.evaluation.benchmarks.labeled_s_4afc.prepare",
            "--data_root",
            str(root),
            "--trials_json",
            str(source),
            "--vocab_json",
            str(vocab),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "Wrote 4150 trials" in result.stdout
    converted = json.loads(output.read_text())["trials"]
    rows = json.loads(source.read_text())["data"]
    for actual, row in zip(converted, rows, strict=True):
        filenames = [row["target_img_filename"], *row["foil_img_filenames"]]
        assert actual == {
            "text": row["target_category"],
            "images": [str(root / Path(p).parent.name / Path(p).name) for p in filenames],
            "answer": 0,
        }
    assert vocab.read_bytes() == original_vocab
    contents = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_manifest(*inputs)
    assert output.read_bytes() == contents


@pytest.mark.parametrize(
    "change,error,match",
    [
        ("trial_count", ValueError, "4,150"),
        ("category_count", ValueError, "23 categories"),
        ("foil_count", ValueError, "three foils"),
        ("missing_label", ValueError, "model vocabulary"),
        ("multiword_label", ValueError, "one token"),
        ("missing_image", FileNotFoundError, "image.jpg"),
    ],
)
def test_invalid_inputs_do_not_write_output(inputs, change, error, match):
    root, source, vocab_path, output = inputs
    rows = json.loads(source.read_text())["data"]
    vocab = json.loads(vocab_path.read_text())
    if change == "trial_count":
        rows.pop()
    elif change == "category_count":
        for row in rows:
            row["target_category"] = "category_0"
    elif change == "foil_count":
        rows[-1]["foil_img_filenames"].pop()
    elif change == "missing_label":
        vocab.pop("category_0")
    elif change == "multiword_label":
        vocab["two words"] = vocab.pop("category_0")
        for row in rows:
            if row["target_category"] == "category_0":
                row["target_category"] = "two words"
    else:
        (root / "category_0/image.jpg").unlink()
    source.write_text(json.dumps({"data": rows}))
    vocab_path.write_text(json.dumps(vocab))
    with pytest.raises(error, match=match):
        prepare_manifest(*inputs)
    assert not output.exists()
