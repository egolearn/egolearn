"""Konkle 4AFC: trial sampling, chance credit, and manifest preparation on synthetic data."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from egolearn.evaluation.benchmarks.konkle_4afc import Konkle4AFCProbe, sample_trials
from egolearn.evaluation.benchmarks.konkle_4afc.prepare import (
    IMAGES_PER_CATEGORY,
    TEST_CATEGORIES,
    prepare_manifest,
)


def test_trial_draws_follow_the_reference_order():
    sizes = {"dog": 5, "ball": 3, "cat": 4, "apple": 2, "shoe": 6}
    cats = sorted(sizes)
    starts = dict(zip(cats, np.cumsum([0, *[sizes[c] for c in cats]]).tolist(), strict=False))
    pools = {c: list(range(starts[c], starts[c] + sizes[c])) for c in cats}
    rng = np.random.default_rng(0)
    expected = {}
    for cat in cats:
        others = [c for c in cats if c != cat]
        # The draw below is the trial construction of eval_4afc_region in mcfrank/vlm-headcam.
        expected[cat] = [
            [rng.choice(pools[cat])]
            + [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
            for _ in range(7)
        ]
    trials = sample_trials(sizes, n_trials=7, seed=0)
    assert {c: [[starts[k] + i for k, i in t] for t in trials[c]] for c in cats} == expected
    assert all(t[0][0] == c and len({k for k, _ in t}) == 4 for c in cats for t in trials[c])
    with pytest.raises(ValueError, match="four categories"):
        sample_trials({"a": 1, "b": 1, "c": 1})


class _Model:
    """Images encode to their mean colour; each word to its category's colour."""

    def __init__(self, text):
        self.text = text

    def encode(self, modality, x, lengths=None):
        pooled = x.mean(dim=(2, 3)) if modality == "image" else self.text[x[:, 0]]
        return SimpleNamespace(pooled=pooled)

    def apply_head(self, head, x):
        return x


def test_out_of_vocabulary_categories_score_chance(tmp_path, monkeypatch):
    colours = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]])
    names = ["apple", "ball", "cat", "dog", "shoe", "tent"]
    categories = {}
    for index, name in enumerate(names):
        colour = colours[min(index, 4)]
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (8, 8), tuple(int(255 * v) for v in colour)).save(path)
        categories[name] = [{"path": str(path), "sha256": "0" * 64}] * 2
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "test60", "categories": categories}))
    vocab = tmp_path / "vocab.json"
    ids = {name: index + 2 for index, name in enumerate(names[:5])}
    vocab.write_text(json.dumps({"<pad>": 0, "<unk>": 1, **ids}))

    monkeypatch.setattr(
        "egolearn.evaluation.benchmarks.konkle_4afc.probe.VIEWS.get",
        lambda view: (
            lambda size: (
                lambda image: torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(
                    2, 0, 1
                )
            )
        ),
    )
    probe = Konkle4AFCProbe(
        "konkle_4afc",
        str(manifest),
        str(vocab),
        "embed",
        n_trials=20,
        output_json=str(tmp_path / "detail.json"),
    )
    text = torch.zeros(7, 3)
    text[2:7] = colours
    result = probe.run(_Model(text), SimpleNamespace(device=torch.device("cpu")))

    assert result["four_afc_acc_in_vocab"] == 1.0
    assert result["four_afc_oov_categories"] == 1.0 and result["four_afc_categories"] == 6.0
    assert result["four_afc_acc"] == pytest.approx((5 * 1.0 + 0.25) / 6)
    detail = json.loads((tmp_path / "detail.json").read_text())
    assert detail["out_of_vocabulary"] == ["tent"] and detail["per_category"]["tent"] == 0.25
    assert probe.metadata()["n_images"] == 12


def test_prepare_reads_test_items_and_rejects_incomplete_categories(tmp_path):
    root = tmp_path / "17-objects"
    for category in TEST_CATEGORIES:
        (root / category / "TestItems").mkdir(parents=True)
        for index in range(IMAGES_PER_CATEGORY - 2):
            (root / category / f"b{index:02d}.jpg").write_bytes(b"x")
        (root / category / "TestItems" / "a.JPG").write_bytes(b"y")
        (root / category / "TestItems" / "z.jpg").write_bytes(b"z")
        (root / category / "Thumbs.db").touch()
        (root / category / "._b00.jpg").touch()
    output = tmp_path / "out/manifest.json"
    assert prepare_manifest(root, output) == 60 * IMAGES_PER_CATEGORY
    listed = json.loads(output.read_text())["categories"]
    assert list(listed) == sorted(TEST_CATEGORIES) and len(listed) == 60
    names = [entry["path"].rsplit("/", 1)[1] for entry in listed["dog"]]
    assert names == ["a.JPG", *[f"b{i:02d}.jpg" for i in range(15)], "z.jpg"]
    with pytest.raises(FileExistsError):
        prepare_manifest(root, output)

    (root / "dog" / "b00.jpg").unlink()
    incomplete = tmp_path / "incomplete.json"
    with pytest.raises(ValueError, match="17 images for 'dog'"):
        prepare_manifest(root, incomplete)
    assert not incomplete.exists()
