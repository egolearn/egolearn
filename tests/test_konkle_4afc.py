"""Konkle 4AFC: trial sampling, chance and tie credit, and manifest preparation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from egolearn.evaluation.benchmarks.konkle_4afc.prepare import (
    IMAGES_PER_CATEGORY,
    TEST_CATEGORIES,
    prepare_manifest,
    sample_trials,
)
from egolearn.evaluation.four_afc import FourAFCProbe


def test_trial_draws_follow_the_reference_order():
    sizes = {"dog": 5, "ball": 3, "cat": 4, "apple": 2, "shoe": 6}
    cats = sorted(sizes)
    pools = {c: [f"{c}/{i}.jpg" for i in range(sizes[c])] for c in cats}
    rng = np.random.default_rng(0)
    expected = []
    for cat in cats:
        others = [c for c in cats if c != cat]
        # The draw below is the trial construction of eval_4afc_region in mcfrank/vlm-headcam.
        for _ in range(7):
            images = [rng.choice(pools[cat])]
            images += [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
            expected.append({"text": cat, "images": images, "answer": 0})
    trials = sample_trials(pools, n_trials=7, seed=0)
    assert trials == expected
    assert all(len({image.split("/")[0] for image in t["images"]}) == 4 for t in trials)
    with pytest.raises(ValueError, match="four categories"):
        sample_trials({"a": ["x"], "b": ["x"], "c": ["x"]})


def _image_root(tmp_path):
    root = tmp_path / "17-objects"
    for category in TEST_CATEGORIES:
        (root / category / "TestItems").mkdir(parents=True)
        for index in range(IMAGES_PER_CATEGORY - 2):
            (root / category / f"b{index:02d}.jpg").write_bytes(b"x")
        (root / category / "TestItems" / "a.JPG").write_bytes(b"y")
        (root / category / "TestItems" / "z.jpg").write_bytes(b"z")
        (root / category / "Thumbs.db").touch()
        (root / category / "._b00.jpg").touch()
    return root


def test_prepare_reads_test_items_and_excludes_out_of_vocabulary_categories(tmp_path):
    root = _image_root(tmp_path)
    vocab = tmp_path / "vocab.json"
    oov = ["tent", "trumpet"]
    vocab.write_text(json.dumps({c: i for i, c in enumerate(TEST_CATEGORIES) if c not in oov}))
    output = tmp_path / "out/trials.json"
    report = prepare_manifest(root, vocab, output, n_trials=5)
    assert report["oov_categories_at_chance"] == oov and report["n_categories"] == 60
    manifest = json.loads(output.read_text())
    assert manifest["chance_trials"] == 10 and len(manifest["trials"]) == 58 * 5
    names = {image.rsplit("/", 1)[1] for t in manifest["trials"] for image in t["images"]}
    assert names <= {"a.JPG", "z.jpg", *[f"b{i:02d}.jpg" for i in range(15)]} and "a.JPG" in names
    images = [image for t in manifest["trials"] for image in t["images"]]
    assert not any("/tent/" in image or "/trumpet/" in image for image in images)
    assert all(t["text"] not in oov for t in manifest["trials"])
    assert all(str(root / t["text"]) in t["images"][0] for t in manifest["trials"])
    with pytest.raises(FileExistsError):
        prepare_manifest(root, vocab, output)

    (root / "dog" / "b00.jpg").unlink()
    incomplete = tmp_path / "incomplete.json"
    with pytest.raises(ValueError, match="17 images for 'dog'"):
        prepare_manifest(root, vocab, incomplete)
    assert not incomplete.exists()


class _Model:
    """Images encode to their mean colour (or ``image_value``); words to their category's colour."""

    def __init__(self, text, image_value=None):
        self.text = text
        self.image_value = image_value

    def encode(self, modality, x, lengths=None):
        if modality == "text":
            return SimpleNamespace(pooled=self.text[x[:, 0]])
        pooled = x.mean(dim=(2, 3))
        if self.image_value is not None:
            pooled = torch.full_like(pooled, self.image_value)
        return SimpleNamespace(pooled=pooled)

    def apply_head(self, head, x):
        return x


NAMES = ["apple", "ball", "cat", "dog"]


def _colour_images(tmp_path, monkeypatch):
    """Save one solid-colour image per name; return their paths, vocabulary, and word colours."""
    colours = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    paths = []
    for name, colour in zip(NAMES, colours, strict=True):
        paths.append(str(tmp_path / f"{name}.png"))
        Image.new("RGB", (8, 8), tuple(int(255 * v) for v in colour)).save(paths[-1])
    vocab = tmp_path / "vocab.json"
    ids = {name: index + 2 for index, name in enumerate(NAMES)}
    vocab.write_text(json.dumps({"<pad>": 0, "<unk>": 1, **ids}))
    monkeypatch.setattr(
        "egolearn.evaluation.four_afc.VIEWS.get",
        lambda view: (
            lambda size: (
                lambda image: torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(
                    2, 0, 1
                )
            )
        ),
    )
    text = torch.zeros(6, 3)
    text[2:6] = colours
    return paths, vocab, text


def test_four_afc_scores_chance_trials_at_one_quarter(tmp_path, monkeypatch):
    paths, vocab, text = _colour_images(tmp_path, monkeypatch)
    trials = [
        {"text": name, "images": [paths[i], *paths[:i], *paths[i + 1 :]], "answer": 0}
        for i, name in enumerate(NAMES)
    ]
    trials.append({"text": "apple", "images": paths[1:] + paths[:1], "answer": 0})
    ctx = SimpleNamespace(device=torch.device("cpu"))

    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"trials": trials}))
    assert FourAFCProbe("p", str(plain), str(vocab), "embed").run(_Model(text), ctx) == {
        "four_afc_acc": 0.8
    }
    credited = tmp_path / "credited.json"
    credited.write_text(json.dumps({"trials": trials, "chance_trials": 5}))
    result = FourAFCProbe("p", str(credited), str(vocab), "embed").run(_Model(text), ctx)
    assert result == {"four_afc_acc": pytest.approx(0.525), "four_afc_acc_scored": 0.8}
    for bad in (0.9, -1, True, "5"):
        credited.write_text(json.dumps({"trials": trials, "chance_trials": bad}))
        with pytest.raises(ValueError, match="non-negative integer"):
            FourAFCProbe("p", str(credited), str(vocab), "embed")


def test_four_afc_splits_credit_among_tied_images(tmp_path, monkeypatch):
    paths, vocab, text = _colour_images(tmp_path, monkeypatch)
    red, green, blue, _ = paths
    trials = [
        {"text": "apple", "images": [red, red, green, blue], "answer": 0},
        {"text": "apple", "images": [red, green, red, red], "answer": 0},
        {"text": "apple", "images": [green, red, red, blue], "answer": 2},
    ]
    manifest = tmp_path / "trials.json"
    manifest.write_text(json.dumps({"trials": trials}))
    probe = FourAFCProbe("p", str(manifest), str(vocab), "embed")
    assert probe.version == "2"
    ctx = SimpleNamespace(device=torch.device("cpu"))

    result = probe.run(_Model(text), ctx)
    assert result == {"four_afc_acc": pytest.approx((1 / 2 + 1 / 3 + 1 / 2) / 3)}
    # Identical image embeddings tie all four images, so the model scores chance.
    assert probe.run(_Model(text, image_value=1.0), ctx) == {"four_afc_acc": 0.25}
    with pytest.raises(ValueError, match="non-finite"):
        probe.run(_Model(text, image_value=float("nan")), ctx)
    # A negative answer would otherwise index the last image.
    for bad in (-1, 4, True, "0"):
        manifest.write_text(json.dumps({"trials": [{**trials[0], "answer": bad}]}))
        with pytest.raises(ValueError, match="answer must index"):
            FourAFCProbe("p", str(manifest), str(vocab), "embed")
