"""Benchmark data preparation (babyview_4afc) and the generic linear probe."""

import csv
import random

import pytest
import torch
from PIL import Image

from egolearn.evaluation.benchmarks.babyview_4afc import prepare


def _fake_crops(tmp_path, n_cats=6, per_cat=6, size=32):
    """A miniature copy of the release layout: manifest.csv + crops/<cat>/*.jpg,
    one flat color per category, varied aspect ratios."""
    data = tmp_path / "data"
    rows = []
    for ci in range(n_cats):
        cat = f"cat{ci}"
        d = data / "crops" / cat
        d.mkdir(parents=True)
        color = (200 - 30 * ci, 30 * ci, 40 + 25 * ci)
        for j in range(per_cat):
            path = d / f"{cat}_{j}.jpg"
            Image.new("RGB", (size + 7 * j, size), color).save(path)
            rows.append((cat, f"{cat}_{j}", f"crops/{cat}/{cat}_{j}.jpg"))
    with (data / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "stem", "jpeg_path"])
        w.writerows(rows)
    return data


def test_babyview_prepare_follows_the_frozen_protocol(tmp_path):
    data = _fake_crops(tmp_path)
    # cat4/cat5 are out of vocabulary and must never appear in trials
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "cat0": 3, "cat1": 4, "cat2": 5, "cat3": 6}

    by_cat = prepare.load_manifest(data)
    all_cats = sorted(by_cat)
    in_vocab = [c for c in all_cats if c in vocab]
    assert in_vocab == ["cat0", "cat1", "cat2", "cat3"]

    trials = prepare.generate_trials(by_cat, in_vocab, n_trials=60, n_foils=3, seed=0)
    assert len(trials) == 60
    for t in trials:
        assert t["answer"] == 0 and len(t["images"]) == 4
        assert t["text"] in vocab
        cats = [p.split("/crops/")[1].split("/")[0] for p in t["images"]]
        assert cats[0] == t["text"]  # target first
        assert len(set(cats)) == 4  # target + 3 DISTINCT foil categories
        assert not {"cat4", "cat5"} & set(cats)  # OOV never appears
    # frozen: same seed regenerates the identical trial list
    assert prepare.generate_trials(by_cat, in_vocab, 60, 3, 0) == trials

    split = prepare.generate_split(by_cat, test_frac=0.2, seed=0)
    assert split["classes"] == all_cats  # ALL categories, no vocabulary filter
    train, test = split["train"], split["test"]
    assert not {e["image"] for e in train} & {e["image"] for e in test}
    assert len(train) + len(test) == 6 * 6
    for label in range(len(all_cats)):  # stratified: every class in both sides
        assert any(e["label"] == label for e in train)
        assert any(e["label"] == label for e in test)
    assert prepare.generate_split(by_cat, 0.2, 0) == split


def test_linear_probe_learns_and_consumes_no_randomness(tmp_path):
    import json

    from egolearn.core.context import Context
    from egolearn.evaluation.linear_probe import LinearProbe
    from egolearn.model.encoders.image import ToyCNNEncoder
    from egolearn.model.model import ComposedModel

    # two flat-color classes: linearly separable from any conv features
    items: dict[str, list] = {"classes": ["red", "blue"], "train": [], "test": []}
    for label, color in enumerate([(220, 40, 40), (40, 40, 220)]):
        for j in range(12):
            p = tmp_path / f"img_{label}_{j}.jpg"
            Image.new("RGB", (40, 32), color).save(p)
            dest = "train" if j < 9 else "test"
            items[dest].append({"image": str(p), "label": label})
    manifest = tmp_path / "linear.json"
    manifest.write_text(json.dumps(items))

    model = ComposedModel({"vision": ToyCNNEncoder(dim=32)}).eval()
    ctx = Context(seed=0, device="cpu")
    probe = LinearProbe(name="lp", manifest=str(manifest), view="center_crop", size=32)

    torch_state = torch.get_rng_state().clone()
    py_state = random.getstate()
    with torch.no_grad():  # ProbeHook runs probes under no_grad
        first = probe.run(model, ctx)
        second = probe.run(model, ctx)

    assert first["top1"] == pytest.approx(1.0)  # trivially separable
    assert first == second  # deterministic solver, frozen split
    # the probe must not perturb the run it is scoring
    assert torch.equal(torch_state, torch.get_rng_state())
    assert py_state == random.getstate()
