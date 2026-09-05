"""Benchmark data preparation and the generic linear probe."""

import csv
import hashlib
import random

import pytest
import torch
import torch.nn as nn
from PIL import Image

from egolearn.evaluation import gradient_linear_probe
from egolearn.evaluation.benchmarks.babyview_4afc import prepare
from egolearn.evaluation.benchmarks.imagenet_linear import probe as imagenet_linear
from egolearn.evaluation.benchmarks.labeled_s_linear import probe as labeled_s_linear
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage


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
    # a degenerate fraction cannot empty either side (autoresearch finding)
    with pytest.raises(SystemExit, match="test_frac"):
        prepare.generate_split(by_cat, 1.0, 0)
    high = prepare.generate_split(by_cat, 0.99, 0)
    assert all(any(e["label"] == c for e in high["train"]) for c in range(len(all_cats)))


def test_labeled_s_adapter_follows_the_frozen_protocol(tmp_path, monkeypatch):
    rows = []
    expected_classes = tuple(f"class_{label}" for label in range(labeled_s_linear.N_CLASSES))
    for index in range(labeled_s_linear.TOTAL_SIZE):
        label = index % labeled_s_linear.N_CLASSES
        rows.append(f"class_{label}/image_{index}.jpeg {label}")

    split_text = "\n".join(rows) + "\n"
    split_file = tmp_path / "labeledS_train_val.txt"
    split_file.write_text(split_text)
    monkeypatch.setattr(
        labeled_s_linear, "SPLIT_SHA256", hashlib.sha256(split_text.encode()).hexdigest()
    )
    probe = labeled_s_linear.LabeledSLinearProbe(
        name="labeled_s_linear",
        data_root=str(tmp_path / "labeled_s"),
        split_file=str(split_file),
        check_files=False,
        size=32,
    )
    gradient_probe = labeled_s_linear.LabeledSGradientLinearProbe(
        name="labeled_s_linear_sgd",
        data_root=str(tmp_path / "labeled_s"),
        split_file=str(split_file),
        check_files=False,
        workers=0,
    )

    assert probe.split.classes == expected_classes
    assert gradient_probe.split == probe.split
    assert gradient_probe.size == 112
    assert gradient_probe.epochs == 10
    assert len(probe.split.train) == 2886
    assert len(probe.split.test) == 2901
    assert probe.split.train[-1].image.endswith("class_25/image_2885.jpeg")
    assert probe.split.test[0].image.endswith("class_0/image_2886.jpeg")
    assert {item.label for item in probe.split.train} == set(range(26))
    assert {item.label for item in probe.split.test} == set(range(26))


def test_labeled_s_adapter_rejects_noncanonical_metadata(tmp_path):
    with pytest.raises(ValueError, match="expected 5787 metadata rows"):
        labeled_s_linear.parse_split("ball/image.jpeg 0\n", tmp_path, check_files=False)


def test_labeled_s_adapter_verifies_local_split(tmp_path):
    split_file = tmp_path / "labeledS_train_val.txt"
    split_file.write_text("ball/image.jpeg 0\n")
    with pytest.raises(ValueError, match="unexpected SHA-256"):
        labeled_s_linear.load_split_text(split_file)


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


def test_gradient_linear_probe_learns_and_preserves_model_and_rng(tmp_path):
    from egolearn.core.context import Context
    from egolearn.model.model import ComposedModel

    class MeanEncoder(nn.Module):
        modality = "image"
        out_dim = 3

        def forward(self, value):
            return value.mean(dim=(2, 3))

    train: list[LabeledImage] = []
    test: list[LabeledImage] = []
    for label, color in enumerate([(230, 20, 20), (20, 20, 230)]):
        for index in range(10):
            path = tmp_path / f"gradient_{label}_{index}.png"
            Image.new("RGB", (32, 32), color).save(path)
            item = LabeledImage(str(path), label)
            (train if index < 8 else test).append(item)
    split = ClassificationSplit(("red", "blue"), tuple(train), tuple(test))

    model = ComposedModel({"vision": MeanEncoder()}).train()
    ctx = Context(seed=17, device="cpu")
    probe = gradient_linear_probe.GradientLinearProbe(
        name="gradient",
        split=split,
        encoder="vision",
        size=16,
        batch_size=4,
        workers=0,
        epochs=4,
        lr=1.0,
        momentum=0.0,
        larc_trust_coefficient=0.1,
        seed=5,
    )

    model_state = {name: value.clone() for name, value in model.state_dict().items()}
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    with torch.no_grad():
        first = probe.run(model, ctx)
        second = probe.run(model, ctx)

    assert first["top1"] == pytest.approx(1.0)
    assert first == second
    assert model.training
    assert all(torch.equal(model_state[name], value) for name, value in model.state_dict().items())
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()


def test_gradient_linear_probe_augmentations_are_worker_count_invariant(tmp_path):
    from egolearn.core.context import Context

    items: list[LabeledImage] = []
    for index in range(6):
        image = Image.new("RGB", (48 + index, 36))
        pixels = image.load()
        assert pixels is not None
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = (
                    (11 * x + 17 * y + index) % 256,
                    (23 * x + 5 * y + index) % 256,
                    (3 * x + 29 * y + index) % 256,
                )
        path = tmp_path / f"worker_invariant_{index}.png"
        image.save(path)
        items.append(LabeledImage(str(path), index))

    split = ClassificationSplit(
        tuple(str(index) for index in range(len(items))), tuple(items), tuple(items)
    )
    ctx = Context(seed=17, device="cpu")

    def collect(workers: int, num_epochs: int) -> list[tuple[list[int], dict[int, torch.Tensor]]]:
        probe = gradient_linear_probe.GradientLinearProbe(
            name="gradient",
            split=split,
            size=16,
            batch_size=2,
            workers=workers,
            epochs=2,
            seed=5,
        )
        generator = torch.Generator().manual_seed(probe.seed)
        loader = probe._loader(
            split.train,
            gradient_linear_probe._train_transform(probe.size),
            True,
            generator,
            ctx.derive_seed,
            "train",
        )
        epochs = []
        for _ in range(num_epochs):
            order: list[int] = []
            views: dict[int, torch.Tensor] = {}
            for images, labels in loader:
                order.extend(int(label) for label in labels)
                views.update(
                    (int(label), image.clone()) for image, label in zip(images, labels, strict=True)
                )
            epochs.append((order, views))
        return epochs

    serial = collect(0, 2)
    pooled = collect(2, 2)

    for epoch in range(len(pooled)):
        serial_order, serial_views = serial[epoch]
        pooled_order, pooled_views = pooled[epoch]
        assert serial_order == pooled_order
        assert serial_views.keys() == pooled_views.keys()
        assert all(torch.equal(serial_views[index], pooled_views[index]) for index in serial_views)
    assert any(not torch.equal(serial[0][1][index], serial[1][1][index]) for index in serial[0][1])


def test_gradient_linear_probe_larc_matches_reference_update():
    parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([6.0, 8.0])
    optimizer = torch.optim.SGD([parameter], lr=2.0, momentum=0.0, weight_decay=0.0)
    larc = gradient_linear_probe._LARC(optimizer, trust_coefficient=0.1, eps=0.0)

    larc.step()

    assert parameter.detach().tolist() == pytest.approx([2.4, 3.2])
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert gradient_linear_probe._cosine_lr(3.0, 0, 10) == pytest.approx(3.0)
    assert gradient_linear_probe._cosine_lr(3.0, 5, 10) == pytest.approx(1.5)


def test_imagenet_linear_adapter_reads_imagefolder_layout(tmp_path):
    for split_name in ("train", "val"):
        for label, class_name in enumerate(("n00000001", "n00000002")):
            directory = tmp_path / split_name / class_name
            directory.mkdir(parents=True)
            Image.new("RGB", (20, 20), (200 * label, 20, 20)).save(directory / "image.jpg")

    split = imagenet_linear.build_split(tmp_path, expected_classes=2)

    assert split.classes == ("n00000001", "n00000002")
    assert [item.label for item in split.train] == [0, 1]
    assert [item.label for item in split.test] == [0, 1]

    probe = imagenet_linear.ImageNetLinearProbe(
        name="imagenet_linear", data_root=str(tmp_path), expected_classes=2, workers=0
    )
    assert probe.size == 224
