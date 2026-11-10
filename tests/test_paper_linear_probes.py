"""BabyCL paper linear-probe protocols: stratified Labeled-S, mini-ImageNet, sklearn-style C."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from egolearn.core.context import Context
from egolearn.evaluation.benchmarks.labeled_s_linear import stratified
from egolearn.evaluation.benchmarks.mini_imagenet_linear import probe as mini_imagenet
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage, LinearProbe


def _image_folder(root, counts):
    """ImageFolder layout with ``counts[i]`` files in class ``c{i:02d}``, created unsorted."""
    for label, count in enumerate(counts):
        directory = root / f"c{label:02d}"
        directory.mkdir(parents=True)
        for j in reversed(range(count)):
            Image.new("RGB", (8, 8), (label, j, 0)).save(directory / f"img_{j:03d}.jpg")


def test_stratified_split_shuffles_each_category_in_imagefolder_order(tmp_path):
    counts = [1 + (label * 7) % 11 for label in range(stratified.N_CLASSES)]
    _image_folder(tmp_path, counts)

    split = stratified.build_stratified_split(tmp_path, train_fraction=0.8, seed=42)

    # reference: sorted classes and files; one RandomState drawn in label order
    rng = np.random.RandomState(42)
    expected_train, expected_test = [], []
    for label, count in enumerate(counts):
        files = [
            str(tmp_path.resolve() / f"c{label:02d}" / f"img_{j:03d}.jpg") for j in range(count)
        ]
        order = np.arange(count)
        rng.shuffle(order)
        n_train = max(1, int(0.8 * count))
        expected_train += [(files[i], label) for i in order[:n_train]]
        expected_test += [(files[i], label) for i in order[n_train:]]
    assert split.classes == tuple(f"c{label:02d}" for label in range(stratified.N_CLASSES))
    assert [(item.image, item.label) for item in split.train] == expected_train
    assert [(item.image, item.label) for item in split.test] == expected_test


def test_stratified_split_rejects_a_wrong_category_count(tmp_path):
    _image_folder(tmp_path, [2, 2])
    with pytest.raises(ValueError, match="26 Labeled-S categories"):
        stratified.build_stratified_split(tmp_path)


def test_stratified_probe_records_its_protocol_without_local_paths(tmp_path):
    from egolearn.evaluation.evaluator import _configured_benchmark

    root = tmp_path / "saycam_labeled"
    _image_folder(root, [5] * stratified.N_CLASSES)
    probe, config = _configured_benchmark(
        {
            "_target_": "probe:labeled_s_stratified_linear",
            "name": "labeled_s_linear_paper",
            "data_root": str(root),
            "head": "embed512",
        }
    )
    assert isinstance(probe, stratified.LabeledSStratifiedLinearProbe)
    assert config["feature_norm"] == "l2" and config["c"] == 0.316
    assert config["max_iter"] == 1000 and config["split_seed"] == 42
    assert config["l2"] == pytest.approx(1 / (2 * 0.316 * len(probe.split.train)))
    assert config["data_root"] == "saycam_labeled"
    assert str(tmp_path) not in json.dumps(config) + json.dumps(probe.metadata())


def test_mini_imagenet_probe_reads_train_and_val(tmp_path):
    for split_name in ("train", "val"):
        _image_folder(tmp_path / split_name, [1] * mini_imagenet.N_CLASSES)

    probe = mini_imagenet.MiniImageNetLinearProbe(name="mini", data_root=str(tmp_path))

    assert len(probe.split.classes) == 100
    assert [item.label for item in probe.split.train] == list(range(100))
    assert [item.label for item in probe.split.test] == list(range(100))
    assert probe.feature_norm == "l2" and probe.c == 0.316
    small = tmp_path / "small"
    for split_name in ("train", "val"):
        _image_folder(small / split_name, [1, 1])
    with pytest.raises(ValueError, match="expected 100"):
        mini_imagenet.MiniImageNetLinearProbe(name="mini", data_root=str(small))


class _FixedFeatureProbe(LinearProbe):
    """Fixed features stand in for a model: only the classifier fit is under test."""

    def __init__(self, features, labels, **kwargs):
        items = tuple(LabeledImage(image=f"/x/{i}.jpg", label=int(y)) for i, y in enumerate(labels))
        split = ClassificationSplit(classes=("a", "b", "c"), train=items, test=items)
        super().__init__(name="lp", split=split, **kwargs)
        self.features, self.labels = features, torch.as_tensor(labels)

    def _extract(self, model, ctx, items):
        return self.features.clone(), self.labels


def _fixed_feature_probe(features, labels, **kwargs):
    return _FixedFeatureProbe(features, labels, **kwargs)


def test_unit_length_features_and_sklearn_penalty_define_the_objective():
    generator = torch.Generator().manual_seed(0)
    labels = [i % 3 for i in range(60)]
    centers = torch.randn(3, 8, generator=generator)
    features = centers[labels] + 0.8 * torch.randn(60, 8, generator=generator)
    ctx = Context(seed=0, device="cpu")

    probe = _fixed_feature_probe(features, labels, feature_norm="l2", c=0.316, max_iter=200)
    assert probe.l2 == pytest.approx(1 / (2 * 0.316 * 60))
    # unit-length scaling makes the fit invariant to each sample's norm
    scales = torch.linspace(0.1, 10.0, 60).unsqueeze(1)
    rescaled = _fixed_feature_probe(
        features * scales, labels, feature_norm="l2", c=0.316, max_iter=200
    )
    assert probe.run(None, ctx) == pytest.approx(rescaled.run(None, ctx))


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"feature_norm": "zscore"}, "feature_norm"), ({"c": 0.0}, "positive")]
)
def test_linear_probe_rejects_invalid_protocol_settings(kwargs, match):
    items = (LabeledImage(image="/x/a.jpg", label=0), LabeledImage(image="/x/b.jpg", label=1))
    split = ClassificationSplit(classes=("a", "b"), train=items, test=items)
    with pytest.raises(ValueError, match=match):
        LinearProbe(name="lp", split=split, **kwargs)


def test_linear_probe_defaults_are_unchanged():
    items = (LabeledImage(image="/x/a.jpg", label=0), LabeledImage(image="/x/b.jpg", label=1))
    probe = LinearProbe(
        name="lp", split=ClassificationSplit(classes=("a", "b"), train=items, test=items)
    )
    assert probe.feature_norm == "standardize" and probe.c is None and probe.l2 == 1e-3
