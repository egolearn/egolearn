"""Dataset integrity and the Experiment 3 target-selection regression."""

import json
from collections import Counter, defaultdict

import pytest

pytest.importorskip("PIL")
from PIL import Image

from egolearn.evaluation.benchmarks.shape_bias.stimuli import sha256_file
from egolearn.evaluation.benchmarks.spatial_relations.prepare import (
    _trial,
    audit_trials,
    prepare_exp3,
    validate_dataset,
)


def _renders(root):
    for folder, reference in (("quinn_above_below", "Table"), ("quinn_between", "Rack")):
        directory = root / folder
        directory.mkdir(parents=True)
        for index in range(2):
            for obj in ("Ball", "Duck", "Lego", "Robot"):
                for i, role in enumerate(("habituation", "same", "different")):
                    Image.new("RGB", (16, 16), (20 * i, index, len(obj))).save(
                        directory / f"{reference}_{obj}_{index:04d}_{role}.png"
                    )


def _prepare(source, output, seed=33):
    output.mkdir()
    return prepare_exp3(source, output, seed=seed, expected_objects=4, expected_configurations=2)


def test_exp3_balanced_complete_and_reproducible(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = _prepare(source, first)
    _prepare(source, second)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "trials.jsonl").read_bytes() == (second / "trials.jsonl").read_bytes()
    assert manifest["n_trials"] == 32
    assert manifest["n_images"] == 48
    assert set(manifest["condition_counts"].values()) == {8}
    trials = [json.loads(line) for line in (first / "trials.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial["group_id"], trial["target_condition"]].append(trial)
        chosen = trial["candidates"][trial["correct_index"]]
        foil = trial["candidates"][1 - trial["correct_index"]]
        assert chosen.endswith("_same.png")
        assert foil.endswith("_different.png")
        assert (trial["anchor_object"] == trial["candidate_object"]) == (
            trial["target_condition"] == "identical"
        )
    for group in grouped.values():
        assert Counter(row["anchor_object"] for row in group) == Counter(
            row["candidate_object"] for row in group
        )
        assert Counter(row["correct_index"] for row in group) == {0: 2, 1: 2}
    assert validate_dataset(first)["n_trials"] == 32


def test_exp3_rejects_incomplete_source(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    (source / "quinn_between/Rack_Duck_0000_same.png").unlink()
    with pytest.raises(ValueError, match="incomplete image triplet"):
        _prepare(source, tmp_path / "output")


def test_exp3_rejects_missing_object(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    for path in (source / "quinn_between").glob("Rack_Duck_0000_*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="object coverage"):
        _prepare(source, tmp_path / "output")


def test_exp3_rejects_entire_missing_configuration_in_both_relations(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    for path in source.glob("*/*_0001_*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="configuration coverage"):
        _prepare(source, tmp_path / "output")


def test_validation_detects_corruption(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    output = tmp_path / "output"
    manifest = _prepare(source, output)
    image = output / manifest["images"][0]["path"]
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="image checksum"):
        validate_dataset(output)


def test_validation_rejects_path_escape(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    output = tmp_path / "output"
    manifest = _prepare(source, output)
    manifest["trials"]["path"] = "../outside.jsonl"
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonlocal"):
        validate_dataset(output)


def test_validation_rejects_missing_condition_even_with_updated_hash(tmp_path):
    source = tmp_path / "source"
    _renders(source)
    output = tmp_path / "output"
    manifest = _prepare(source, output)
    path = output / "trials.jsonl"
    trials = [json.loads(line) for line in path.read_text().splitlines()]
    trials = [row for row in trials if row["target_condition"] == "identical"]
    path.write_text("".join(json.dumps(row) + "\n" for row in trials))
    manifest["trials"]["sha256"] = sha256_file(path)
    manifest["n_trials"] = len(trials)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unbalanced conditions"):
        validate_dataset(output)


@pytest.mark.parametrize("correct_index", [0, 1])
def test_exp1_candidate_order_preserves_correct_answer(correct_index):
    row = _trial(
        "id",
        "group",
        "above_below",
        "different",
        "split_text",
        ["anchor.png", "same.png", "other.png"],
        correct_index,
    )
    assert row["candidates"][row["correct_index"]] == "same.png"


def test_audit_detects_repeated_images_with_reversed_candidate_order():
    images = [{"path": p, "sha256": p} for p in ("anchor", "same", "other")]
    trials = [
        _trial(
            str(i),
            "group",
            "above_below",
            "identical",
            "different_shapes",
            ["anchor", "same", "other"],
            i,
        )
        for i in (0, 1)
    ]
    audit = audit_trials(trials, images)
    assert audit["n_repeated_draws"] == 1
    assert audit["repeated_trial_ids"] == [["0", "1"]]
    assert audit["correct_index_counts"] == {"0": 1, "1": 1}


def _full_subset_fixture(root):
    from egolearn.evaluation.benchmarks.spatial_relations.prepare import (
        RELATIONS,
        RENDERERS,
        TARGET_CONDITIONS,
        _finish,
    )

    images = root / "exp1"
    images.mkdir(parents=True)
    rows = []
    for renderer in RENDERERS:
        for relation in RELATIONS:
            for condition in TARGET_CONDITIONS:
                for index in range(4):
                    trial_id = f"{renderer}/{relation}/{condition}/{index}"
                    paths = []
                    for role in ("anchor", "same", "other"):
                        relative = f"images/{trial_id}/{role}.png"
                        target = images / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (8, 8), (index, len(role), 10)).save(target)
                        paths.append(relative)
                    rows.append(
                        _trial(
                            trial_id,
                            f"exp1/{index}",
                            relation,
                            condition,
                            renderer,
                            paths,
                            index % 2,
                            configuration=index,
                        )
                    )
    _finish(images, "exp1", rows, {"kind": "test_fixture"})
    source = root / "source"
    _renders(source)
    _prepare(source, root / "exp3")


def test_monitoring_subset_preserves_parent_trials_and_shared_configs(tmp_path):
    from egolearn.evaluation.benchmarks.spatial_relations.subset import prepare_subset

    full, first, second = (tmp_path / name for name in ("full", "first", "second"))
    _full_subset_fixture(full)
    parent_hashes = {name: sha256_file(full / name / "manifest.json") for name in ("exp1", "exp3")}
    results = prepare_subset(full, first, exp1_per_condition=2, exp3_configurations=1)
    prepare_subset(full, second, exp1_per_condition=2, exp3_configurations=1)
    assert (results["exp1"]["n_trials"], results["exp1"]["n_images"]) == (24, 72)
    assert (results["exp3"]["n_trials"], results["exp3"]["n_images"]) == (16, 24)
    for name in ("exp1", "exp3"):
        assert sha256_file(full / name / "manifest.json") == parent_hashes[name]
        assert (first / name / "manifest.json").read_bytes() == (
            second / name / "manifest.json"
        ).read_bytes()
        assert (first / name / "trials.jsonl").read_bytes() == (
            second / name / "trials.jsonl"
        ).read_bytes()
        manifest = validate_dataset(first / name)
        assert manifest["source"]["parent_manifest_sha256"] == parent_hashes[name]
        parent = {
            r["id"]: r
            for r in map(json.loads, (full / name / "trials.jsonl").read_text().splitlines())
        }
        subset = list(map(json.loads, (first / name / "trials.jsonl").read_text().splitlines()))
        assert all(row == parent[row["id"]] for row in subset)
        for record in manifest["images"]:
            assert (first / name / record["path"]).read_bytes() == (
                full / name / record["path"]
            ).read_bytes()
        if name == "exp3":
            configs = {row["configuration"] for row in subset}
            assert len(configs) == 1
            assert subset == sorted(
                [row for row in parent.values() if row["configuration"] in configs],
                key=lambda r: r["id"],
            )
    with pytest.raises(FileExistsError):
        prepare_subset(full, first, exp1_per_condition=2, exp3_configurations=1)


def test_monitoring_selection_is_independent_of_parent_row_order():
    from egolearn.evaluation.benchmarks.spatial_relations.subset import select_trials

    trials = [
        {
            "id": str(i),
            "renderer": "split_text",
            "relation": "above_below",
            "target_condition": "identical",
            "correct_index": i % 2,
        }
        for i in range(100)
    ]
    selected = select_trials(trials, "exp1", exp1_per_condition=10)
    assert selected == select_trials(trials[::-1], "exp1", exp1_per_condition=10)
    assert Counter(row["correct_index"] for row in selected) == {0: 5, 1: 5}
    assert selected != select_trials(trials, "exp1", seed=34, exp1_per_condition=10)
    with pytest.raises(ValueError, match="positive and even"):
        select_trials(trials, "exp1", exp1_per_condition=3)
    with pytest.raises(ValueError, match="insufficient"):
        select_trials(trials, "exp1", exp1_per_condition=102)


def test_monitoring_rejects_unshared_configuration_coverage():
    from egolearn.evaluation.benchmarks.spatial_relations.subset import select_trials

    trials = [
        {
            "id": str(i),
            "relation": relation,
            "target_condition": "identical",
            "reference_object": "reference",
            "anchor_object": "object",
            "configuration": i,
        }
        for relation, indices in [("above_below", [0, 1]), ("between_outside", [0])]
        for i in indices
    ]
    with pytest.raises(ValueError, match="shared across"):
        select_trials(trials, "exp3", exp3_configurations=1)


def test_spatial_comparison_scores_labels_and_half_credit_ties():
    import numpy as np

    from egolearn.evaluation.benchmarks.spatial_relations.compare import score_trials

    records = [{"path": p} for p in ("anchor", "same", "other")]
    trials = [
        _trial(
            str(i),
            "group",
            "above_below",
            "identical",
            "different_shapes",
            ["anchor", "same", "other"],
            i,
        )
        for i in (0, 1)
    ]
    scores, margins = score_trials(np.array([[1, 0], [2, 0], [0, 1]]), records, trials)
    np.testing.assert_array_equal(scores, [1, 1])
    np.testing.assert_allclose(margins, [1, 1])
    scores, margins = score_trials(np.array([[1, 0], [0, 1], [0, -1]]), records, trials)
    np.testing.assert_array_equal(scores, [0.5, 0.5])
    scores, margins = score_trials(np.array([[1, 0], [0, 1], [2, 0]]), records, trials)
    np.testing.assert_array_equal(scores, [0, 0])
    with pytest.raises(ValueError, match="finite nonzero"):
        score_trials(np.zeros((3, 2)), records, trials)


def _aggregation_fixture():
    import numpy as np

    from egolearn.evaluation.benchmarks.spatial_relations.score import DOMAINS

    domains = {}
    for domain, spec in DOMAINS.items():
        rows, scores = [], []
        for renderer in spec["renderers"]:
            for relation in ("above_below", "between_outside"):
                for target in ("identical", "different"):
                    # Deliberately oversample easier cells: a pooled mean is wrong.
                    count = 9 if target == "identical" else 1
                    rows.extend(
                        [{"renderer": renderer, "relation": relation, "target_condition": target}]
                        * count
                    )
                    score = float(target == "identical") if domain == "abstract_2d" else 0.0
                    scores.extend([score] * count)
        domains[domain] = (rows, np.array(scores))
    return domains


def test_spatial_hierarchical_average_does_not_weight_images_trials_or_renderers():
    from egolearn.evaluation.benchmarks.spatial_relations.score import aggregate_scores

    domains = _aggregation_fixture()
    result = aggregate_scores(domains)
    assert result["abstract_2d/accuracy"] == 0.5
    assert result["rendered_3d/accuracy"] == 0
    assert result["accuracy"] == 0.25
    assert result["same_object/accuracy"] == 0.5
    assert result["changed_object/accuracy"] == 0
    assert result["abstract_2d/object_change_gap"] == 1
    rows, scores = domains["abstract_2d"]
    domains["abstract_2d"] = (rows * 3, scores.tolist() * 3)
    assert aggregate_scores(domains)["accuracy"] == 0.25


def test_spatial_aggregation_rejects_partial_results():
    from egolearn.evaluation.benchmarks.spatial_relations.score import aggregate_scores

    domains = _aggregation_fixture()
    with pytest.raises(ValueError, match="both"):
        aggregate_scores({"abstract_2d": domains["abstract_2d"]})
    rows, scores = domains["rendered_3d"]
    domains["rendered_3d"] = (rows[:-1], scores[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_scores(domains)


def test_spatial_probe_registry_shared_images_and_artifact(tmp_path):
    from types import SimpleNamespace

    import torch

    from egolearn.config.registry import PROBES
    from egolearn.evaluation.benchmarks.spatial_relations import SpatialRelationsProbe

    _full_subset_fixture(tmp_path)
    probe = PROBES.get("spatial_relations")(
        name="spatial_relations_monitoring",
        abstract_2d_manifest=str(tmp_path / "exp1/manifest.json"),
        rendered_3d_manifest=str(tmp_path / "exp3/manifest.json"),
        batch_size=31,
        output_json=str(tmp_path / "result_{step}.json"),
    )
    assert isinstance(probe, SpatialRelationsProbe)

    class Model:
        n_images = 0

        def encode(self, modality, inputs):
            assert modality == "image"
            assert inputs.shape[1:] == (3, 224, 224)
            assert not torch.is_grad_enabled()
            self.n_images += len(inputs)
            return SimpleNamespace(pooled=torch.ones(len(inputs), 3))

    model = Model()
    state = torch.random.get_rng_state().clone()
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    metrics = probe.run(model, SimpleNamespace(device="cpu", clock=SimpleNamespace(step=7)))
    assert torch.equal(state, torch.random.get_rng_state())
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    assert model.n_images == 144 + 48  # 3D images are shared across 32 trials.
    assert metrics["spatial_relations_monitoring/accuracy"] == 0.5
    artifact = json.loads((tmp_path / "result_7.json").read_text())
    assert artifact["aggregation"] == "equal_domains_equal_relation_target_renderer_cells_v1"
    assert artifact["domains"]["abstract_2d"]["label"] == "2D abstract scenes"
    assert artifact["domains"]["rendered_3d"]["tie_rate"] == 1
    assert "images" not in artifact["domains"]["rendered_3d"]["source"]
    assert all("exp1" not in key and "exp3" not in key for key in metrics)
    assert artifact["domains"]["abstract_2d"]["manifest_sha256"] == sha256_file(
        tmp_path / "exp1/manifest.json"
    )


def test_spatial_probe_rejects_swapped_domains_and_mixed_profiles(tmp_path):
    from egolearn.evaluation.benchmarks.spatial_relations import SpatialRelationsProbe
    from egolearn.evaluation.benchmarks.spatial_relations.subset import prepare_subset

    full, small = tmp_path / "full", tmp_path / "small"
    _full_subset_fixture(full)
    with pytest.raises(ValueError, match="wrong dataset"):
        SpatialRelationsProbe(
            "spatial", str(full / "exp3/manifest.json"), str(full / "exp1/manifest.json")
        )
    prepare_subset(full, small, seed=33, exp1_per_condition=2, exp3_configurations=1)
    with pytest.raises(ValueError, match="same full or monitoring"):
        SpatialRelationsProbe(
            "spatial", str(small / "exp1/manifest.json"), str(full / "exp3/manifest.json")
        )


def test_spatial_probe_restores_precision_settings_on_encoder_failure(tmp_path):
    from types import SimpleNamespace

    import torch

    from egolearn.evaluation.benchmarks.spatial_relations import SpatialRelationsProbe

    _full_subset_fixture(tmp_path)
    probe = SpatialRelationsProbe(
        "spatial", str(tmp_path / "exp1/manifest.json"), str(tmp_path / "exp3/manifest.json")
    )
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    class Model:
        def encode(self, *args):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        probe.run(Model(), SimpleNamespace(device="cpu"))
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_spatial_sampling_preserves_strata_and_configuration_blocks(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from egolearn.evaluation.benchmarks.spatial_relations.compare import sampling_variability

    full, output = tmp_path / "full", tmp_path / "scores"
    _full_subset_fixture(full)
    for experiment in ("exp1", "exp3"):
        rows = [
            json.loads(line)
            for line in (full / experiment / "trials.jsonl").read_text().splitlines()
        ]
        # Constant within each 2D stratum; exactly half correct in every 3D block.
        scores = np.array([float(row["target_condition"] == "identical") for row in rows])
        selected = np.array(
            [row["configuration"] < (2 if experiment == "exp1" else 1) for row in rows]
        )
        for model, values in (("balanced", scores), ("perfect", np.ones(len(rows)))):
            (output / model).mkdir(parents=True, exist_ok=True)
            np.savez(output / model / f"{experiment}.npz", scores=values, selected=selected)
    args = SimpleNamespace(full_root=full, output_dir=output)
    sampling_variability(args, {"balanced": {}, "perfect": {}}, repetitions=100)
    first = (output / "sampling_variability.json").read_bytes()
    result = json.loads(first)
    assert all(row["random_subset_error_95_range_pp"] == [0, 0] for row in result["models"])
    assert all(row["random_subset_order_agreement"] == 1 for row in result["model_pairs"])
    sampling_variability(args, {"balanced": {}, "perfect": {}}, repetitions=100)
    assert first == (output / "sampling_variability.json").read_bytes()


def test_hub_snapshot_symlinks_are_materialized_for_dataset_validation(tmp_path, monkeypatch):
    import huggingface_hub

    from egolearn.config.instantiate import instantiate

    source = tmp_path / "source"
    _renders(source)
    prepared = tmp_path / "prepared"
    _prepare(source, prepared)
    cache = tmp_path / "hub/datasets--org--spatial/snapshots" / ("a" * 40)
    for path in prepared.rglob("*"):
        if path.is_file():
            cached = cache / path.relative_to(prepared)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.symlink_to(path)
    # Reproduce the actual snapshot-cache incompatibility without weakening
    # the validator's protection against local symlink escapes.
    with pytest.raises(ValueError, match="nonlocal"):
        validate_dataset(cache)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(cache))
    config = {"manifest": "hf://datasets/org/spatial@v1/manifest.json"}
    resolved = instantiate(config)
    from pathlib import Path

    materialized = Path(resolved["manifest"]).parent
    assert validate_dataset(materialized)["n_trials"] == 32
    assert not any(p.is_symlink() for p in materialized.rglob("*"))
    assert instantiate(config) == resolved


def test_packaged_eval_config_uses_hosted_monitoring_subsets(tmp_path, monkeypatch):
    import sys

    import torch
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    import egolearn.utils.input_provenance as inputs
    from egolearn.evaluation import evaluate
    from egolearn.evaluation.benchmarks.spatial_relations import SpatialRelationsProbe
    from egolearn.evaluation.benchmarks.spatial_relations.subset import prepare_subset
    from egolearn.evaluation.evaluator import _configured_benchmark
    from egolearn.model.build import build_model

    _full_subset_fixture(tmp_path / "full")
    subset = tmp_path / "subset"
    prepare_subset(tmp_path / "full", subset, exp1_per_condition=2, exp3_configurations=1)
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = OmegaConf.to_container(compose(config_name="spatial_relations_eval"), resolve=True)
    assert isinstance(cfg, dict)
    spec = cfg["benchmarks"][0]
    prefix = "hf://datasets/agentic-learning-ai-lab/egolearn-spatial-relation-categorization@v1/"
    hosted = {
        prefix + "monitoring/2d-stimuli/manifest.json": subset / "exp1/manifest.json",
        prefix + "monitoring/3d-stimuli/manifest.json": subset / "exp3/manifest.json",
    }
    assert {spec["abstract_2d_manifest"], spec["rendered_3d_manifest"]} == set(hosted)
    monkeypatch.setattr(sys.modules["egolearn.config.instantiate"], "resolve", hosted.__getitem__)

    probe, config = _configured_benchmark(spec)

    assert isinstance(probe, SpatialRelationsProbe)
    assert probe.profile == "monitoring"
    assert config["abstract_2d_manifest"] == spec["abstract_2d_manifest"]

    calls = []

    def resolve(source, repo_type):
        assert repo_type == "dataset"
        calls.append(source)
        return hosted[source], {"reference": source, "commit": "b" * 40, "revision": "v1"}

    def unexpected_resolution(source):
        raise AssertionError("recording must resolve and pin the manifest before construction")

    monkeypatch.setattr(inputs, "resolve_with_metadata", resolve)
    monkeypatch.setattr(
        sys.modules["egolearn.config.instantiate"], "resolve", unexpected_resolution
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_model({"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 4}}})
    record_path = tmp_path / "record.json"
    result = evaluate(model, benchmarks=[spec], record_json=record_path)
    record = json.loads(record_path.read_text())
    assert record["status"] == "completed"
    assert sorted(calls) == sorted(hosted)
    for key in ("abstract_2d_manifest", "rendered_3d_manifest"):
        reference = spec[key]
        identity = record["input_provenance"][f"benchmarks.0.{key}"]
        assert identity["commit"] == "b" * 40
        assert identity["reference"] == reference
        assert identity["sha256"] == sha256_file(hosted[reference])
        assert record["config"]["benchmarks"][0][key] == str(hosted[reference])
        assert result.configs[spec["name"]][key] == reference
