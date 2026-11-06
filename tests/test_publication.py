"""Publish synthetic evidence without uploading private runs or contacting the Hub."""

import copy
import hashlib
import json
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from egolearn.publication import hub
from egolearn.publication.prepare import import_evaluations, prepare
from egolearn.publication.schema import (
    DEFAULT_COLUMNS,
    build_table,
    digest,
    encoded,
    merge_runs,
    model_directory,
    read_json,
    run_id,
    validate_run,
)
from egolearn.utils.model_artifact import file_digest


def make_run(seed=1, expected=(1, 2, 3), value=None):
    protocol = {
        "version": "1",
        "settings": {"size": 224},
        "input_sha256": {"manifest": "a" * 64},
        "input_scope": "Manifest hashes only",
        "evaluation_seed": 0,
        "code_commit": None,
    }
    run = {
        "schema_version": 1,
        "id": "",
        "condition": {
            "id": "toy",
            "label": "Toy",
            "dataset": "Synthetic",
            "regime": "offline",
            "architecture": "ToyCNN",
            "description": "Synthetic tests only.",
        },
        "training_seed": seed,
        "expected_seeds": list(expected),
        "checkpoint": {
            "sha256": f"{seed:064x}",
            "weights_sha256": "b" * 64,
            "metadata_sha256": "c" * 64,
            "step": 12,
            "repo_id": f"test/model-{seed}",
            "revision": "a" * 40,
            "private": True,
        },
        "provenance": {
            "training_config_sha256": "d" * 64,
            "training_code_commit": None,
            "legacy_evaluation": False,
        },
        "evaluations": [
            {
                "column": DEFAULT_COLUMNS[0],
                "value": value if value is not None else seed / 10,
                "protocol": protocol,
                "protocol_sha256": digest(protocol),
                "source_sha256": "e" * 64,
            }
        ],
    }
    run["id"] = run_id(run)
    return run


def table(runs):
    return build_table(runs, results_repo="test/results", updated_at="2026-01-01T00:00:00Z")


def cell(runs):
    return table(runs)["rows"][0]["cells"]["labeled_s_4afc"]


def test_aggregation_requires_complete_matching_seeds_and_uses_sample_sd():
    runs = [make_run(seed) for seed in (1, 2, 3)]
    assert cell(runs)["mean"] == pytest.approx(0.2)
    assert cell(runs)["sd"] == pytest.approx(0.1)
    assert cell(runs[:2])["mean"] is None
    assert cell(runs[:2])["n"] == 2
    singleton = cell([make_run(1, (1,))])
    assert singleton["mean"] == 0.1 and singleton["sd"] is None
    runs[-1]["evaluations"][0]["protocol"]["settings"]["size"] = 128
    runs[-1]["evaluations"][0]["protocol_sha256"] = digest(runs[-1]["evaluations"][0]["protocol"])
    assert cell(runs)["status"] == "incompatible"
    assert cell(runs)["mean"] is None


def test_current_defaults_can_extend_schema_one_results():
    fixture = read_json(Path(__file__).parent / "fixtures" / "publication-v1.json")
    new_run = make_run(3)
    prototype = new_run["evaluations"][0]
    new_run["evaluations"] = [
        {**copy.deepcopy(prototype), "column": column} for column in DEFAULT_COLUMNS
    ]
    # This historical fixture also feeds the JS validator. Changing presentation
    # metadata needs an explicit migration, not an incompatible default edit.
    assert table(merge_runs(fixture["runs"], [new_run])) == fixture["table"]

    mixed = read_json(Path(__file__).parent / "fixtures" / "publication-v2.json")
    assert table(merge_runs(mixed["runs"][:1], mixed["runs"][1:])) == mixed["table"]


def test_reference_checkpoint_keeps_unknown_training_seed_without_inventing_uncertainty():
    reference = make_run(1, (1,))
    reference.update(schema_version=2, training_seed=None, expected_seeds=[None], not_applicable=[])
    reference["provenance"].update(reference=True, training_config_sha256=None)
    reference["id"] = run_id(reference)
    result = table([reference])
    assert result["schema_version"] == 2
    assert result["rows"][0]["expected_seeds"] == [None]
    assert result["rows"][0]["runs"][0]["seed"] is None
    assert cell([reference])["mean"] == 0.1 and cell([reference])["sd"] is None
    assert model_directory(reference) == "reference"
    for expected in ([None, None], [None, 1], [1]):
        with pytest.raises(ValueError, match="reference"):
            validate_run({**reference, "expected_seeds": expected})
    other = copy.deepcopy(reference)
    other["checkpoint"]["sha256"] = "f" * 64
    other["id"] = run_id(other)
    with pytest.raises(ValueError, match="same condition and training seed"):
        table([reference, other])
    with pytest.raises(ValueError, match="fields"):
        validate_run({**reference, "provenance": None})
    with pytest.raises(ValueError, match="boolean"):
        validate_run({**reference, "provenance": {**reference["provenance"], "reference": 1}})
    reference["provenance"]["reference"] = False
    with pytest.raises(ValueError, match="expected_seeds"):
        validate_run(reference)


def test_not_applicable_requires_consistent_reasons_and_cannot_hide_scores():
    run = make_run(1, (1,))
    run.update(
        schema_version=2,
        not_applicable=[
            {
                "column": DEFAULT_COLUMNS[1],
                "reason": "No trained image-text head.",
            }
        ],
    )
    run["provenance"]["reference"] = False
    result = table([run])["rows"][0]["cells"]
    assert result["labeled_s_4afc"]["status"] == "complete"
    assert result["babyview_4afc"] == {
        "mean": None,
        "sd": None,
        "n": 0,
        "expected_n": 1,
        "status": "not_applicable",
        "protocols": [],
        "reason": "No trained image-text head.",
    }
    bad = copy.deepcopy(run)
    bad["not_applicable"][0]["column"] = DEFAULT_COLUMNS[0]
    with pytest.raises(ValueError, match="evaluated and not applicable"):
        validate_run(bad)
    bad["not_applicable"][0]["column"] = {**DEFAULT_COLUMNS[0], "id": "alias"}
    with pytest.raises(ValueError, match="evaluated and not applicable"):
        validate_run(bad)
    bad = copy.deepcopy(run)
    bad["not_applicable"][0]["reason"] = " "
    with pytest.raises(ValueError, match="nonempty reason"):
        validate_run(bad)
    second = copy.deepcopy(run)
    run["expected_seeds"] = second["expected_seeds"] = [1, 2]
    second["training_seed"] = 2
    second["checkpoint"]["sha256"] = "f" * 64
    second["id"] = run_id(second)
    second["not_applicable"] = []
    with pytest.raises(ValueError, match="conflicting benchmark applicability"):
        table([run, second])


def test_duplicates_cannot_inflate_seed_counts_or_overwrite_results():
    run = make_run()
    assert merge_runs([run], [run]) == [run]
    newer_card = copy.deepcopy(run)
    newer_card["checkpoint"]["revision"] = "b" * 40
    assert merge_runs([run], [newer_card]) == [run]
    with pytest.raises(ValueError, match="more than one"):
        table([run, run])
    changed = copy.deepcopy(run)
    changed["evaluations"][0]["value"] = 0.9
    with pytest.raises(ValueError, match="immutable"):
        merge_runs([run], [changed])
    changed = copy.deepcopy(run)
    changed["checkpoint"]["sha256"] = "f" * 64
    changed["id"] = run_id(changed)
    with pytest.raises(ValueError, match="same condition and training seed"):
        table([run, changed])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -0.1, 1.01])
def test_invalid_metric_values_are_rejected(value):
    with pytest.raises(ValueError):
        validate_run(make_run(value=value))


def test_metadata_cannot_smuggle_paths_or_unrecognized_fields():
    run = make_run()
    run["local_directory"] = "/private/experiment"
    with pytest.raises(ValueError, match="fields"):
        validate_run(run)
    run = make_run()
    run["evaluations"][0]["protocol"]["settings"]["input"] = "/private/experiment"
    with pytest.raises(ValueError, match="path"):
        validate_run(run)


def test_malformed_publication_metadata_is_rejected_before_upload():
    run = make_run()
    run["evaluations"][0]["column"] = {**DEFAULT_COLUMNS[0], "label": ["invalid"]}
    with pytest.raises(ValueError, match="column fields"):
        validate_run(run)
    run = make_run()
    run["evaluations"][0]["protocol"]["input_scope"] = False
    with pytest.raises(ValueError, match="input scope"):
        validate_run(run)
    run = make_run()
    run["provenance"]["training_code_commit"] = 123
    with pytest.raises(ValueError, match="training code commit"):
        validate_run(run)


def evaluation(checkpoint_sha):
    return {
        "schema_version": 2,
        "status": "complete",
        "seed": 0,
        "checkpoint": {"sha256": checkpoint_sha, "step": 12},
        "results": {"labeled_s_4afc": {"four_afc_acc": 0.5}},
        "configs": {"labeled_s_4afc": {"size": 224, "manifest": "/private/trials.json"}},
        "versions": {"labeled_s_4afc": "1"},
        "metadata": {},
    }


def test_import_binds_to_checkpoint_and_requires_explicit_legacy_opt_in(tmp_path):
    path = tmp_path / "evaluation.json"
    checkpoint = make_run()["checkpoint"]
    source = evaluation(checkpoint["sha256"])
    path.write_bytes(encoded(source))
    kwargs = {
        "checkpoint": checkpoint,
        "columns": DEFAULT_COLUMNS,
        "input_hashes": [{"benchmark": "labeled_s_4afc", "kind": "manifest", "sha256": "a" * 64}],
    }
    results, legacy = import_evaluations([path], **kwargs)
    assert not legacy
    assert "manifest" not in results[0]["protocol"]["settings"]
    assert results[0]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="missing recorded input hashes"):
        import_evaluations([path], **{**kwargs, "input_hashes": []})
    source["status"] = "failed"
    path.write_bytes(encoded(source))
    with pytest.raises(ValueError, match="not complete"):
        import_evaluations([path], **kwargs)
    source.pop("status")
    source["schema_version"] = 1
    path.write_bytes(encoded(source))
    with pytest.raises(ValueError, match="not complete"):
        import_evaluations([path], **kwargs)
    assert import_evaluations([path], allow_legacy=True, **kwargs)[1]
    source["checkpoint"]["sha256"] = "f" * 64
    path.write_bytes(encoded(source))
    with pytest.raises(ValueError, match="does not match"):
        import_evaluations([path], allow_legacy=True, **kwargs)


@pytest.mark.parametrize(
    "key", ["abstract_2d_manifest", "rendered_3d_manifest", "trials_json", "metadata_csv"]
)
def test_custom_file_inputs_require_hashes_and_do_not_publish_paths(tmp_path, key):
    checkpoint = make_run()["checkpoint"]
    source = evaluation(checkpoint["sha256"])
    source["configs"]["labeled_s_4afc"] = {key: "private/trials.json", "image_root": "images"}
    source["metadata"] = {"labeled_s_4afc": {"count": 12}}
    path = tmp_path / "evaluation.json"
    path.write_bytes(encoded(source))
    kwargs = {"checkpoint": checkpoint, "columns": DEFAULT_COLUMNS}
    with pytest.raises(ValueError, match="missing recorded input hashes"):
        import_evaluations([path], input_hashes=[], **kwargs)
    results, _ = import_evaluations(
        [path],
        input_hashes=[{"benchmark": "labeled_s_4afc", "kind": key, "sha256": "a" * 64}],
        **kwargs,
    )
    assert results[0]["protocol"]["settings"] == {}
    assert results[0]["protocol"]["input_sha256"][key] == "a" * 64


@pytest.mark.parametrize("reference", [False, True])
def test_prepare_exports_and_strictly_reloads_synthetic_model(tmp_path, monkeypatch, reference):
    import torch

    from egolearn.model.build import build_model

    model = build_model({"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 4}}})
    checkpoint = tmp_path / "model.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.construction_config,
            "export_contract": model.export_contract(),
            "step": 12,
            "run_provenance": {
                "commit": "f" * 40,
                "dirty": False,
                **({} if reference else {"seed": 1}),
            },
        },
        checkpoint,
    )
    source = evaluation(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    result = tmp_path / "evaluation.json"
    result.write_bytes(encoded(source))
    config = tmp_path / "train.yaml"
    config.write_text("seed: 1\n")
    pins = tmp_path / "pins.json"
    pins.write_bytes(
        encoded(
            [
                {"benchmark": "labeled_s_4afc", "kind": "manifest", "sha256": "a" * 64},
            ]
        )
    )
    plan = {
        "schema_version": 1,
        "condition": make_run()["condition"],
        "expected_seeds": [1],
        "runs": [
            {
                "training_seed": 1,
                "checkpoint": str(checkpoint),
                "repo_id": "test/model",
                "training_config": str(config),
                "input_hashes": str(pins),
                "evaluations": [str(result)],
            }
        ],
    }
    if reference:
        plan.update(
            reference=True,
            expected_seeds=[None],
            not_applicable={
                "babyview_4afc": "Benchmark does not apply to this synthetic reference.",
            },
        )
        plan["runs"][0].update(training_seed=None, training_config=None)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(encoded(plan))
    destination = prepare(manifest, tmp_path / "publication")
    bundle = read_json(destination / "bundle.json")
    assert bundle["runs"][0]["checkpoint"]["sha256"] == source["checkpoint"]["sha256"]
    assert bundle["runs"][0]["provenance"]["training_code_commit"] == "f" * 40
    if reference:
        assert bundle["runs"][0]["provenance"]["training_config_sha256"] is None
        assert bundle["runs"][0]["training_seed"] is None
    assert str(tmp_path) not in (destination / "bundle.json").read_text()
    assert not list(destination.rglob("*.pth"))
    with pytest.raises(FileExistsError):
        prepare(manifest, destination)
    if reference:
        original = torch.load(checkpoint, weights_only=True)
        original["run_provenance"]["seed"] = 1
        torch.save(original, checkpoint)
        source["checkpoint"]["sha256"] = file_digest(checkpoint)
        result.write_bytes(encoded(source))
    else:
        config.write_text("seed: 2\n")
    with pytest.raises(ValueError, match="training seed"):
        prepare(manifest, tmp_path / "wrong-seed")
    assert not (tmp_path / "wrong-seed").exists()

    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="a" * 40, private=True)
    api.list_repo_files.return_value = []
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    model_upload = Mock(side_effect=AssertionError("must not upload weights"))
    monkeypatch.setattr(hub, "_model", model_upload)
    captured = []

    def publish_results(api, runs, **kwargs):
        for run in runs:
            validate_run(run, published=True)
            assert run["checkpoint"]["repo_id"] is None
            assert run["checkpoint"]["revision"] is None
        captured.extend(runs)
        return "a" * 40

    monkeypatch.setattr(hub, "publish_results", publish_results)
    receipt = hub.publish(
        destination, results_repo="test/results", collection="test/collection", results_only=True
    )
    assert len(captured) == 1
    assert receipt["models"] == [] and receipt["results_only"]
    api.add_collection_item.assert_called_once_with(
        "test/collection",
        "test/results",
        "dataset",
        exists_ok=True,
    )

    # A re-export after a code change must fail before any model upload.
    index = tmp_path / "index.json"
    index.write_bytes(encoded({"schema_version": 1, "runs": captured}))
    api.list_repo_files.return_value = ["index.json"]
    monkeypatch.setattr(hub, "_download", lambda *args: index)
    changed = tmp_path / "changed-export"
    copytree(destination, changed)
    model_dir = model_directory(bundle["runs"][0])
    metadata_path = changed / "models" / model_dir / "model.json"
    metadata = read_json(metadata_path)
    metadata["provenance"]["code_revision"] = {"commit": "e" * 40, "dirty": False}
    metadata_path.write_bytes(encoded(metadata))
    changed_bundle = read_json(changed / "bundle.json")
    changed_bundle["runs"][0]["checkpoint"]["metadata_sha256"] = file_digest(metadata_path)
    (changed / "bundle.json").write_bytes(encoded(changed_bundle))
    with pytest.raises(ValueError, match="Reuse the original prepared bundle"):
        hub.publish(changed, results_repo="test/results")
    model_upload.assert_not_called()

    # The retained bundle still attaches weights successfully.
    upload = Mock(return_value="b" * 40)
    commit = Mock(return_value="c" * 40)
    monkeypatch.setattr(hub, "_model", upload)
    monkeypatch.setattr(hub, "publish_results", commit)
    receipt = hub.publish(destination, results_repo="test/results")
    upload.assert_called_once()
    assert receipt["models"][0]["revision"] == "b" * 40
    assert commit.call_args.args[1][0]["checkpoint"]["metadata_sha256"] == file_digest(
        destination / "models" / model_dir / "model.json"
    )


@pytest.mark.parametrize(
    "case", ["new", "reuse", "incomplete", "metadata", "weights", "unmanaged", "public"]
)
def test_model_upload_only_reuses_verified_private_exports(tmp_path, monkeypatch, case):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "model.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"synthetic weights")
    run = make_run()
    run["checkpoint"]["metadata_sha256"] = file_digest(directory / "model.json")
    run["checkpoint"]["weights_sha256"] = file_digest(directory / "model.safetensors")
    metadata = tmp_path / "remote.json"
    metadata.write_text('{"changed": true}' if case == "metadata" else "{}")
    monkeypatch.setattr(hub, "_download", lambda *args: metadata)
    files = ["model.json", "model.safetensors"]
    if case == "new":
        files = [".gitattributes"]
    elif case == "incomplete":
        files = ["model.json"]
    elif case == "unmanaged":
        files.append("snapshot.pth")
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(
        sha="a" * 40,
        private=case != "public",
        siblings=[
            SimpleNamespace(
                rfilename="model.safetensors",
                lfs=SimpleNamespace(
                    sha256="f" * 64 if case == "weights" else run["checkpoint"]["weights_sha256"]
                ),
            )
        ],
    )
    api.list_repo_files.return_value = files
    api.create_commit.return_value = SimpleNamespace(oid="b" * 40)
    if case not in {"new", "reuse"}:
        with pytest.raises(ValueError):
            hub._model(api, run, directory)
        api.create_commit.assert_not_called()
    elif case == "reuse":
        assert hub._model(api, run, directory) == "a" * 40
        api.create_commit.assert_not_called()
    else:
        assert hub._model(api, run, directory) == "b" * 40
        kwargs = api.create_commit.call_args.kwargs
        assert kwargs["parent_commit"] == "a" * 40
        assert {op.path_in_repo for op in kwargs["operations"]} == {
            "model.json",
            "model.safetensors",
            "checksums.sha256",
            "README.md",
        }


@pytest.fixture
def remote(tmp_path, monkeypatch):
    state = SimpleNamespace(files={}, sha="a" * 40, private=True, commits=[], conflict=None)

    class API:
        def repo_info(self, repository, **kwargs):
            return SimpleNamespace(sha=state.sha, private=state.private)

        def list_repo_files(self, repository, **kwargs):
            assert kwargs["revision"] == state.sha
            return list(state.files)

        def create_commit(self, repository, **kwargs):
            assert kwargs["parent_commit"] == state.sha
            if state.conflict:
                from huggingface_hub.errors import HfHubHTTPError

                state.files["index.json"] = encoded({"schema_version": 1, "runs": [state.conflict]})
                state.conflict = None
                state.sha = "c" * 40
                raise HfHubHTTPError("conflict", response=Mock(status_code=409, headers={}))
            state.commits.append(kwargs)
            for operation in kwargs["operations"]:
                state.files[operation.path_in_repo] = operation.path_or_fileobj
            state.sha = f"{len(state.commits):040x}"
            return SimpleNamespace(oid=state.sha)

    def download(api, repository, kind, revision, filename):
        assert revision == state.sha
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(state.files[filename])
        return path

    monkeypatch.setattr(hub, "_download", download)
    return API(), state


def test_publication_is_atomic_idempotent_and_preserves_prior_runs(remote):
    api, state = remote
    runs = [make_run(seed) for seed in (1, 2, 3)]
    hub.publish_results(api, runs[:2], results_repo="test/results")
    partial = json.loads(state.files["benchmarks.json"])["rows"][0]["cells"]["labeled_s_4afc"]
    assert partial["mean"] is None
    revision = hub.publish_results(api, runs, results_repo="test/results")
    assert len(json.loads(state.files["index.json"])["runs"]) == 3
    assert hub.publish_results(api, runs, results_repo="test/results") == revision
    assert len(state.commits) == 2
    assert {op.path_in_repo for op in state.commits[-1]["operations"]} == {
        "index.json",
        "benchmarks.json",
        f"runs/{runs[2]['id']}.json",
    }


def test_concurrent_commit_retry_keeps_the_other_publishers_run(remote):
    api, state = remote
    state.conflict = make_run(1)
    hub.publish_results(api, [make_run(2), make_run(3)], results_repo="test/results")
    assert len(json.loads(state.files["index.json"])["runs"]) == 3
    result = json.loads(state.files["benchmarks.json"])["rows"][0]["cells"]["labeled_s_4afc"]
    assert result["mean"] == pytest.approx(0.2)


def test_private_public_visibility_mismatch_fails_without_mutation(remote):
    api, state = remote
    state.private = False
    with pytest.raises(ValueError, match="visibility"):
        hub.publish_results(api, [make_run()], results_repo="test/results")
    assert not state.commits


def test_results_only_records_can_gain_model_links_without_changing_evidence(remote):
    api, state = remote
    hosted = make_run(1, (1,))
    local = copy.deepcopy(hosted)
    for key in ("repo_id", "revision", "private"):
        local["checkpoint"][key] = None
    validate_run(local, published=True)
    assert cell([local])["mean"] == 0.1
    hub.publish_results(api, [local], results_repo="test/results")
    hub.publish_results(api, [hosted], results_repo="test/results")
    record = json.loads(state.files[f"runs/{hosted['id']}.json"])
    assert record == hosted
    revision = state.sha
    hub.publish_results(api, [local], results_repo="test/results")
    assert state.sha == revision
    assert len(state.commits) == 2
    invalid = copy.deepcopy(local)
    invalid["checkpoint"]["revision"] = "a" * 40
    with pytest.raises(ValueError, match="local checkpoints"):
        validate_run(invalid, published=True)
