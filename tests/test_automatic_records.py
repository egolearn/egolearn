"""Ordinary CLI/API runs retain independent, truthful records, including failures."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from PIL import Image

import egolearn
from egolearn.config.registry import BENCHMARKS
from egolearn.evaluation import Benchmark, EvaluationError, evaluate, evaluate_checkpoint
from egolearn.evaluation.linear_probe import ClassificationSplit, LabeledImage, LinearProbe
from egolearn.hub import sha256_file, write_checksums
from egolearn.run_record import record_run
from egolearn.training.hooks import CheckpointHook
from egolearn.utils.input_provenance import input_identity, prepare_inputs
from egolearn.utils.snapshot import load_snapshot, save_snapshot


def records(root):
    return [json.loads(path.read_text()) for path in sorted(root.rglob("record.json"))]


class Score(Benchmark):
    name = "score"

    def configuration(self):
        return {"seed": 17, "normalize": True}

    def metadata(self):
        return {"split_sha256": "a" * 64}

    def run(self, model, ctx):
        return {"accuracy": 0.75}


class Fails(Score):
    name = "fails"

    def run(self, model, ctx):
        raise RuntimeError("synthetic failure")


@pytest.fixture
def training_config(tmp_path, toy_config):
    def build(mode="streaming"):
        cfg = toy_config(tmp_path)
        if mode == "offline":
            for key in ("admission", "memories", "recipe", "routes", "label_stores"):
                cfg.pop(key)
            cfg["driver"] = {
                "mode": "offline",
                "max_steps": 2,
                "batch": {"slot": "curr", "n": 4, "views": ["v1", "v2"]},
            }
            cfg["losses"] = cfg["losses"][:1]
        return cfg

    return build


def test_default_records_preserve_each_attempt_and_explicit_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        with record_run({}, "train") as record:
            assert record is not None
            assert record.path == (
                Path("runs/default/records") / record.value["record_id"] / "record.json"
            )
            record.value["metrics"]["loss"] = 1.0
    assert len(records(tmp_path)) == 2
    explicit = tmp_path / "explicit.json"
    with record_run({"record_json": explicit}, "eval"):
        pass
    before = explicit.read_bytes()
    with pytest.raises(FileExistsError), record_run({"record_json": explicit}, "eval"):
        pass
    assert explicit.read_bytes() == before
    with record_run({"record_json": None, "run_dir": tmp_path}, "train") as disabled:
        assert disabled is None
    assert len(records(tmp_path)) == 2


def test_inputs_distinguish_files_unversioned_and_verified_collections(tmp_path):
    source = tmp_path / "labels.json"
    source.write_text("[]")
    config = {"stream": {"root_dir": str(tmp_path), "transcript_json": str(source)}}
    observed = prepare_inputs(config)
    assert observed["stream.transcript_json"]["sha256"] == sha256_file(source)
    assert observed["stream.root_dir"]["sha256"] == "unavailable"
    assert str(tmp_path) not in json.dumps(observed)
    write_checksums(tmp_path)
    _, identity = input_identity(tmp_path, verify_directory=True)
    assert identity["sha256"] == sha256_file(tmp_path / "checksums.sha256")
    source.write_text('["changed"]')
    _, identity = input_identity(tmp_path, verify_directory=True)
    assert identity["sha256"] == "unavailable"
    assert identity["reason"] == "checksum_verification_failed"


@pytest.mark.parametrize("verify", [False, True])
def test_invalid_inventory_does_not_abort_ordinary_records(tmp_path, verify):
    (tmp_path / "checksums.sha256").write_text(f"{'a' * 64}  ./image.png\n")
    _, identity = input_identity(tmp_path, verify_directory=verify)
    assert identity["sha256"] == "unavailable"
    assert identity["reason"] == "invalid_checksum_inventory"
    with record_run({"stream": {"root_dir": str(tmp_path)}, "run_dir": tmp_path}, "train") as run:
        assert run is not None
        assert run.value["input_provenance"]["stream.root_dir"] == identity


def test_automatic_directory_identity_does_not_read_dataset_bytes(tmp_path, monkeypatch):
    import egolearn.utils.input_provenance as inputs

    source = tmp_path / "data.txt"
    source.write_text("synthetic")
    write_checksums(tmp_path)
    real_hash = inputs.sha256_file

    def inventory_only(path):
        assert Path(path).name == "checksums.sha256"
        return real_hash(path)

    def must_not_verify(*args):
        pytest.fail("automatic capture must not re-read the dataset")

    monkeypatch.setattr(inputs, "sha256_file", inventory_only)
    monkeypatch.setattr(inputs, "verify_checksums", must_not_verify)
    identity = prepare_inputs({"stream": {"root_dir": str(tmp_path)}})["stream.root_dir"]
    assert identity["sha256"] == "unavailable"
    assert identity["inventory_sha256"] == real_hash(tmp_path / "checksums.sha256")
    assert identity["reason"] == "inventory_not_verified"


def test_dry_run_does_not_hash_or_resolve_inputs(tmp_path, monkeypatch):
    import egolearn.utils.input_provenance as inputs

    def must_not_read(*args, **kwargs):
        pytest.fail("dry-run provenance must not read local or remote inputs")

    monkeypatch.setattr(inputs, "input_identity", must_not_read)
    cfg = {
        "run_dir": tmp_path,
        "dry_run": True,
        "stream": {"root_dir": "hf://org/data@main"},
        "init_from": {"checkpoint": str(tmp_path / "model.pth")},
    }
    with record_run(cfg, "train") as run:
        assert run is not None
        assert run.config["stream"]["root_dir"] == cfg["stream"]["root_dir"]
        assert {item["reason"] for item in run.value["input_provenance"].values()} == {"dry_run"}
    assert records(tmp_path)[0]["status"] == "dry_run"


@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_pipeline_records_without_changing_training_or_rng(tmp_path, training_config, mode):
    cfg = training_config(mode)
    torch.manual_seed(12)
    plain = egolearn.build({**cfg, "record_json": None})
    plain.driver.fit()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(12)
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, tensor in plain.model.state_dict().items():
        assert torch.equal(pipe.model.state_dict()[key], tensor)
    [record] = records(tmp_path)
    assert record["status"] == "completed"
    assert record["metrics"]["train/steps"] == 2
    assert record["seed"] == cfg["seed"]
    assert record["provenance"]["uv_lock_sha256"] != "unavailable"
    assert not any(type(h).__name__ == "RunRecord" for h in pipe.driver.hooks)


def test_evaluation_failure_retains_completed_protocol_and_checkpoint(tmp_path, toy_config):
    pipe = egolearn.build(toy_config(tmp_path))
    checkpoint = tmp_path / "checkpoint.pth"
    CheckpointHook(str(tmp_path))._save(pipe.driver, checkpoint.name)
    record_path = tmp_path / "record.json"
    with pytest.raises(EvaluationError, match="fails") as error:
        evaluate_checkpoint(checkpoint, benchmarks=[Score(), Fails()], record_json=record_path)
    assert str(error.value.__cause__) == "synthetic failure"
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed"
    assert record["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert record["checkpoint"]["training_provenance"] == "unavailable"
    assert record["metrics"] == {"score/accuracy": 0.75}
    assert record["evaluation"]["metadata"]["score"]["split_sha256"] == "a" * 64
    assert record["evaluation"]["configs"]["score"]["seed"] == 17


def test_hub_reference_resolved_once_and_original_config_preserved(
    tmp_path, toy_config, monkeypatch
):
    import egolearn.utils.input_provenance as inputs

    pipe = egolearn.build(toy_config(tmp_path))
    checkpoint = tmp_path / "checkpoint.pth"
    CheckpointHook(str(tmp_path))._save(pipe.driver, checkpoint.name)
    calls = []

    def resolve(source, repo_type):
        calls.append(source)
        return checkpoint, {"reference": source, "commit": "b" * 40, "revision": "main"}

    monkeypatch.setattr(inputs, "resolve_with_metadata", resolve)
    result = evaluate_checkpoint(
        "hf://models/example/weights@main/model.pth",
        benchmarks=[Score()],
        record_json=tmp_path / "record.json",
    )
    assert len(calls) == 1
    assert result.checkpoint is not None
    assert result.checkpoint["commit"] == "b" * 40
    record = records(tmp_path)[0]
    assert record["input_provenance"]["checkpoint"]["commit"] == "b" * 40
    assert record["checkpoint"]["sha256"] == sha256_file(checkpoint)


def test_checkpoints_carry_training_identity_into_later_evaluation(tmp_path, toy_config):
    cfg = toy_config(tmp_path)
    cfg["hooks"]["checkpoint_every"] = 2
    original = deepcopy(cfg)
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    assert cfg == original
    checkpoint = tmp_path / "run/final.pth"
    result = evaluate_checkpoint(
        checkpoint, benchmarks=[Score()], record_json=tmp_path / "eval/record.json"
    )
    assert result.checkpoint is not None
    training = result.checkpoint["training_provenance"]
    assert training["seed"] == 7
    assert len(training["commit"]) == 40
    [record] = records(tmp_path / "run")
    assert record["outputs"]["final.pth"]["sha256"] == sha256_file(checkpoint)
    child_config = toy_config(tmp_path / "child")
    child_config["init_from"] = {"checkpoint": str(checkpoint)}
    child_config["hooks"]["checkpoint_every"] = 2
    child = egolearn.build(child_config)
    child.driver.fit()
    [child_record] = records(tmp_path / "child")
    parent = child_record["input_provenance"]["init_from.checkpoint"]["training_provenance"]
    assert parent["record_id"] == record["record_id"]
    assert parent["record_id"] != child_record["record_id"]
    child_result = evaluate_checkpoint(
        tmp_path / "child/run/final.pth",
        benchmarks=[Score()],
        record_json=tmp_path / "child-eval/record.json",
    )
    assert child_result.checkpoint is not None
    lineage = child_result.checkpoint["training_provenance"]["input_provenance"]
    assert (
        lineage["init_from.checkpoint"]["training_provenance"]["record_id"] == record["record_id"]
    )


def test_cli_dry_run_and_build_failure_write_automatic_records(tmp_path):
    command = [
        sys.executable,
        "-m",
        "egolearn.train",
        "--config-name",
        "toy_streaming",
        "dry_run=true",
        f"run_dir={tmp_path / 'run'}",
    ]
    good = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    bad = subprocess.run(
        [*command, "+driver.mode=invalid"], cwd=tmp_path, capture_output=True, text=True
    )
    assert bad.returncode != 0
    assert {record["status"] for record in records(tmp_path / "run")} == {"dry_run", "failed"}


@pytest.mark.parametrize("entry", ["api", "cli_hub"])
def test_resume_records_restored_seed_parent_and_new_attempt(
    tmp_path, toy_config, monkeypatch, entry
):
    cfg = toy_config(tmp_path)
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    snapshot = tmp_path / "saved.pth"
    save_snapshot(str(snapshot), pipe.driver)
    saved = torch.load(snapshot, weights_only=False)
    assert saved["run_provenance"]["seed"] == 7
    import egolearn.run_record as run_records

    current_identity = run_records.source_identity()
    monkeypatch.setattr(
        run_records, "source_identity", lambda: {**current_identity, "commit": "d" * 40}
    )
    cfg["seed"] = 99
    cfg["driver"]["max_steps"] = 3
    if entry == "api":
        resumed = egolearn.build(cfg)
        load_snapshot(str(snapshot), resumed.driver)
        resumed.driver.fit()
    else:
        import egolearn.utils.input_provenance as inputs
        from egolearn.train import _train

        reference = "hf://models/example/weights@main/snapshot.pth"
        calls = []

        def resolve(source, repo_type):
            calls.append(source)
            return snapshot, {"reference": source, "commit": "c" * 40, "revision": "main"}

        monkeypatch.setattr(inputs, "resolve_with_metadata", resolve)
        with record_run({**cfg, "resume": reference}, "train") as record:
            assert record is not None
            _train(dict(record.config))
        assert calls == [reference]
    results = sorted(records(tmp_path / "run"), key=lambda r: r["start_step"])
    assert len(results) == 2
    assert results[1]["start_step"] == 2
    assert results[1]["seed"] == 7
    assert results[1]["config"]["seed"] == 99
    assert results[1]["input_provenance"]["resume"]["sha256"] == sha256_file(snapshot)
    parent = results[1]["input_provenance"]["resume"]["training_provenance"]
    assert parent["record_id"] == results[0]["record_id"]
    assert parent["commit"] == results[0]["provenance"]["commit"]
    assert results[1]["provenance"]["commit"] == "d" * 40
    assert parent["commit"] != results[1]["provenance"]["commit"]
    assert results[1]["metrics"]["train/steps"] == 3
    if entry == "cli_hub":
        assert results[1]["input_provenance"]["resume"]["reference"] == reference
        assert results[1]["input_provenance"]["resume"]["commit"] == "c" * 40


@pytest.mark.parametrize("entry", ["model", "checkpoint"])
def test_evaluation_records_in_memory_split_without_changing_benchmark_arguments(
    tmp_path, toy_config, monkeypatch, entry
):
    monkeypatch.chdir(tmp_path)
    pipe = egolearn.build(toy_config(tmp_path))
    items = []
    for label, color in enumerate(("red", "blue")):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (24, 24), color).save(path)
        items.append(LabeledImage(str(path), label))
    split = ClassificationSplit(classes=("red", "blue"), train=tuple(items), test=tuple(items))
    spec = {
        "_target_": "probe:linear",
        "name": "linear",
        "encoder": "vision",
        "size": 16,
        "max_iter": 2,
        "split": split,
    }
    original = deepcopy(spec)
    run = LinearProbe.run

    def check_split(self, model, ctx):
        assert self.split is split
        return run(self, model, ctx)

    monkeypatch.setattr(LinearProbe, "run", check_split)
    if entry == "model":
        result = evaluate(pipe.model, benchmarks=[spec])
    else:
        checkpoint = tmp_path / "checkpoint.pth"
        CheckpointHook(str(tmp_path))._save(pipe.driver, checkpoint.name)
        result = evaluate_checkpoint(checkpoint, benchmarks=[spec])
    assert spec == original
    assert spec["split"] is split
    [record] = records(tmp_path / "runs/eval")
    assert record["status"] == "completed"
    assert record["evaluation"]["results"] == result.results
    if entry == "model":
        assert record["evaluation"]["checkpoint"] is None
    plain_split = json.loads(json.dumps(asdict(split)))
    assert record["config"]["benchmarks"][0]["split"] == plain_split
    assert record["evaluation"]["configs"]["linear"]["split"] == split.metadata()
    result.save_json(tmp_path / "evaluation.json")


def test_evaluation_preserves_nested_dataclasses_and_resolved_hub_inputs(
    tmp_path, toy_config, monkeypatch
):
    import egolearn.utils.input_provenance as inputs

    item = LabeledImage("synthetic.png", 0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    reference = "hf://datasets/org/data@main/manifest.json"
    calls = []

    def resolve(source, repo_type):
        calls.append((source, repo_type))
        return manifest, {"reference": source, "commit": "b" * 40}

    class ConfiguredScore(Score):
        def __init__(self, options):
            assert isinstance(options["examples"], tuple)
            assert options["examples"][0][0] is item
            assert options["manifest"] == str(manifest)

    monkeypatch.setattr(inputs, "resolve_with_metadata", resolve)
    monkeypatch.setitem(BENCHMARKS._items, "configured_score", ConfiguredScore)
    spec = {
        "_target_": "benchmark:configured_score",
        "options": {"examples": ([item],), "manifest": reference},
    }
    pipe = egolearn.build(toy_config(tmp_path))
    evaluate(pipe.model, benchmarks=[spec], record_json=tmp_path / "record.json")
    assert calls == [(reference, "dataset")]
    assert spec["options"] == {"examples": ([item],), "manifest": reference}
    [record] = records(tmp_path)
    options = record["config"]["benchmarks"][0]["options"]
    assert options == {"examples": [[asdict(item)]], "manifest": str(manifest)}
    assert record["input_provenance"]["benchmarks.0.options.manifest"]["commit"] == "b" * 40


@pytest.mark.parametrize("provenance", ["present", "missing", "null"])
def test_snapshot_load_replaces_provenance_before_saving_without_fit(
    tmp_path, training_config, monkeypatch, provenance
):
    import egolearn.run_record as run_records

    # Offline drivers support loading a snapshot after an earlier fit().
    cfg = training_config("offline")
    source = egolearn.build({**cfg, "run_dir": str(tmp_path / "source")})
    source.driver.fit()
    snapshot = tmp_path / "source.pth"
    save_snapshot(str(snapshot), source.driver)
    state = torch.load(snapshot, weights_only=False)
    if provenance == "missing":
        state.pop("run_provenance")
    elif provenance == "null":
        state["run_provenance"] = None
    torch.save(state, snapshot)

    identity = run_records.source_identity()
    monkeypatch.setattr(run_records, "source_identity", lambda: {**identity, "commit": "d" * 40})
    destination = egolearn.build({**cfg, "seed": 99, "run_dir": str(tmp_path / "destination")})
    destination.driver.fit()
    assert destination.driver.run_provenance != state.get("run_provenance")
    load_snapshot(str(snapshot), destination.driver)
    assert destination.driver.run_provenance == state.get("run_provenance")

    saved_again = tmp_path / "saved-again.pth"
    save_snapshot(str(saved_again), destination.driver)
    restored = torch.load(saved_again, weights_only=False)
    assert restored["run_provenance"] == state.get("run_provenance")
    for key, tensor in state["model"].items():
        assert torch.equal(restored["model"][key], tensor)


def test_preconstructed_benchmark_inputs_remain_descriptive(tmp_path, toy_config, monkeypatch):
    import egolearn.utils.input_provenance as inputs

    class CachedScore(Score):
        def configuration(self):
            return {"manifest": "hf://org/data@main/manifest.json"}

    def must_not_resolve(*args, **kwargs):
        pytest.fail("resolving the config cannot identify an instance's cached data")

    monkeypatch.setattr(inputs, "resolve_with_metadata", must_not_resolve)
    pipe = egolearn.build(toy_config(tmp_path))
    evaluate(pipe.model, benchmarks=[CachedScore()], record_json=tmp_path / "record.json")
    [record] = records(tmp_path)
    observed = record["input_provenance"]["benchmarks.0.manifest"]
    assert observed["sha256"] == "unavailable"
    assert observed["revision"] == "unavailable"
    assert observed["reason"] == "preconstructed_benchmark"


@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_ordinary_recording_preserves_nonfinite_skip_and_final_checkpoint(
    tmp_path, training_config, monkeypatch, mode
):
    cfg = training_config(mode)
    cfg["hooks"]["checkpoint_every"] = 2
    pipe = egolearn.build(cfg)

    def skip_once(compute):
        first = True

        def compute_with_skip(*args):
            nonlocal first
            if first:
                first = False
                return {"loss": torch.tensor(float("nan"))}
            return compute(*args)

        return compute_with_skip

    for loss in pipe.losses:
        monkeypatch.setattr(loss, "compute", skip_once(loss.compute))
    pipe.driver.fit()
    [record] = records(tmp_path)
    assert record["status"] == "nonfinite"
    assert record["nonfinite_steps"] == 1
    assert record["metrics"]["train/steps"] == 2
    assert record["outputs"]["final.pth"]["sha256"] == sha256_file(tmp_path / "run/final.pth")
    snapshot = tmp_path / "after-skip.pth"
    save_snapshot(str(snapshot), pipe.driver)
    cfg["driver"]["max_steps"] = 3
    resumed = egolearn.build(cfg)
    load_snapshot(str(snapshot), resumed.driver)
    assert resumed.driver.skipped_steps == 1
    assert resumed.driver.nonfinite_steps == 1
    resumed.driver.fit()
    [resumed_record] = [item for item in records(tmp_path) if item["start_step"] == 2]
    assert resumed_record["status"] == "nonfinite"
    assert resumed_record["nonfinite_steps"] == 1
    assert resumed_record["metrics"]["train/steps"] == 3


def test_snapshot_persists_link_to_still_running_parent_record(tmp_path, toy_config):
    cfg = toy_config(tmp_path)
    with record_run(cfg, "train") as record:
        assert record is not None
        pipe = egolearn.build(record.config)
        snapshot = tmp_path / "snapshot.pth"
        save_snapshot(str(snapshot), pipe.driver)
        assert record.path is not None
        persisted = json.loads(record.path.read_text())
        saved = torch.load(snapshot, weights_only=False)
        assert persisted["status"] == "running"
        assert persisted["outputs"][snapshot.name]["sha256"] == sha256_file(snapshot)
        assert saved["run_provenance"]["record_id"] == persisted["record_id"]


def test_babycl_suite_composes_default_training_and_protected_inputs(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from egolearn.commands.bench import _compose, _overrides
    from egolearn.evaluation.benchmarks.labeled_s_linear.probe import SPLIT_SHA256

    root = Path(__file__).resolve().parents[1]
    for name in (
        "SAYCAM_ROOT",
        "SAYCAM_TRANSCRIPTS",
        "BABYCL_VOCAB",
        "LABELED_S_MANIFEST",
        "LABELED_S_ROOT",
        "LABELED_S_SPLIT",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    for name in (
        "SAYCAM_SHA256",
        "SAYCAM_TRANSCRIPTS_SHA256",
        "BABYCL_VOCAB_SHA256",
        "LABELED_S_MANIFEST_SHA256",
        "LABELED_S_SHA256",
    ):
        monkeypatch.setenv(name, "a" * 64)
    suite = OmegaConf.to_container(
        OmegaConf.load(root / "benchmarks/babycl_saycam.yaml"), resolve=True
    )
    assert isinstance(suite, dict)
    suite["config"] = str((root / "benchmarks" / suite["config"]).resolve())
    result = _compose(suite, tmp_path / "run", 1337)
    with initialize_config_dir(
        config_dir=str(root / "projects/babycl/configs"), version_base="1.3"
    ):
        baseline = OmegaConf.to_container(compose(config_name="saycam_default"), resolve=True)
    assert isinstance(baseline, dict)
    for key in ("model", "losses", "solver", "driver", "views", "memories", "admission"):
        assert result[key] == baseline[key]
    assert suite["seeds"] == [1337, 1338, 1339]
    assert result["record_fail_on_nonfinite"] is True
    assert suite["inputs"]["probes.1.split_file"]["sha256"] == SPLIT_SHA256
    assert result["probes"][1]["data_root"] == str(tmp_path / "labeled_s_root")
    for key in suite["inputs"]:
        with pytest.raises(ValueError, match="pinned"):
            _overrides([f"{key}=changed"], set(suite["inputs"]))
