"""Portable, pickle-free model export and reconstruction using synthetic weights."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from pickle import UnpicklingError

import pytest
import torch

from egolearn.evaluation import Benchmark, evaluate_checkpoint, load_model
from egolearn.export import export_checkpoint, main
from egolearn.model.build import build_model
from egolearn.model.init import load_pretrained_weights
from egolearn.utils.checkpoint import read_checkpoint
from egolearn.utils.snapshot import save_snapshot


class WeightBenchmark(Benchmark):
    name = "weights"

    def run(self, model, ctx):
        return {"sum": sum(float(t.sum()) for t in model.state_dict().values())}


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    model = build_model(
        {
            "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 4}},
            "heads": {
                "proj": {"encoder": "vision", "_target_": "head:linear", "in_dim": 4, "out_dim": 2}
            },
        }
    )
    assert model.construction_config is not None
    model.construction_config["init_from"] = {
        "checkpoint": str(tmp_path / "private-source" / "previous.pth"),
        "sha256": "d" * 64,
        "step": 5,
    }
    state = {
        "model": model.state_dict(),
        "model_config": model.construction_config,
        "export_contract": model.export_contract(),
        "step": 17,
        "optimizer": {"must_not_publish": torch.zeros(1)},
        "run_dir": str(tmp_path / "private-run"),
    }
    path = tmp_path / "checkpoint.pth"
    torch.save(state, path)
    monkeypatch.setattr(
        "egolearn.utils.provenance.code_revision",
        lambda: {"commit": "c" * 40, "dirty": False},
    )
    return path, model


def test_export_load_and_initialize_without_pickle(checkpoint, tmp_path, monkeypatch):
    source, model = checkpoint
    exported = export_checkpoint(source, tmp_path / "export")
    assert sorted(p.name for p in exported.iterdir()) == ["model.json", "model.safetensors"]
    document = json.loads((exported / "model.json").read_text())
    assert document["provenance"] == {
        "filename": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "step": 17,
        "code_revision": {"commit": "c" * 40, "dirty": False},
        "training_provenance": "unavailable",
    }
    assert document["model_config"] == {
        **model.construction_config,
        "init_from": {"checkpoint": "previous.pth", "sha256": "d" * 64, "step": 5},
    }
    assert document["export_contract"] == model.export_contract()
    assert str(tmp_path) not in (exported / "model.json").read_text()
    assert "optimizer" not in document and "run_dir" not in document

    def forbidden(*args, **kwargs):
        raise AssertionError("exported weights were read with pickle")

    monkeypatch.setattr(torch, "load", forbidden)
    loaded = load_model(exported)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name
    with torch.no_grad():
        for parameter in loaded.parameters():
            parameter.zero_()
    summary = load_pretrained_weights(loaded, str(exported / "model.safetensors"))
    assert summary["checkpoint"] == "model.safetensors" and summary["step"] == 17
    assert summary["source"] == document["provenance"]
    assert all(torch.equal(t, loaded.state_dict()[n]) for n, t in model.state_dict().items())
    record_path = tmp_path / "record.json"
    result = evaluate_checkpoint(exported, benchmarks=[WeightBenchmark()], record_json=record_path)
    assert result.checkpoint is not None
    assert result.checkpoint["sha256"] == summary["sha256"]
    assert result.checkpoint["source"] == document["provenance"]
    record = json.loads(record_path.read_text())
    assert record["input_provenance"]["checkpoint"]["sha256"] == summary["sha256"]


def test_exported_metadata_changes_artifact_identity(checkpoint, tmp_path):
    source, _ = checkpoint
    exported = export_checkpoint(source, tmp_path / "export")
    _, before = read_checkpoint(exported)
    path = exported / "model.json"
    document = json.loads(path.read_text())
    document["provenance"]["step"] = 18
    path.write_text(json.dumps(document))
    _, after = read_checkpoint(exported)
    assert before["sha256"] != after["sha256"]
    assert before["weights_sha256"] == after["weights_sha256"]


def test_export_handles_shared_storage_and_noncontiguous_views(checkpoint, tmp_path):
    source, _ = checkpoint
    state = torch.load(source, weights_only=True)
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    state["model"] = {"shared": tensor, "alias": tensor, "view": tensor.T}
    torch.save(state, source)
    exported = export_checkpoint(source, tmp_path / "export")
    restored, _ = read_checkpoint(exported)
    assert restored["model"].keys() == state["model"].keys()
    assert all(torch.equal(t, restored["model"][n]) for n, t in state["model"].items())


def test_export_snapshot_requires_explicit_trust(toy_config, tmp_path):
    import egolearn

    pipe = egolearn.build(toy_config(tmp_path))
    pipe.driver.fit()
    pipe.ctx.clock.step = 23
    snapshot = tmp_path / "snapshot.pth"
    save_snapshot(str(snapshot), pipe.driver)
    output = tmp_path / "export"
    with pytest.raises(UnpicklingError):
        export_checkpoint(snapshot, output)
    assert not output.exists()
    exported = export_checkpoint(snapshot, output, trusted_snapshot=True)
    state, metadata = read_checkpoint(exported)
    assert metadata["step"] == 23
    assert set(state) == {"model", "model_config", "export_contract", "step", "run_provenance"}
    assert state["run_provenance"] == pipe.driver.run_provenance
    assert metadata["training_provenance"] == pipe.driver.run_provenance
    loaded = load_model(exported)
    assert all(torch.equal(t, loaded.state_dict()[n]) for n, t in pipe.model.state_dict().items())


def test_export_refuses_existing_destination_without_modifying_it(checkpoint, tmp_path):
    source, _ = checkpoint
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        export_checkpoint(source, output)
    assert sorted(p.name for p in output.iterdir()) == [marker.name]
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("missing", ["model_config", "export_contract"])
def test_export_rejects_missing_metadata_without_partial_output(checkpoint, tmp_path, missing):
    source, _ = checkpoint
    state = torch.load(source, weights_only=True)
    state.pop(missing)
    torch.save(state, source)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="export requires"):
        export_checkpoint(source, output)
    assert not output.exists()


def test_failed_export_leaves_no_partial_destination(checkpoint, tmp_path, monkeypatch):
    source, _ = checkpoint

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr("safetensors.torch.save_file", fail)
    with pytest.raises(OSError, match="simulated"):
        export_checkpoint(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()
    assert not list(tmp_path.glob(".egolearn-export-*"))


def test_corrupt_export_never_falls_back_to_pickle(checkpoint, tmp_path, monkeypatch):
    source, _ = checkpoint
    exported = export_checkpoint(source, tmp_path / "export")
    (exported / "model.safetensors").write_bytes(source.read_bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid export used pickle fallback")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA-256"):
        read_checkpoint(exported, trusted_snapshot=True)


def test_tensor_read_errors_have_checkpoint_context(checkpoint, tmp_path, monkeypatch):
    from safetensors import SafetensorError

    source, _ = checkpoint
    exported = export_checkpoint(source, tmp_path / "export")

    def fail(*args, **kwargs):
        raise SafetensorError("simulated tensor read failure")

    monkeypatch.setattr("safetensors.torch.load_file", fail)
    with pytest.raises(ValueError, match=r"invalid safetensors weights.*simulated"):
        read_checkpoint(exported)


@pytest.mark.parametrize(
    "error",
    ["existing", "metadata", "snapshot", "empty", "malformed", "malformed_trusted", "blocked"],
)
def test_export_cli_reports_expected_errors_without_tracebacks(checkpoint, tmp_path, capsys, error):
    source, _ = checkpoint
    output = tmp_path / "export"
    if error == "existing":
        output.mkdir()
        match = "already exists"
    elif error == "empty":
        source.write_bytes(b"")
        match = "checkpoint is empty or truncated"
    elif error == "blocked":
        # A harmless global from a module torch blocks even in legacy pickle input.
        source.write_bytes(b"cos\ngetcwd\n)R.")
        match = "module os is blocked"
    elif error in {"malformed", "malformed_trusted"}:
        source.write_bytes(b"not a checkpoint")
        match = "invalid load key" if error == "malformed_trusted" else "Unsupported operand"
    else:
        state = torch.load(source, weights_only=True)
        if error == "metadata":
            state.pop("model_config")
            match = "construction metadata"
        else:
            state["snapshot_only"] = Path("synthetic-state")
            match = "--trusted-snapshot"
        torch.save(state, source)
    with pytest.raises(SystemExit) as caught:
        options = ["--trusted-snapshot"] if error == "malformed_trusted" else []
        main([str(source), str(output), *options])
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert match in captured.err and "Traceback" not in captured.err
    assert "will likely succeed" not in captured.err
    assert not captured.out


@pytest.mark.parametrize("entry_point", ["load", "evaluate", "initialize"])
def test_hub_export_load_pins_metadata_to_weights_commit(
    checkpoint, tmp_path, monkeypatch, entry_point
):
    import huggingface_hub

    source, model = checkpoint
    exported = export_checkpoint(source, tmp_path / "export")
    cache = tmp_path / "hub" / "snapshots" / ("a" * 40)
    cache.mkdir(parents=True)
    calls = []

    def download(*, repo_id, repo_type, revision, filename):
        assert (repo_id, repo_type) == ("org/model", "model")
        calls.append((filename, revision))
        path = cache / filename
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.symlink_to(exported / Path(filename).name)
        return str(path)

    def forbidden(**kwargs):
        raise AssertionError("model export used snapshot_download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    reference = "hf://models/org/model@v1/nested/model.safetensors"
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    if entry_point == "evaluate":
        result = evaluate_checkpoint(reference, benchmarks=[WeightBenchmark()])
        assert result.results["weights"]["sum"] == sum(float(t.sum()) for t in expected.values())
        metadata = result.checkpoint
    elif entry_point == "initialize":
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        metadata = load_pretrained_weights(model, reference)
        assert all(torch.equal(t, model.state_dict()[n]) for n, t in expected.items())
    else:
        loaded = load_model(reference)
        assert all(torch.equal(t, loaded.state_dict()[n]) for n, t in expected.items())
        assert calls == [("nested/model.safetensors", "v1"), ("nested/model.json", "a" * 40)]
        calls.clear()
        _, metadata = read_checkpoint(reference)
    assert calls == [("nested/model.safetensors", "v1"), ("nested/model.json", "a" * 40)]
    assert metadata is not None
    assert metadata["reference"] == reference and metadata["commit"] == "a" * 40


def test_export_cli_runs_outside_repository(checkpoint, tmp_path):
    source, model = checkpoint
    output = tmp_path / "published model"
    command = str(Path(sys.executable).parent / "egl")
    result = subprocess.run(
        [command, "export", str(source), str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    loaded = load_model(output)
    assert all(torch.equal(t, loaded.state_dict()[n]) for n, t in model.state_dict().items())
