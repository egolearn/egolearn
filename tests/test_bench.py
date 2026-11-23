"""Pin enforcement, frozen configurations, per-seed execution and array dispatch."""

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egolearn.commands import bench
from egolearn.commands.bench_table import summarize
from egolearn.hub import sha256_file, write_checksums


def suite_file(tmp_path, **changes):
    suite = {
        "version": 1,
        "name": "toy",
        "entry": "train",
        "config": "toy_streaming",
        "seeds": [7, 8],
        "overrides": ["++driver.max_steps=1"],
        "inputs": {},
        "metrics": {"train/last_step/loss": "min", "train/steps": "max"},
        **changes,
    }
    path = tmp_path / "suite.yaml"
    OmegaConf.save(OmegaConf.create(suite), path)
    return path


@pytest.mark.parametrize(
    "override",
    [
        "++seed=42",
        "run_dir=elsewhere",
        "~seed",
        "--multirun",
        "model={}",
        "model.weights=elsewhere",
        "model.weights.child=elsewhere",
    ],
)
def test_overrides_cannot_replace_managed_or_pinned_keys(override):
    with pytest.raises(ValueError):
        bench._overrides([override], {"model.weights"})


def test_pinned_file_and_directory_are_verified_before_outputs(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "data.txt").write_text("synthetic")
    checksums = write_checksums(payload)
    spec = suite_file(
        tmp_path,
        inputs={
            "stream.root_dir": {
                "path": "payload",
                "sha256": sha256_file(checksums),
            }
        },
    )
    suite = bench.load_suite(spec, [])
    assert suite["inputs"]["stream.root_dir"]["path"] == str(payload)
    (payload / "data.txt").write_text("changed")
    with pytest.raises(ValueError, match="altered"):
        bench.plan_suite(spec, tmp_path / "output", [])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"entry": "shell"},
        {"config": "--help"},
        {"entry": "eval", "config": "evaluate"},
    ],
)
def test_invalid_suite_fails_before_writing(tmp_path, changes):
    with pytest.raises(ValueError):
        bench.plan_suite(suite_file(tmp_path, **changes), tmp_path / "output", [])
    assert not (tmp_path / "output").exists()


def test_real_synthetic_training_suite_records_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    spec = suite_file(tmp_path)
    manifest = bench.plan_suite(spec, tmp_path / "output with spaces lr=5e-4", [])
    # Planning freezes the full composed configuration, not just the primary YAML.
    spec.write_text("not: the planned config\n")
    assert bench.run_suite(manifest, False, []) == 0
    result = summarize(manifest, allow_unverified=True)
    assert not result["problems"], result
    assert {row["metric"]: row["n"] for row in result["rows"]} == {
        "train/last_step/loss": 2,
        "train/steps": 2,
    }
    for seed in (7, 8):
        record = json.loads((manifest.parent / f"seed-{seed}/record.json").read_text())
        assert record["config"]["seed"] == seed
        assert record["metrics"]["train/steps"] == 1
        assert record["wall_seconds"] > 0
    with pytest.raises(FileExistsError):
        bench.run_one(manifest, 0, sha256_file(manifest))


def test_modified_config_fails_and_records_error(tmp_path):
    spec = suite_file(tmp_path, seeds=[7])
    manifest = bench.plan_suite(spec, tmp_path / "output", [])
    (manifest.parent / "seed-7/config.yaml").write_text("seed: 99\n")
    assert bench.run_suite(manifest, False, []) == 1
    record = json.loads((manifest.parent / "seed-7/record.json").read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "ValueError"


def test_array_uses_compute_python_and_preserves_argument_boundaries(tmp_path, monkeypatch):
    manifest = bench.plan_suite(suite_file(tmp_path), tmp_path / "output with spaces", [])
    monkeypatch.setattr("egolearn.commands.submit.launcher", lambda: tmp_path / "submit")
    calls = []

    def submit_call(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(subprocess, "call", submit_call)
    assert bench.run_suite(manifest, True, ["--gres=gpu:1"]) == 0
    commands = (manifest.parent / "commands.sh").read_text().splitlines()
    assert len(commands) == 2
    for index, command in enumerate(commands):
        args = shlex.split(command)
        assert args[:4] == ["python", "-m", "egolearn.commands.bench", "_run"]
        assert args[4:6] == [str(manifest), str(index)]
        assert str(Path(sys.executable)) not in command
    assert calls[0] == [
        "bash",
        str(tmp_path / "submit"),
        "--gres=gpu:1",
        "--no-requeue",
        "--commands",
        str(manifest.parent / "commands.sh"),
    ]


def test_pins_apply_before_interpolation_and_runtime_env_is_not_frozen(tmp_path, monkeypatch):
    config = tmp_path / "custom=1.yaml"
    config.write_text(
        "run_dir: old\nderived: ${run_dir}/child\nnode: ${oc.env:EGOLEARN_TEST_NODE}\n"
    )
    monkeypatch.setenv("EGOLEARN_TEST_NODE", "login")
    spec = suite_file(
        tmp_path,
        config=config.name,
        overrides=["++from_override=${oc.env:EGOLEARN_TEST_NODE}", "++derived_override=${run_dir}"],
        seeds=[1, 2],
    )
    manifest = bench.plan_suite(spec, tmp_path / "résultats with spaces", [])
    monkeypatch.setenv("EGOLEARN_TEST_NODE", "compute")
    for seed in (1, 2):
        frozen = manifest.parent / f"seed-{seed}/config.yaml"
        assert "login" not in frozen.read_text()
        cfg = OmegaConf.load(frozen)
        assert cfg.derived == str(frozen.parent / "artifacts/child")
        assert cfg.node == "compute"
        assert cfg.from_override == "compute"
        assert cfg.derived_override == cfg.run_dir
        assert cfg.seed == seed  # the original config did not declare seed


def test_duplicate_yaml_keys_and_sweeps_are_rejected(tmp_path):
    spec = suite_file(tmp_path)
    spec.write_text(spec.read_text() + "name: replacement\n")
    with pytest.raises(Exception, match=r"duplicate|Duplicate"):
        bench.load_suite(spec, [])
    with pytest.raises(ValueError, match="single values"):
        bench._overrides(["device=cpu,cuda"], set())


def test_cluster_visibility_and_readonly_paths_fail_before_writes(tmp_path, monkeypatch):
    from egolearn.commands.doctor import cluster_variables

    for names in cluster_variables().values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("APPTAINER_BIND", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    out = tmp_path / "outside" / "runs"
    with pytest.raises(ValueError, match="outside configured"):
        bench.check_cluster_paths(out, {})
    monkeypatch.setenv("APPTAINER_BIND", f"{out.parent}:{out.parent}:ro")
    with pytest.raises(ValueError, match="read-only"):
        bench.check_cluster_paths(out, {})
    assert not out.exists()


def test_cluster_planning_rejects_a_different_installed_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "egolearn.commands.submit.launcher", lambda: tmp_path / "scripts/slurm/submit"
    )
    monkeypatch.setattr(bench, "source_checkout", lambda: tmp_path / "other-checkout")
    with pytest.raises(ValueError, match="editable installation of this checkout"):
        bench.plan_suite(tmp_path / "missing.yaml", tmp_path / "out", [], submit=True)
    assert not (tmp_path / "out").exists()


def test_unignored_output_fails_before_it_dirties_the_running_source(tmp_path, monkeypatch):
    checkout = tmp_path.resolve() / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / ".gitignore").write_text("runs/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=checkout, check=True)
    monkeypatch.setattr(bench, "source_checkout", lambda: checkout)
    # The caller may be somewhere entirely different from the running source.
    monkeypatch.chdir(tmp_path)
    spec = suite_file(tmp_path, seeds=[7])
    with pytest.raises(ValueError, match="git-ignored"):
        bench.plan_suite(spec, checkout / "unignored", [])
    assert not (checkout / "unignored").exists()
    manifest = bench.plan_suite(spec, checkout / "runs/ignored", [])
    assert manifest.exists()


def test_cluster_paths_preserve_bound_scratch_symlinks(tmp_path, monkeypatch):
    from egolearn.commands.doctor import cluster_variables

    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "scratch"
    alias.symlink_to(physical, target_is_directory=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for names in cluster_variables().values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APPTAINER_BIND", str(alias))
    payload = alias / "input"
    payload.write_text("synthetic")
    spec = suite_file(
        alias,
        seeds=[7],
        inputs={"unused": {"path": "input", "sha256": sha256_file(payload)}},
    )
    monkeypatch.setattr(bench, "cluster_identity", bench.source_identity)
    manifest = bench.plan_suite(spec, alias / "out", [], submit=True)
    assert manifest.parent == alias / "out"
    assert json.loads(manifest.read_text())["inputs"]["unused"]["path"] == str(payload)
    assert bench.run_suite(manifest, False, []) == 0


def test_inputs_changed_during_execution_do_not_get_averaged(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.write_text("before")
    spec = suite_file(
        tmp_path,
        seeds=[7],
        inputs={
            "unused_pin": {
                "path": "payload",
                "sha256": sha256_file(payload),
            }
        },
    )
    manifest = bench.plan_suite(spec, tmp_path / "output", [])
    identity = bench.source_identity()
    monkeypatch.setattr(bench, "source_identity", lambda: identity)
    original = subprocess.run

    def changed(command, **kwargs):
        if "egolearn.cli" in command:
            payload.write_text("after")
            return subprocess.CompletedProcess(command, 0)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", changed)
    assert bench.run_suite(manifest, False, []) == 1
    record = json.loads((manifest.parent / "seed-7/record.json").read_text())
    assert record["status"] == "failed"


@pytest.mark.parametrize("artifact", ["legacy", "hub_export"])
def test_real_synthetic_evaluation_suite(tmp_path, monkeypatch, artifact):
    import torch
    from PIL import Image

    from egolearn.model.build import build_model

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    model = build_model({"encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 8}}})
    checkpoint = tmp_path / "model.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.construction_config,
            "export_contract": model.export_contract(),
        },
        checkpoint,
    )
    pin = {"path": "model.pth", "sha256": sha256_file(checkpoint)}
    downloads = []
    if artifact != "legacy":
        import huggingface_hub

        from egolearn.export import export_checkpoint

        exported = export_checkpoint(checkpoint, tmp_path / "export")
        digest = hashlib.sha256(
            (
                sha256_file(exported / "model.safetensors") + sha256_file(exported / "model.json")
            ).encode("ascii")
        ).hexdigest()
        pin = {"path": "export/model.safetensors", "sha256": digest}
        if artifact == "hub_export":
            cache = tmp_path / "hub/snapshots" / ("a" * 40)

            def download(*, repo_id, repo_type, revision, filename):
                assert (repo_id, repo_type) == ("org/model", "model")
                downloads.append((filename, revision))
                local = cache / filename
                local.parent.mkdir(parents=True, exist_ok=True)
                local.symlink_to(exported / Path(filename).name)
                return str(local)

            monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
            pin["path"] = "hf://models/org/model@v1/nested/model.safetensors"
    rows = []
    for label, color in enumerate(("red", "blue")):
        image = tmp_path / f"image-{label}.png"
        Image.new("RGB", (16, 16), color).save(image)
        rows.append({"image": str(image), "label": label})
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"classes": ["red", "blue"], "train": rows, "test": rows}))
    config = tmp_path / "suite-eval.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "defaults": ["/evaluate", "_self_"],
                "_imports_": ["egolearn.data.views", "egolearn.evaluation.linear_probe"],
                "benchmarks": [
                    {"_target_": "probe:linear", "name": "objects", "size": 16, "max_iter": 2}
                ],
            }
        ),
        config,
    )
    spec = suite_file(
        tmp_path,
        entry="eval",
        config="suite-eval.yaml",
        seeds=[0],
        overrides=[],
        inputs={
            "checkpoint": pin,
            "benchmarks.0.manifest": {"path": "split.json", "sha256": sha256_file(split)},
        },
        metrics={"objects/top1": "max"},
    )
    manifest = bench.plan_suite(spec, tmp_path / "output", [])
    assert bench.run_suite(manifest, False, []) == 0
    result = summarize(manifest, allow_unverified=True)
    assert not result["problems"], result
    assert 0 <= result["rows"][0]["mean"] <= 1
    record = json.loads((manifest.parent / "seed-0/record.json").read_text())
    assert record["checkpoint"]["sha256"] == pin["sha256"]
    if artifact == "hub_export":
        assert downloads == [("nested/model.safetensors", "v1"), ("nested/model.json", "a" * 40)]


@pytest.mark.parametrize(
    "hub,key,filename",
    [
        (False, "checkpoint", "model.json"),
        (True, "checkpoint", "model.safetensors"),
        (True, "init_from.checkpoint", "model.json"),
    ],
)
def test_export_pins_cover_both_files_before_and_after_planning(
    tmp_path, monkeypatch, key, filename, hub
):
    weights = tmp_path / "model.safetensors"
    metadata = tmp_path / "model.json"
    # Pin validation hashes bytes without loading model code or tensor frameworks.
    weights.write_bytes(b"synthetic weights")
    metadata.write_text('{"model": "synthetic metadata"}')
    digest = hashlib.sha256(
        (sha256_file(weights) + sha256_file(metadata)).encode("ascii")
    ).hexdigest()
    reference = weights.name
    downloads = []
    if hub:
        import huggingface_hub

        cache = tmp_path / "hub/snapshots" / ("a" * 40)

        def download(*, repo_id, repo_type, revision, filename):
            downloads.append((filename, revision))
            local = cache / filename
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                local.symlink_to(tmp_path / Path(filename).name)
            return str(local)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        reference = "hf://models/org/model@v1/nested/model.safetensors"
    spec = suite_file(tmp_path, inputs={key: {"path": reference, "sha256": digest}})
    planned = bench.load_suite(spec, [])
    if hub:
        assert downloads == [("nested/model.safetensors", "v1"), ("nested/model.json", "a" * 40)]
    (tmp_path / filename).write_text("changed after planning")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bench.verify_inputs(planned["inputs"])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bench.load_suite(spec, [])
