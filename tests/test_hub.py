"""Synthetic Hub collections: no network, images, or private data."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from egolearn import hub


@pytest.fixture
def remote(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    root = tmp_path / "remote"
    root.mkdir()
    (root / "data.txt").write_text("synthetic")
    hub.write_checksums(root)
    state = SimpleNamespace(private=True, exists=True, downloads=[], uploads=[], creates=[])

    class API:
        def repo_info(self, *, repo_id, repo_type, revision=None):
            if not state.exists:
                raise RepositoryNotFoundError("missing", response=Mock(headers={}, request=None))
            return SimpleNamespace(sha="a" * 40, private=state.private)

        def create_repo(self, **kwargs):
            state.creates.append(kwargs)
            state.exists = True
            state.private = kwargs["private"]

        def list_repo_files(self, **kwargs):
            return ["old[1].txt", ".gitattributes", "data.txt"]

        def create_commit(self, **kwargs):
            additions = [op for op in kwargs["operations"] if hasattr(op, "path_or_fileobj")]
            stage = Path(additions[0].path_or_fileobj).parent
            state.uploads.append(kwargs)
            state.card = (stage / hub.CARD).read_text()
            state.files = sorted(p.name for p in stage.iterdir())
            assert hub.verify_checksums(stage) == []
            assert kwargs["parent_commit"] == "a" * 40
            return SimpleNamespace(oid="b" * 40)

    def download(*, repo_id, repo_type, revision, local_dir=None):
        state.downloads.append((repo_id, repo_type, revision))
        if local_dir is not None:
            shutil.copytree(root, local_dir)
            return str(local_dir)
        cache = tmp_path / "snapshots" / ("a" * 40)
        shutil.copytree(root, cache, dirs_exist_ok=True)
        return str(cache)

    def download_file(*, repo_id, repo_type, revision, filename):
        return str(
            Path(download(repo_id=repo_id, repo_type=repo_type, revision=revision)) / filename
        )

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download_file)
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return root, state


def test_fetch_no_replace_preserves_local_collection(remote, tmp_path):
    _, state = remote
    target = tmp_path / "local" / "shape-bias"
    target.mkdir(parents=True)
    (target / "original").write_text("local data")
    with pytest.raises(FileExistsError, match="replace"):
        hub.fetch("shape-bias", target.parent, replace=False)
    assert not state.downloads
    assert (target / "original").read_text() == "local data"


def test_resolve_directory_file_and_commit_log(remote, caplog):
    _, state = remote
    with caplog.at_level("INFO"):
        path = hub.resolve("hf://org/weights@v1/data.txt", repo_type="model")
    assert path.read_text() == "synthetic"
    assert state.downloads == [("org/weights", "model", "v1")]
    assert "a" * 40 in caplog.text
    with pytest.raises(FileNotFoundError):
        hub.resolve("hf://org/data@v1/absent")


@pytest.mark.parametrize(
    "reference,repo_type",
    [
        ("hf://models/org/repo@v1", "dataset"),
        ("hf://org/repo@v1", "model"),
    ],
)
def test_model_reference_requires_filename_before_any_download(monkeypatch, reference, repo_type):
    import huggingface_hub

    def forbidden(**kwargs):
        raise AssertionError("a filename-less model reference attempted a download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    with pytest.raises(ValueError, match="model Hub reference requires an explicit filename"):
        hub.resolve(reference, repo_type=repo_type)


def test_explicit_dataset_directory_still_overrides_model_default(remote):
    path = hub.resolve("hf://datasets/org/data@v1", repo_type="model")
    assert (path / "data.txt").read_text() == "synthetic"


@pytest.mark.parametrize("path", ["hf://x/y", "hf://x/y@v/../bad"])
def test_bad_reference(path):
    with pytest.raises(ValueError):
        hub.resolve(path)


def test_config_and_tokenizer_paths(remote):
    pytest.importorskip("torch")
    from egolearn.config.instantiate import instantiate
    from egolearn.data.text import VocabTokenizer

    root, state = remote
    (root / "vocab.json").write_text(json.dumps({"hello": 1}))
    result = instantiate({"nested": ["hf://org/data@v1/data.txt"]})
    assert Path(result["nested"][0]).read_text() == "synthetic"
    assert VocabTokenizer.from_json("hf://models/org/weights@v1/vocab.json").vocab_size > 0
    assert state.downloads[-1][1] == "model"


def test_checksum_roundtrip_and_corruption(remote):
    root, _ = remote
    assert hub.verify_checksums(root) == []
    (root / "data.txt").write_text("corrupt")
    (root / "extra").write_text("extra")
    assert hub.verify_checksums(root) == ["altered: data.txt", "unlisted: extra"]


@pytest.mark.parametrize("entry", ["../escape", "/escape", "a\\b"])
def test_checksum_rejects_traversal(remote, entry):
    root, _ = remote
    (root / hub.CHECKSUMS).write_text(f"{'a' * 64}  {entry}\n")
    with pytest.raises(ValueError):
        hub.verify_checksums(root)


def test_checksum_rejects_symlinks_outside_and_duplicate(remote, tmp_path):
    root, _ = remote
    outside = tmp_path / "outside"
    outside.write_text("payload")
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        hub.write_checksums(root)
    (root / "escape").unlink()
    checksum = root / hub.CHECKSUMS
    checksum.write_text(checksum.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        hub.verify_checksums(root)


def test_fetch_replaces_and_pins_revision(remote, tmp_path):
    _, state = remote
    dest = tmp_path / "data"
    old = dest / "sample"
    old.mkdir(parents=True)
    (old / "old").touch()
    result = hub.fetch("sample", dest, revision="v1")
    assert not (result / "old").exists()
    assert state.downloads[0][2] == "a" * 40
    record = json.loads((result / hub.FETCH_RECORD).read_text())
    assert record["revision"] == "v1" and record["verified"] is True
    assert record["commit"] == "a" * 40


def test_fetch_failure_preserves_previous(remote, tmp_path):
    root, _ = remote
    dest = tmp_path / "data"
    previous = hub.fetch("sample", dest)
    (root / "data.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="verification"):
        hub.fetch("sample", dest)
    assert (previous / "data.txt").read_text() == "synthetic"
    assert list(dest.glob(".sample-*/")) == []


def test_fetch_missing_checksums_and_env(remote, tmp_path, monkeypatch):
    root, _ = remote
    (root / hub.CHECKSUMS).unlink()
    monkeypatch.setenv("EGOLEARN_DATA", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="allow-unverified"):
        hub.fetch("sample")
    target = hub.fetch("sample", allow_unverified=True, repo_type="model")
    assert json.loads((target / hub.FETCH_RECORD).read_text())["verified"] is False


def test_fetch_rename_failure_rolls_back(remote, tmp_path, monkeypatch):
    dest = tmp_path / "data"
    target = hub.fetch("sample", dest)
    rename = Path.rename

    def fail_stage(path, target):
        if path.name == "download":
            raise OSError("simulated rename failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(OSError, match="simulated"):
        hub.fetch("sample", dest)
    assert (target / "data.txt").read_text() == "synthetic"


def test_large_push_is_one_private_commit_with_staging_and_exact_deletions(remote):
    root, state = remote
    (root / ".env").write_text("synthetic hidden input")
    for index in range(300):
        (root / f"file{index}.txt").write_text(str(index))
    before = (root / hub.CHECKSUMS).read_bytes()
    state.exists = False
    assert hub.push(root, name="sample", license_id="MIT") == "b" * 40
    assert state.creates[0]["private"] is True
    assert ".env" not in state.files
    assert len(state.uploads) == 1
    assert len(state.uploads[0]["operations"]) == 304
    deleted = [
        op.path_in_repo
        for op in state.uploads[0]["operations"]
        if not hasattr(op, "path_or_fileobj")
    ]
    assert deleted == ["old[1].txt"]
    assert "license: mit" in state.card
    assert (root / hub.CHECKSUMS).read_bytes() == before


def test_push_requires_public_opt_in(remote):
    root, state = remote
    state.private = False
    with pytest.raises(ValueError, match="--public"):
        hub.push(root, name="sample", license_id="MIT")
    assert not state.uploads
    hub.push(root, name="sample", license_id="MIT", public=True)
    assert state.uploads[0]["repo_type"] == "dataset"


@pytest.mark.parametrize("filename", ["nested/snapshot.pt", "data.txt"])
def test_model_upload_rejects_non_export_payload_before_api_calls(tmp_path, monkeypatch, filename):
    import huggingface_hub

    payload = tmp_path / filename
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"synthetic payload")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid model payload reached the Hub API")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    with pytest.raises(ValueError, match="egl export"):
        hub.push(tmp_path, name="sample", license_id="MIT", repo_type="model")


@pytest.fixture
def model_export(tmp_path):
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    root = tmp_path / "export"
    root.mkdir()
    save_file({"weight": np.ones((2, 2), dtype=np.float32)}, root / "model.safetensors")
    document = {
        "format_version": 1,
        "model_config": {"model": {}},
        "export_contract": {},
        "weights_sha256": hub.sha256_file(root / "model.safetensors"),
        "provenance": {
            "filename": "final.pth",
            "sha256": "c" * 64,
            "step": 7,
            "code_revision": {"commit": "d" * 40, "dirty": False},
        },
    }
    (root / "model.json").write_text(json.dumps(document))
    return root


def test_model_upload_uses_validated_export_and_keeps_private_default(remote, model_export):
    _, state = remote
    state.exists = False
    hub.push(model_export, name="sample", license_id="MIT", repo_type="model")
    assert state.creates[0]["private"] is True
    assert state.uploads[0]["repo_type"] == "model"
    assert state.files == ["README.md", "checksums.sha256", "model.json", "model.safetensors"]


def test_model_export_validation_does_not_import_torch(model_export):
    script = """
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

class NoTorch(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('model upload validation imported torch')

sys.meta_path.insert(0, NoTorch())
from egolearn.utils.model_artifact import validate_export
validate_export(Path(sys.argv[1]), publishing=True)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(model_export)], check=True)


@pytest.mark.parametrize(
    "corruption",
    [
        "extra_pickle",
        "invalid_header",
        "invalid_json",
        "nonfinite",
        "overflow",
    ],
)
def test_model_upload_validates_export_before_creating_repo(remote, model_export, corruption):
    _, state = remote
    if corruption == "extra_pickle":
        (model_export / "snapshot.pth").write_bytes(b"not exported")
    elif corruption == "invalid_header":
        (model_export / "model.safetensors").write_bytes(b"not safetensors")
        path = model_export / "model.json"
        document = json.loads(path.read_text())
        document["weights_sha256"] = hub.sha256_file(model_export / "model.safetensors")
        path.write_text(json.dumps(document))
    elif corruption == "nonfinite":
        path = model_export / "model.json"
        document = json.loads(path.read_text())
        document["model_config"]["invalid"] = float("nan")
        path.write_text(json.dumps(document))
    elif corruption == "overflow":
        path = model_export / "model.json"
        document = json.loads(path.read_text())
        document["provenance"]["extra"] = "overflowed-number"
        path.write_text(json.dumps(document).replace('"overflowed-number"', "1e999"))
    else:
        (model_export / "model.json").write_text("{}")
    with pytest.raises(ValueError):
        hub.push(model_export, name="sample", license_id="MIT", repo_type="model")
    assert not state.creates and not state.uploads


@pytest.mark.parametrize("name", ["../escape", "-name"])
def test_fetch_bad_name(remote, name):
    with pytest.raises(ValueError):
        hub.fetch(name)


def test_cli(remote, tmp_path):
    root, _ = remote
    assert hub.main(["verify", str(root)]) == 0
    assert hub.main(["fetch", "sample", "--dest", str(tmp_path / "data")]) == 0
    assert hub.main(["push", str(root), "--name", "sample", "--license", "MIT"]) == 0


def test_model_file_download_does_not_fetch_other_checkpoints(remote, monkeypatch):
    import huggingface_hub

    def forbidden(**kwargs):
        raise AssertionError("model file reference downloaded the whole snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    path, metadata = hub.resolve_with_metadata("hf://models/org/weights@v1/data.txt")
    assert path.read_text() == "synthetic"
    assert metadata == {
        "reference": "hf://models/org/weights@v1/data.txt",
        "repo": "org/weights",
        "repo_type": "model",
        "revision": "v1",
        "commit": "a" * 40,
    }


def test_unknown_license_fails_before_creating_repository(remote):
    root, state = remote
    state.exists = False
    with pytest.raises(ValueError, match="unsupported license"):
        hub.push(root, name="sample", license_id="LicenseRef-Foo")
    assert not state.creates and not state.uploads


def test_existing_materialization_can_be_read_without_cache_writes(tmp_path, monkeypatch):
    cache = tmp_path / "snapshots" / ("a" * 40)
    cache.mkdir(parents=True)
    blob = tmp_path / "blob"
    blob.write_text("synthetic")
    (cache / "data.txt").symlink_to(blob)
    target = hub._materialize(cache)

    def no_write(*args, **kwargs):
        raise PermissionError("shared cache is read-only")

    monkeypatch.setattr(Path, "mkdir", no_write)
    monkeypatch.setattr(Path, "open", no_write)
    assert hub._materialize(cache) == target


def test_repaired_partial_snapshot_gets_a_complete_materialization(tmp_path):
    cache = tmp_path / "snapshots" / ("a" * 40)
    cache.mkdir(parents=True)
    blob = tmp_path / "blob"
    blob.write_text("synthetic")
    (cache / "first.txt").symlink_to(blob)
    partial = hub._materialize(cache)
    (cache / "second.txt").symlink_to(blob)
    complete = hub._materialize(cache)
    assert partial != complete
    assert not (partial / "second.txt").exists()
    assert (complete / "second.txt").read_text() == "synthetic"
