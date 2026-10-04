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


def test_torch_free_and_local_passthrough():
    subprocess.run(
        [sys.executable, "-c", "import egolearn.hub,sys; assert 'torch' not in sys.modules"],
        check=True,
    )
    assert hub.resolve("relative/file") == Path("relative/file")


def test_resolve_directory_file_and_commit_log(remote, caplog):
    _, state = remote
    with caplog.at_level("INFO"):
        path = hub.resolve("hf://org/weights@v1/data.txt", repo_type="model")
    assert path.read_text() == "synthetic"
    assert state.downloads == [("org/weights", "model", "v1")]
    assert "a" * 40 in caplog.text
    with pytest.raises(FileNotFoundError):
        hub.resolve("hf://org/data@v1/absent")


@pytest.mark.parametrize("path", ["hf://x", "hf://x/y", "hf://x/y@", "hf://x/y@v/../bad"])
def test_bad_reference(path):
    with pytest.raises(ValueError):
        hub.resolve(path)


def test_config_and_tokenizer_paths(remote):
    pytest.importorskip("torch")
    from egolearn.config.instantiate import instantiate
    from egolearn.data.text import VocabTokenizer

    root, _ = remote
    (root / "vocab.json").write_text(json.dumps({"hello": 1}))
    result = instantiate({"nested": ["hf://org/data@v1/data.txt"]})
    assert Path(result["nested"][0]).read_text() == "synthetic"
    assert VocabTokenizer.from_json("hf://org/data@v1/vocab.json").vocab_size > 0


def test_checksum_roundtrip_and_corruption(remote):
    root, _ = remote
    assert hub.verify_checksums(root) == []
    (root / "data.txt").write_text("corrupt")
    (root / "extra").write_text("extra")
    assert hub.verify_checksums(root) == ["altered: data.txt", "unlisted: extra"]


@pytest.mark.parametrize("entry", ["../escape", "/escape", "a/../escape", "a\\b", "./data.txt"])
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


def test_push_private_defaults_staging_and_stale_files(remote):
    root, state = remote
    (root / ".env").write_text("synthetic hidden input")
    before = (root / hub.CHECKSUMS).read_bytes()
    state.exists = False
    assert hub.push(root, name="sample", license_id="MIT") == "b" * 40
    assert state.creates[0]["private"] is True
    assert ".env" not in state.files
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
    hub.push(root, name="sample", license_id="MIT", public=True, repo_type="model")
    assert state.uploads[0]["repo_type"] == "model"


@pytest.mark.parametrize("name", ["../escape", ".", "bad/name", "-name"])
def test_fetch_bad_name(remote, name):
    with pytest.raises(ValueError):
        hub.fetch(name)


def test_cli(remote, tmp_path):
    root, _ = remote
    assert hub.main(["verify", str(root)]) == 0
    assert hub.main(["fetch", "sample", "--dest", str(tmp_path / "data")]) == 0
    assert hub.main(["push", str(root), "--name", "sample", "--license", "MIT"]) == 0


def test_large_push_is_one_commit_with_exact_deletions(remote):
    root, state = remote
    for index in range(300):
        (root / f"file{index}.txt").write_text(str(index))
    hub.push(root, name="many", license_id="MIT")
    assert len(state.uploads) == 1
    operations = state.uploads[0]["operations"]
    assert len(operations) == 304  # 301 payload files, card, checksums, one deletion
    assert [op.path_in_repo for op in operations if not hasattr(op, "path_or_fileobj")] == [
        "old[1].txt"
    ]


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


def test_model_vocabulary_can_override_dataset_default(remote):
    pytest.importorskip("torch")
    from egolearn.data.text import VocabTokenizer

    root, state = remote
    (root / "vocab.json").write_text(json.dumps({"hello": 1}))
    VocabTokenizer.from_json("hf://models/org/weights@v1/vocab.json")
    assert state.downloads[-1][1] == "model"


def test_unknown_license_fails_before_creating_repository(remote):
    root, state = remote
    state.exists = False
    with pytest.raises(ValueError, match="unsupported license"):
        hub.push(root, name="sample", license_id="LicenseRef-Foo")
    assert not state.creates and not state.uploads
