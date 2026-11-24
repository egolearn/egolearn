"""Public CLI routing, access rules and on-disk checksum verification."""

import json
import os
import subprocess
import sys

import pytest

from egolearn.commands import data
from egolearn.hub import write_checksums

PREPARER_OPTIONS = {
    "shape-bias": "--force-download",
    "spatial-relations": "--generator-root",
    "spatial-monitoring": "--exp1-per-condition",
    "babyview-4afc": "--vocab_json",
    "labeled-s-4afc": "--trials_json",
    "grasp": "--data_root",
    "konkle-4afc": "--vocab_json",
}


def test_catalog_is_unique_and_distinguishes_metadata_from_pixels(capsys):
    assert len(data.DATASETS) == len({item.name for item in data.DATASETS})
    assert data.main(["list", "--json"]) == 0
    rows = {row["name"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["labeled-s"]["access"] == "IRB"
    assert rows["labeled-s-splits"]["access"] == "gated"
    assert rows["babyview-4afc"]["access"] == "gated"
    assert rows["imagenet"]["access"] == "upstream-licensed"
    assert rows["intphys2"]["access"] == "upstream-licensed"
    assert rows["grasp"]["access"] == "upstream-licensed"
    assert rows["grasp"]["collection"] is None and rows["grasp"]["prepare_module"]


@pytest.mark.parametrize("name", ["saycam", "imagenet", "grasp", "unknown"])
def test_unavailable_fetch_fails_without_contacting_hub(monkeypatch, name):
    def unexpected(*args, **kwargs):
        pytest.fail("restricted/unknown dataset must not contact Hub")

    monkeypatch.setattr(data.hub, "fetch", unexpected)
    with pytest.raises(SystemExit) as error:
        data.main(["fetch", name, "--repo", "test/override"])
    assert error.value.code == 1


def test_fetch_uses_shared_root_and_preserves_verification_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("EGOLEARN_DATA", str(tmp_path / "env root"))
    monkeypatch.setattr(data.hub, "fetch", lambda *args, **kwargs: calls.append((args, kwargs)))
    data.main(["fetch", "shape-bias", "--revision", "v1", "--repo", "test/stimuli"])
    assert calls == [
        (
            ("shape-bias", tmp_path / "env root"),
            {
                "revision": "v1",
                "repo": "test/stimuli",
                "allow_unverified": False,
                "replace": False,
            },
        )
    ]
    data.main(["--root", str(tmp_path), "fetch", "shape-bias", "--allow-unverified"])
    assert calls[-1][0][1] == tmp_path
    assert calls[-1][1]["allow_unverified"] is True
    assert calls[-1][1]["repo"] == "egolearn/shape-bias"


def test_prepare_forwards_arguments_and_explicit_paths_win(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(os, "execv", lambda *args: calls.append(args))
    data.main(
        [
            "--root",
            str(tmp_path / "with spaces"),
            "prepare",
            "labeled-s-4afc",
            "--vocab_json",
            "vocab file.json",
            "--trials_json",
            "trials.json",
            "--output=explicit.json",
        ]
    )
    executable, command = calls[0]
    assert executable == sys.executable
    assert command[:3] == [sys.executable, "-m", data.dataset("labeled-s-4afc").prepare_module]
    assert str(tmp_path / "with spaces/labeled-s-4afc/manifest.json") in command
    assert command[-1] == "--output=explicit.json"
    assert "vocab file.json" in command


def test_verify_detects_corruption_and_never_rewrites_files(tmp_path, capsys):
    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    checksums = write_checksums(tmp_path)
    before = checksums.read_bytes()
    assert data.main(["verify", str(tmp_path)]) == 0
    payload.write_text('{"changed":true}')
    assert data.main(["verify", str(tmp_path)]) == 1
    assert "altered: payload.json" in capsys.readouterr().out
    assert checksums.read_bytes() == before
    checksums.unlink()
    with pytest.raises(SystemExit) as error:
        data.main(["verify", str(tmp_path)])
    assert error.value.code == 1
    assert not checksums.exists()


def test_fetch_preserves_prepared_directory_without_explicit_replace(tmp_path, monkeypatch):
    target = tmp_path / "shape-bias"
    target.mkdir()
    sentinel = target / "locally-prepared.json"
    sentinel.write_text("preserve me")
    calls = []
    monkeypatch.setattr(data.hub, "fetch", lambda *args, **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit):
        data.main(["--root", str(tmp_path), "fetch", "shape-bias"])
    assert not calls
    assert sentinel.read_text() == "preserve me"
    data.main(["--root", str(tmp_path), "fetch", "shape-bias", "--replace"])
    assert calls[0]["replace"] is True


def test_fetch_and_prepare_use_the_same_layout(tmp_path):
    for name in ("shape-bias", "babyview-4afc"):
        item = data.dataset(name)
        defaults = dict(item.prepare_defaults)
        output = defaults.get("--output-dir", defaults.get("--out_dir"))
        assert output == item.collection
    # The hosted spatial collection bundles subsets; neither preparer fetches alone.
    assert data.dataset("spatial-relation-categorization").prepare_module is None
    assert data.dataset("spatial-relations").collection is None
    assert data.dataset("spatial-monitoring").collection is None
    assert (
        dict(data.dataset("spatial-monitoring").prepare_defaults)["--full-root"]
        == "spatial-relations"
    )


@pytest.mark.parametrize("name", ["shape-bias", "babyview-4afc", "spatial-relations"])
def test_prepare_preserves_default_fetched_collection(tmp_path, monkeypatch, name):
    target = tmp_path / name
    target.mkdir()
    marker = target / ".egolearn-fetch.json"
    marker.write_text('{"commit":"synthetic"}')
    calls = []
    monkeypatch.setattr(os, "execv", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        data.main(["--root", str(tmp_path), "prepare", name])
    assert not calls
    assert marker.read_text() == '{"commit":"synthetic"}'
    with pytest.raises(SystemExit) as help_exit:
        data.main(["--root", str(tmp_path), "prepare", name, "--help"])
    assert help_exit.value.code == 0
    assert not calls
    if name == "spatial-relations":
        data.main(["--root", str(tmp_path), "prepare", name, "--validate-only"])
        assert len(calls) == 1
    flag = "--out_dir" if name == "babyview-4afc" else "--output-dir"
    data.main(["--root", str(tmp_path), "prepare", name, flag, str(tmp_path / "new")])
    assert calls[-1][1][-1] == str(tmp_path / "new")


def test_preparation_explains_missing_extra(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(data.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit):
        data.main(["--root", str(tmp_path), "prepare", "shape-bias"])
    assert "egolearn[train]" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name,help_args",
    [
        ("shape-bias", ["--help"]),
        ("spatial-monitoring", ["--", "--help"]),
        ("grasp", ["--help"]),
        ("konkle-4afc", ["--help"]),
    ],
)
def test_installed_preparation_help_needs_no_training_dependencies(tmp_path, name, help_args):
    item = data.dataset(name)
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'PIL', 'numpy', 'hydra'} or (
            fullname.startswith('egolearn.evaluation')
        ):
            raise AssertionError(f'preparation help imported {fullname}')
sys.meta_path.insert(0, Block())
from egolearn.cli import main
raise SystemExit(main(['data', 'prepare', *sys.argv[1:]]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, item.name, *help_args],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert f"usage: egl data prepare {item.name}" in result.stdout
    assert PREPARER_OPTIONS[item.name] in result.stdout
    assert not list(tmp_path.iterdir())
