"""Code identity belongs to the installed source checkout, not the working directory."""

import subprocess

import pytest

from egolearn.utils import provenance


def test_revision_tracks_checkout_commit_and_dirty_files(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    source = root / "src/egolearn/utils/provenance.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic source\n")
    monkeypatch.setattr(provenance, "__file__", str(source))
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "test",
    )
    commit = git("rev-parse", "HEAD")
    assert provenance.code_revision() == {"commit": commit, "dirty": False}
    source.write_text("# modified source\n")
    assert provenance.code_revision() == {"commit": commit, "dirty": True}
    git("restore", ".")
    (root / "untracked.py").touch()
    assert provenance.code_revision() == {"commit": commit, "dirty": True}


def test_installed_package_does_not_use_callers_git_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "site-packages/egolearn/utils/p.py"))
    assert provenance.code_revision() == {"commit": "unavailable", "dirty": "unavailable"}


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("git", 5)])
def test_revision_is_best_effort(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert provenance.code_revision() == {"commit": "unavailable", "dirty": "unavailable"}


def test_vendored_install_does_not_claim_enclosing_repository(tmp_path, monkeypatch):
    root = tmp_path / "unrelated"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(provenance, "__file__", str(root / "vendor/egolearn/utils/provenance.py"))

    def forbidden(*args, **kwargs):
        raise AssertionError("queried unrelated repository")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert provenance.code_revision() == {"commit": "unavailable", "dirty": "unavailable"}
