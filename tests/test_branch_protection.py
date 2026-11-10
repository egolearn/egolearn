"""Repository selection for the branch-protection setup command."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup_branch_protection.sh"


@pytest.fixture
def github_call(tmp_path, monkeypatch):
    executable = tmp_path / "gh"
    executable.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$GH_CALL_ARGS"\ncat > "$GH_CALL_BODY"\n'
    )
    executable.chmod(0o755)
    args, body = tmp_path / "args", tmp_path / "body"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GH_CALL_ARGS", str(args))
    monkeypatch.setenv("GH_CALL_BODY", str(body))
    return args, body


def test_missing_repository_does_not_call_github(github_call):
    args, body = github_call
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "owner/repo" in result.stderr
    assert not args.exists()
    assert not body.exists()


def test_explicit_repository_receives_protection_request(github_call):
    args, body = github_call
    result = subprocess.run(
        ["bash", str(SCRIPT), "egolearn/egolearn"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "repos/egolearn/egolearn/branches/main/protection" in args.read_text().splitlines()
    protection = json.loads(body.read_text())
    assert protection["enforce_admins"] is True
    assert protection["allow_force_pushes"] is False
    assert protection["required_pull_request_reviews"]["require_code_owner_reviews"] is True
