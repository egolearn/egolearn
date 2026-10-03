"""Best-effort identification of the installed source, independent of cwd."""

from __future__ import annotations

import subprocess
from pathlib import Path


def code_revision() -> dict[str, str | bool]:
    """Return the source commit and dirty flag, or explicit unavailable values."""
    unavailable: dict[str, str | bool] = {"commit": "unavailable", "dirty": "unavailable"}
    source = Path(__file__).resolve()
    root = source.parents[3]
    # Wheels have no checkout metadata. Do not report a caller's unrelated repo.
    if source.parents[1] != root / "src" / "egolearn" or not (root / ".git").exists():
        return unavailable
    try:

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()

        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        }
    except (OSError, subprocess.SubprocessError):
        return unavailable
