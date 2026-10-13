"""Submit the current clean checkout using the existing Slurm launcher."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


def launcher(checkout: Path | None = None) -> Path:
    root = checkout if checkout is not None else Path.cwd()
    script = root / "scripts/slurm/submit"
    project = root / "pyproject.toml"
    if project.is_file() and script.is_file():
        with project.open("rb") as handle:
            if tomllib.load(handle).get("project", {}).get("name") == "egolearn":
                return script.resolve()
    raise ValueError(
        "run 'egl submit' from an EgoLearn checkout root containing scripts/slurm/submit; "
        "the launcher snapshots that checkout (an installed wheel alone has no code to archive)"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        print(
            "usage: egl submit [sbatch options] [--dry-run] [--sweep] -- command [args ...]\n"
            "       egl submit [sbatch options] --commands FILE\n\n"
            "Run from a clean EgoLearn checkout after sourcing env.sh.\n"
            "On login nodes, invoke scripts/slurm/submit directly; this alias runs Python.\n"
            "Arguments pass unchanged to scripts/slurm/submit; see docs/cluster.md.\n"
            "--dry-run creates a local snapshot but submits no jobs."
        )
        return 0
    try:
        script = launcher()
        os.execvp("bash", ["bash", str(script), *arguments])
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
