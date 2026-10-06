"""Read-only checks for local installations and cluster environments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def cluster_variables() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"required": [], "storage": []}
    for line in files("egolearn.commands").joinpath("cluster_paths.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            kind, name = line.split()
            result[kind].append(name)
    return result


def storage_check(name: str, value: str, home: Path) -> Check:
    path = Path(value).expanduser()
    try:
        if not path.is_absolute():
            return Check("fail", name, "must be an absolute path outside $HOME")
        if path.resolve().is_relative_to(home.resolve()):
            return Check("fail", name, "resolves under $HOME; configure scratch storage")
    except (OSError, RuntimeError):
        return Check("fail", name, "cannot resolve path")
    return Check("ok", name, "configured outside $HOME")


def check_cuda(required: bool) -> Check:
    # Isolate optional native-library loading. No model, tensor, or GPU allocation.
    script = (
        "import json,torch; print(json.dumps({'available': torch.cuda.is_available(), "
        "'build': torch.version.cuda}))"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTORCH_NVML_BASED_CUDA_CHECK": "1"},
        )
        if process.returncode:
            raise ValueError("torch could not be loaded")
        result = json.loads(process.stdout.splitlines()[-1])
        if result["available"]:
            return Check("ok", "cuda", f"available (torch CUDA {result['build']})")
        detail = "unavailable; CPU training remains available"
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError):
        detail = "could not query torch CUDA in an isolated process"
    return Check("fail" if required else "warn", "cuda", detail)


def inspect_environment(
    *,
    cluster: bool = False,
    extras: tuple[str, ...] = (),
    require_cuda: bool = False,
    required_env: tuple[str, ...] = (),
    skip_cuda: bool = False,
) -> list[Check]:
    checks = [Check("ok" if sys.version_info >= (3, 12) else "fail", "python", "requires 3.12+")]
    virtual = sys.prefix != sys.base_prefix or bool(os.environ.get("CONDA_PREFIX"))
    checks.append(Check("ok" if virtual else "warn", "venv", "active" if virtual else "not active"))
    installed = {}
    for package in ("torch", "torchvision", "pillow", "wandb"):
        with suppress(importlib.metadata.PackageNotFoundError):
            installed[package] = importlib.metadata.version(package)
    for extra, packages in (
        ("train", ("torch", "torchvision", "pillow", "wandb")),
        ("video", ("pillow",)),
    ):
        missing = [package for package in packages if package not in installed]
        status = "fail" if missing and extra in extras else "warn" if missing else "ok"
        checks.append(
            Check(
                status,
                f"extra:{extra}",
                "missing " + ", ".join(missing) if missing else "installed",
            )
        )
    if not skip_cuda:
        if "torch" in installed:
            checks.append(check_cuda(require_cuda))
        else:
            checks.append(
                Check("fail" if require_cuda else "warn", "cuda", "torch is not installed")
            )
    root = Path(os.environ.get("EGOLEARN_DATA", "data")).expanduser()
    if not root.exists():
        checks.append(Check("warn", "data-root", "does not exist; no directory was created"))
    elif not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
        checks.append(Check("fail", "data-root", "not a readable directory"))
    else:
        checks.append(Check("ok", "data-root", "readable"))
    for name in required_env:
        checks.append(
            Check(
                "ok" if os.environ.get(name) else "fail",
                name,
                "set" if os.environ.get(name) else "not set",
            )
        )
    if cluster:
        variables = cluster_variables()
        home = Path.home()
        checks.append(storage_check("checkout", str(Path.cwd()), home))
        for kind, names in variables.items():
            for name in names:
                value = os.environ.get(name)
                if value:
                    checks.append(storage_check(name, value, home))
                elif kind == "required":
                    checks.append(Check("fail", name, "not set; source env.sh"))
        if not os.environ.get("EGOLEARN_DATA"):
            checks.append(storage_check("data-root-location", str(root.absolute()), home))
        for binary in ("bash", "git", "sbatch"):
            checks.append(
                Check(
                    "ok" if shutil.which(binary) else "fail",
                    binary,
                    "on PATH" if shutil.which(binary) else "not on PATH",
                )
            )
        # Apptainer is required on compute nodes, but need not be on a login node.
        if os.environ.get("SLURM_JOB_ID"):
            checks.append(
                Check(
                    "ok" if shutil.which("apptainer") else "fail",
                    "apptainer",
                    "on PATH" if shutil.which("apptainer") else "not on PATH",
                )
            )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cluster", action="store_true", help="also check cluster paths and tools")
    parser.add_argument("--extra", action="append", choices=("train", "video"), default=[])
    parser.add_argument("--require-env", action="append", default=[], metavar="NAME")
    cuda = parser.add_mutually_exclusive_group()
    cuda.add_argument("--require-cuda", action="store_true")
    cuda.add_argument("--skip-cuda", action="store_true", help="skip the isolated CUDA query")
    args = parser.parse_args(argv)
    if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in args.require_env):
        parser.error("--require-env expects an environment variable name")
    checks = inspect_environment(
        cluster=args.cluster or bool(os.environ.get("SLURM_JOB_ID")),
        extras=tuple(args.extra),
        require_cuda=args.require_cuda,
        required_env=tuple(args.require_env),
        skip_cuda=args.skip_cuda,
    )
    for check in checks:
        print(f"{check.status}: {check.name}: {check.detail}")
    return int(any(check.status == "fail" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
