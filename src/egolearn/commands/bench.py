"""Run pinned training/evaluation suites locally or as a Slurm array."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from egolearn.hub import resolve, sha256_file, verify_checksums
from egolearn.run_record import atomic_json, source_checkout, source_identity

_RESERVED = {"seed", "record_json", "run_dir", "output_json", "hydra", "resume", "dry_run"}


def hydra_quote(value: str) -> str:
    # Hydra accepts quoted strings, but does not decode JSON's Unicode escapes.
    if any(ord(character) < 32 for character in value) or "${" in value:
        raise ValueError("paths cannot contain control characters or interpolation expressions")
    return json.dumps(value, ensure_ascii=False)


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return value


def _keys(value: dict, allowed: set[str], label: str) -> None:
    if unknown := set(value) - allowed:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


def _overrides(values: Any, pinned: set[str]) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError("overrides must be a list of Hydra key=value strings")
    for value in values:
        match = re.fullmatch(r"\+{0,2}([A-Za-z_][A-Za-z0-9_.]*)=(.*)", value, flags=re.DOTALL)
        if not match:
            raise ValueError(f"expected a single Hydra key=value override: {value!r}")
        key = match[1]
        for protected in _RESERVED | pinned:
            if (
                key == protected
                or key.startswith(protected + ".")
                or protected.startswith(key + ".")
            ):
                raise ValueError(f"override cannot change pinned/managed key {protected!r}")
    from hydra.core.override_parser.overrides_parser import OverridesParser

    if any(
        override.is_sweep_override()
        for override in OverridesParser.create().parse_overrides(values)
    ):
        raise ValueError("suite overrides must be single values; seeds define the run set")
    return values


def verify_inputs(inputs: dict[str, dict]) -> None:
    for key, item in inputs.items():
        path = Path(item["path"])
        checksum = path / "checksums.sha256" if path.is_dir() else path
        if sha256_file(checksum) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for input {key!r}")
        if path.is_dir():
            problems = verify_checksums(path)
            if problems:
                raise ValueError(f"input {key!r}: " + "; ".join(problems))


def load_suite(path: Path, extra_overrides: list[str]) -> dict[str, Any]:
    from omegaconf import OmegaConf

    path = Path(os.path.abspath(path.expanduser()))
    raw = _mapping(OmegaConf.to_container(OmegaConf.load(path), resolve=False), "suite")
    # Overrides belong to the job configuration, not the suite's interpolation scope.
    overrides = raw.pop("overrides", [])
    suite = _mapping(OmegaConf.to_container(OmegaConf.create(raw), resolve=True), "suite")
    suite["overrides"] = overrides
    _keys(
        suite,
        {"version", "name", "entry", "config", "seeds", "overrides", "inputs", "metrics"},
        "suite",
    )
    if suite.get("version") != 1 or isinstance(suite.get("version"), bool):
        raise ValueError("suite version must be 1")
    if not isinstance(suite.get("name"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", suite["name"]
    ):
        raise ValueError("suite name must contain letters, digits, underscores or hyphens")
    if suite.get("entry") not in {"train", "eval"}:
        raise ValueError("suite entry must be train or eval")
    config = suite.get("config")
    if (
        not isinstance(config, str)
        or not config
        or config.startswith(("-", "+", "~"))
        or ("=" in config and Path(config).suffix != ".yaml")
    ):
        raise ValueError("suite config must be a packaged name or a .yaml file")
    if "/" in config or Path(config).suffix:
        config_file = Path(config).expanduser()
        if not config_file.is_absolute():
            config_file = path.parent / config_file
        if not config_file.is_file() or config_file.suffix != ".yaml":
            raise ValueError("suite config file must exist and have a .yaml extension")
        suite["config"] = str(config_file.resolve())
    seeds = suite.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be a nonempty list of unique integers in [0, 2**32)")
    inputs = _mapping(suite.get("inputs", {}), "inputs")
    for key, item in inputs.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key):
            raise ValueError(f"invalid input config key {key!r}")
        _overrides([key + "=input"], set())
        item = _mapping(item, f"input {key!r}")
        _keys(item, {"path", "sha256"}, "input")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"input {key!r} needs a path")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", item["sha256"]
        ):
            raise ValueError(f"input {key!r} needs a lowercase SHA-256")
        if item["path"].startswith("hf://"):
            kind = (
                "model"
                if key in {"checkpoint", "model_config", "init_from.checkpoint"}
                else "dataset"
            )
            file = resolve(item["path"], repo_type=kind)
        else:
            file = Path(item["path"]).expanduser()
            if not file.is_absolute():
                file = path.parent / file
        # Preserve scratch aliases: the container binds the configured destination,
        # which may differ from the host's physical path behind a symlink.
        inputs[key] = {"path": os.path.abspath(file), "sha256": item["sha256"]}
    # Overlapping paths would cause the later assignment to replace an earlier pin.
    for key in inputs:
        if any(other != key and other.startswith(key + ".") for other in inputs):
            raise ValueError("input keys must not overlap")
    if suite["entry"] == "eval" and "checkpoint" not in inputs:
        raise ValueError("evaluation suites must pin inputs.checkpoint")
    metrics = _mapping(suite.get("metrics", {}), "metrics")
    if not metrics or any(direction not in {"min", "max"} for direction in metrics.values()):
        raise ValueError("metrics must declare at least one metric with direction min or max")
    suite["metrics"] = metrics
    suite["inputs"] = inputs
    suite["overrides"] = _overrides(suite.get("overrides", []), set(inputs)) + _overrides(
        extra_overrides, set(inputs)
    )
    suite["suite_sha256"] = sha256_file(path)
    verify_inputs(inputs)
    return suite


def _compose(suite: dict, run_dir: Path, seed: int) -> dict:
    from omegaconf import OmegaConf

    overrides = [*suite["overrides"], f"++seed={seed}"]
    overrides += [f"++{key}={hydra_quote(item['path'])}" for key, item in suite["inputs"].items()]
    config_path = Path(suite["config"])
    config_args = ["--config-name", suite["config"]]
    if config_path.is_absolute():
        config_args = [
            "--config-path",
            str(config_path.parent),
            "--config-name",
            config_path.stem,
            "--config-dir",
            str(files("egolearn.configs")),
        ]
    command = [
        sys.executable,
        "-m",
        "egolearn.cli",
        suite["entry"],
        *config_args,
        "--cfg",
        "job",
        *overrides,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError("could not compose suite config:\n" + result.stderr)
    config = OmegaConf.create(result.stdout)
    if not OmegaConf.is_dict(config):
        raise ValueError("resolved config must be a mapping")
    for key, item in suite["inputs"].items():
        if OmegaConf.select(config, key) != item["path"]:
            raise ValueError(f"composed input {key!r} does not match its pin")
    if OmegaConf.select(config, "dry_run") or OmegaConf.select(config, "resume"):
        raise ValueError("suite configs must train/evaluate fresh runs, not dry-run or resume")
    config.seed = seed
    config.record_json = str(run_dir / "record.json")
    if suite["entry"] == "train":
        config.run_dir = str(run_dir / "artifacts")
    else:
        config.output_json = str(run_dir / "evaluation.json")
    plain = OmegaConf.to_container(config, resolve=False)
    assert isinstance(plain, dict)
    # Reject values that cannot be represented faithfully in JSON run records.
    json.dumps(plain, allow_nan=False)
    return plain


def cluster_identity() -> dict[str, Any]:
    """Identify the editable checkout used for planning inside a compute allocation."""
    from egolearn import __version__
    from egolearn.commands.submit import launcher

    root = launcher().parent.parent.parent
    if source_checkout() != root.resolve():
        raise ValueError(
            "submitted suites require an editable installation of this checkout; "
            "inside a CPU allocation, run uv sync --extra train here and use uv run egl bench"
        )

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("cluster suites require a clean checkout")
    image = "\n".join(
        line.strip()
        for line in (root / "containers/IMAGE").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-fA-F]{64}", image):
        raise ValueError("containers/IMAGE must pin a sha256 digest")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": False,
        "uv_lock_sha256": sha256_file(root / "uv.lock"),
        "image": image,
        "package_version": __version__,
    }


def check_cluster_paths(output: Path, inputs: dict[str, dict]) -> None:
    """Refuse plans whose absolute files would be hidden by the container's binds."""
    from egolearn.commands.doctor import cluster_variables, storage_check

    mounts: dict[Path, bool] = {}
    for entry in os.environ.get("APPTAINER_BIND", "").split(","):
        if not entry:
            continue
        parts = entry.split(":")
        destination = Path(parts[1] if len(parts) > 1 else parts[0])
        if destination.is_absolute():
            mounts.setdefault(destination, len(parts) < 3 or "ro" not in parts[2].split(","))
    for names in cluster_variables().values():
        for name in names:
            if value := os.environ.get(name):
                mounts.setdefault(Path(value), name != "EGOLEARN_VENV")
    for name, path in [
        ("output", output),
        *[(key, Path(item["path"])) for key, item in inputs.items()],
    ]:
        hydra_quote(str(path))
        check = storage_check(name, str(path), Path.home())
        if check.status != "ok" or ":" in str(path) or "," in str(path):
            raise ValueError(f"{name} must be on scratch and representable in APPTAINER_BIND")
        covering = [root for root in mounts if root.is_absolute() and path.is_relative_to(root)]
        if not covering:
            raise ValueError(
                f"{name} is outside configured container binds; "
                "set a storage variable or APPTAINER_BIND"
            )
        closest = max(covering, key=lambda root: len(root.parts))
        if name == "output" and not mounts[closest]:
            raise ValueError("suite output is under a read-only container bind")
    check_output_location(output, Path.cwd().resolve())


def check_output_location(output: Path, checkout: Path | None) -> None:
    if checkout is not None and output.resolve().is_relative_to(checkout):
        ignored = subprocess.run(
            ["git", "-C", str(checkout), "check-ignore", "-q", str(output.resolve())],
            check=False,
        )
        if ignored.returncode:
            raise ValueError("suite output inside the checkout must be git-ignored")


def plan_suite(spec: Path, output: Path, overrides: list[str], *, submit: bool = False) -> Path:
    from omegaconf import OmegaConf

    provenance = cluster_identity() if submit else source_identity()
    suite = load_suite(spec, overrides)
    output = Path(os.path.abspath(output.expanduser()))
    hydra_quote(str(output))
    if submit:
        check_cluster_paths(output, suite["inputs"])
    else:
        check_output_location(output, source_checkout())
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    runs = []
    configs = []
    for seed in suite["seeds"]:
        directory = f"seed-{seed}"
        config = _compose(suite, output / directory, seed)
        text = OmegaConf.to_yaml(OmegaConf.create(config), resolve=False)
        configs.append(text)
        runs.append(
            {
                "seed": seed,
                "directory": directory,
                "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    for run, text in zip(runs, configs, strict=True):
        directory = output / run["directory"]
        directory.mkdir()
        (directory / "config.yaml").write_text(text)
    manifest = output / "manifest.json"
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "name": suite["name"],
            "entry": suite["entry"],
            "suite_sha256": suite["suite_sha256"],
            "inputs": suite["inputs"],
            "metrics": suite["metrics"],
            "provenance": provenance,
            "runs": runs,
        },
    )
    return manifest


def run_one(manifest_path: Path, index: int, expected_hash: str) -> int:
    manifest_path = Path(os.path.abspath(manifest_path.expanduser()))
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError("suite manifest changed after planning")
    manifest = json.loads(manifest_path.read_text())
    if not 0 <= index < len(manifest["runs"]):
        raise ValueError("run index is outside the manifest")
    run = manifest["runs"][index]
    directory = manifest_path.parent / run["directory"]
    if directory.resolve().parent != manifest_path.parent.resolve():
        raise ValueError("run directory must be directly under the manifest directory")
    record_path = directory / "record.json"
    # A queued task may be retried or launched twice; do not overwrite its first result.
    with (directory / ".claimed").open("x"):
        pass
    identity = source_identity()
    base_record = {
        "schema_version": 1,
        "entry": manifest["entry"],
        "seed": run["seed"],
        "suite_sha256": manifest["suite_sha256"],
        "manifest_sha256": expected_hash,
        "config_sha256": run["config_sha256"],
        "inputs": manifest["inputs"],
        "provenance": identity,
        "metrics": {},
        "status": "failed",
    }
    try:
        verify_inputs(manifest["inputs"])
        config = directory / "config.yaml"
        if sha256_file(config) != run["config_sha256"]:
            raise ValueError("resolved run config changed after planning")
        if identity != manifest["provenance"]:
            raise ValueError("running code/environment identity differs from the planned suite")
        command = [
            sys.executable,
            "-m",
            "egolearn.cli",
            manifest["entry"],
            "--config-path",
            str(directory),
            "--config-name",
            "config",
            f"hydra.run.dir={hydra_quote(str(directory / 'hydra'))}",
            "hydra.job.chdir=false",
        ]
        with (directory / "run.log").open("w") as log:
            process = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
        if record_path.is_file():
            record = json.loads(record_path.read_text())
        else:
            record = {**base_record, "error_type": "MissingRunRecord"}
        # Shared data can change while a task runs; never publish that task's score.
        verify_inputs(manifest["inputs"])
        if (
            manifest["entry"] == "eval"
            and record.get("checkpoint")
            and Path(manifest["inputs"]["checkpoint"]["path"]).is_file()
            and record["checkpoint"].get("sha256") != manifest["inputs"]["checkpoint"]["sha256"]
        ):
            raise ValueError("loaded checkpoint does not match the pinned bytes")
        record.update(
            {
                key: value
                for key, value in base_record.items()
                if key in {"suite_sha256", "manifest_sha256", "config_sha256", "inputs"}
            }
        )
        if (process.returncode or record.get("status") != "completed") and record.get(
            "status"
        ) != "nonfinite":
            record["status"] = "failed"
        record["returncode"] = process.returncode
        atomic_json(record_path, record)
        return int(record["status"] != "completed")
    except BaseException as error:
        atomic_json(record_path, {**base_record, "error_type": type(error).__name__})
        raise


def run_suite(manifest: Path, submit: bool, sbatch: list[str]) -> int:
    payload = json.loads(manifest.read_text())
    digest = sha256_file(manifest)
    if submit:
        from egolearn.commands.submit import launcher

        script = launcher()
        command_file = manifest.parent / "commands.sh"
        command_file.write_text(
            "".join(
                shlex.join(
                    [
                        "python",
                        "-m",
                        "egolearn.commands.bench",
                        "_run",
                        str(manifest),
                        str(index),
                        "--manifest-sha256",
                        digest,
                    ]
                )
                + "\n"
                for index in range(len(payload["runs"]))
            )
        )
        return subprocess.call(
            ["bash", str(script), *sbatch, "--no-requeue", "--commands", str(command_file)]
        )
    failed = 0
    for index in range(len(payload["runs"])):
        try:
            failed += run_one(manifest, index, digest)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f"seed {payload['runs'][index]['seed']}: {error}", file=sys.stderr)
            failed += 1
    return int(bool(failed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="verify inputs and run one task per seed")
    running.add_argument("spec", type=Path)
    running.add_argument("--output", type=Path, required=True, help="new directory for this suite")
    running.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    running.add_argument(
        "--submit",
        action="store_true",
        help="plan and submit a Slurm array from a CPU allocation (never a login node)",
    )
    running.add_argument(
        "--sbatch", action="append", default=[], help="Slurm option; use --sbatch=..."
    )
    worker = commands.add_parser("_run", help=argparse.SUPPRESS)
    worker.add_argument("manifest", type=Path)
    worker.add_argument("index", type=int)
    worker.add_argument("--manifest-sha256", required=True)
    table = commands.add_parser("table", help="summarize complete, finite seed results")
    table.add_argument("manifest", type=Path)
    table.add_argument("--format", choices=("md", "csv", "json"), default="md")
    table.add_argument("--baseline", type=Path)
    table.add_argument(
        "--allow-unverified", action="store_true", help="allow exploratory dirty/unknown code"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.sbatch and not args.submit:
                parser.error("--sbatch requires --submit")
            if args.submit:
                from egolearn.commands.submit import launcher

                launcher()  # reject missing checkout before composing or writing anything
            manifest = plan_suite(args.spec, args.output, args.override, submit=args.submit)
            print(f"Manifest: {manifest}", flush=True)
            return run_suite(manifest, args.submit, args.sbatch)
        if args.command == "_run":
            return run_one(args.manifest, args.index, args.manifest_sha256)
        from egolearn.commands.bench_table import print_table

        return print_table(args.manifest, args.format, args.baseline, args.allow_unverified)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
