"""One installed command for training, evaluation, data, jobs and benchmark suites."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from importlib.resources import files
from pathlib import Path

from egolearn import __version__


def _hydra_arguments(arguments: list[str]) -> list[str]:
    """Translate an optional leading config name/path; preserve other arguments."""
    if not arguments:
        return arguments
    config = arguments[0]
    if (
        "=" in config
        or config.startswith(("-", "+"))
        or (config.startswith("~") and not config.startswith("~/"))
    ):
        return arguments

    rest = arguments[1:]
    path = Path(config).expanduser()
    if "/" in config or path.suffix in {".yaml", ".yml"}:
        if path.suffix != ".yaml":
            raise ValueError("config files must have Hydra's .yaml extension")
        if not path.is_file():
            raise ValueError(f"config file not found: {config}")
        path = path.resolve()
        # Replace the primary config source: --config-dir only appends a search
        # path and could silently select a packaged config with the same name.
        # Keep installed defaults as a fallback without overriding hydra.searchpath.
        return [
            "--config-path",
            str(path.parent),
            "--config-name",
            path.stem,
            "--config-dir",
            str(files("egolearn.configs")),
            *rest,
        ]
    return ["--config-name", config, *rest]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, prepare data, submit jobs and run benchmark suites.",
        epilog=(
            "CONFIG is a packaged name or a .yaml file path. Arguments after train/eval "
            "belong to Hydra, including --help, --multirun, and key=value overrides."
        ),
    )
    parser.add_argument("--version", action="version", version=f"egolearn {__version__}")
    parser.add_argument(
        "command", nargs="?", choices=("train", "eval", "data", "submit", "doctor", "bench")
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="[CONFIG] [HYDRA_ARGS ...]")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command not in {"train", "eval"}:
        command = importlib.import_module(f"egolearn.commands.{args.command}")
        return command.main(args.arguments)
    try:
        hydra_args = _hydra_arguments(args.arguments)
    except ValueError as error:
        parser.error(str(error))
    module = "egolearn.train" if args.command == "train" else "egolearn.evaluate"
    # Replace this process to retain the underlying module's exit status and signals.
    # Do not change cwd: relative data/output paths remain relative to the caller.
    os.execv(sys.executable, [sys.executable, "-m", module, *hydra_args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
