"""Identify configured inputs without inventing revisions for local datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egolearn.hub import resolve_with_metadata, sha256_file, verify_checksums

# These are input arguments used by the shipped streams, tokenizers and probes.
# Do not guess from arbitrary strings or inspect output/config/cache directories.
_PATH_KEYS = {
    "checkpoint",
    "root_dir",
    "data_root",
    "image_root",
    "image_dir",
    "pos_dir",
    "neg_dir",
    "manifest",
    "manifest_json",
    "vocab_json",
    "transcript_json",
    "split_file",
    "csv_path",
    "storyboard_json",
    "metadata_csv",
    "trials_json",
}
_INPUT_SECTIONS = {"stream", "tokenizer", "probes", "benchmarks", "init_from"}


def input_identity(source: str | Path, *, repo_type: str = "dataset") -> tuple[str, dict]:
    """Return a usable path and observed identity; directories need a verified inventory.

    Hub paths are resolved once and the returned local path is used by the run.
    A file hash identifies that file only, never all the images named inside it.
    """
    source = str(source)
    identity: dict[str, Any] = {"sha256": "unavailable", "revision": "unavailable"}
    path = Path(source).expanduser()
    if source.startswith("hf://"):
        path, metadata = resolve_with_metadata(source, repo_type=repo_type)
        identity.update(metadata)
        # `revision` in Hub metadata is the requested tag; commit is immutable.
    identity["name"] = path.name
    try:
        if path.is_file():
            identity.update(kind="file", sha256=sha256_file(path))
        elif path.is_dir():
            identity["kind"] = "directory"
            inventory = path / "checksums.sha256"
            if inventory.is_file():
                problems = verify_checksums(path)
                if problems:
                    identity["reason"] = "checksum_verification_failed"
                else:
                    identity.update(
                        sha256=sha256_file(inventory), checksum_scope="checksums.sha256"
                    )
            else:
                identity["reason"] = "no_checksum_inventory"
        else:
            identity["reason"] = "missing"
    except OSError as error:
        identity["reason"] = type(error).__name__
    return str(path), identity


def prepare_inputs(config: dict) -> dict[str, dict]:
    """Capture known inputs and freeze Hub resolution in this run's config copy.

    Local paths retain their spelling; only Hub references are replaced. Unknown
    component input conventions remain in the config and have no inferred digest.
    """
    inputs: dict[str, dict] = {}
    observed: dict[tuple[str, str], tuple[str, dict]] = {}

    def capture(parent, key, name: str, *, model: bool = False) -> None:
        source = parent[key]
        if not isinstance(source, (str, Path)) or not source or source == "auto":
            return
        kind = "model" if model else "dataset"
        cache_key = (str(source), kind)
        if cache_key not in observed:
            observed[cache_key] = input_identity(source, repo_type=kind)
        path, identity = observed[cache_key]
        inputs[name] = dict(identity)
        if str(source).startswith("hf://"):
            parent[key] = path

    def visit(node, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                name = f"{prefix}.{key}"
                if key in _PATH_KEYS:
                    capture(node, key, name, model=key == "checkpoint")
                elif isinstance(value, (dict, list)):
                    visit(value, name)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{prefix}.{index}")

    for section in sorted(_INPUT_SECTIONS):
        visit(config.get(section), section)
    for key in ("checkpoint", "resume", "model_config"):
        if key in config:
            capture(config, key, key, model=True)
    return inputs
