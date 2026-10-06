"""Export checkpoint weights and construction metadata without publishing pickles."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from pickle import UnpicklingError


def export_checkpoint(
    checkpoint: str | Path, out_dir: str | Path, *, trusted_snapshot: bool = False
) -> Path:
    """Write a new, complete model export; never replace an existing destination."""
    import torch
    from safetensors.torch import save_file

    from egolearn.utils.checkpoint import read_checkpoint
    from egolearn.utils.model_artifact import METADATA, WEIGHTS, file_digest, validate_export
    from egolearn.utils.provenance import code_revision

    destination = Path(out_dir).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"export destination already exists: {destination}")
    state, source = read_checkpoint(checkpoint, trusted_snapshot=trusted_snapshot)
    config = copy.deepcopy(state.get("model_config"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("export requires a checkpoint with model construction metadata")
    contract = state.get("export_contract")
    if not isinstance(contract, dict):
        raise ValueError("export requires a checkpoint with an export contract")
    # Older initialization summaries used a local path. Keep portable lineage only.
    if isinstance(config.get("init_from"), dict) and "checkpoint" in config["init_from"]:
        config["init_from"]["checkpoint"] = Path(config["init_from"]["checkpoint"]).name
    tensors = {}
    for name, tensor in state["model"].items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("export requires a model mapping of names to tensors")
        if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta":
            raise ValueError(f"export requires dense, materialized tensors: {name}")
        # Keep every state-dict key even when tensors share storage or are strided views.
        tensors[name] = tensor.detach().to("cpu").contiguous().clone()
    provenance = {**source, "code_revision": code_revision()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".egolearn-export-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        save_file(tensors, stage / WEIGHTS)
        document = {
            "format_version": 1,
            "model_config": config,
            "export_contract": contract,
            "weights_sha256": file_digest(stage / WEIGHTS),
            "provenance": provenance,
        }
        (stage / METADATA).write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        validate_export(stage)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"export destination already exists: {destination}")
        stage.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egl export", description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--trusted-snapshot",
        action="store_true",
        help="allow unrestricted pickle for an owned snapshot",
    )
    args = parser.parse_args(argv)
    try:
        output = export_checkpoint(
            args.checkpoint, args.out_dir, trusted_snapshot=args.trusted_snapshot
        )
    except (OSError, ValueError, UnpicklingError, RuntimeError) as error:
        parser.exit(1, f"egl export: {error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
