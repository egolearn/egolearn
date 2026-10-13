"""Training entry point: hydra composition over the packaged config tree.

    uv run python -m egolearn.train --config-dir projects/babycl/configs \
        --config-name saycam_default
    ... --config-name saycam_default dry_run=true      # validate wiring only
    ... --config-name saycam_default resume=auto       # continue a run
    ... --config-name saycam_default seed=42 driver.rate.steps_per_admit=4

`dry_run=true` builds and validates the full pipeline (including model and
optimizer) without iterating the stream, so wiring mistakes show up
before training starts. Hub references may download artifacts during resolution.
hydra writes the fully-resolved config to
`.hydra/config.yaml` in the run's output dir; that is the file to use when
reproducing the run.
"""

from __future__ import annotations

import glob
import os
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from egolearn.run_record import RunRecord, record_run


@hydra.main(version_base="1.3", config_path="pkg://egolearn.configs", config_name=None)
def main(cfg: DictConfig) -> None:
    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(plain, dict)
    with record_run(plain, "train") as record:
        _train(dict(plain), record)


def _train(plain: dict, record: RunRecord | None) -> None:
    import egolearn

    dry_run = bool(plain.pop("dry_run", False))
    resume = plain.pop("resume", None)

    if resume and plain.get("init_from"):
        raise SystemExit(
            "init_from and resume are mutually exclusive: use init_from for a fresh run, "
            "or set init_from=null to resume a snapshot"
        )

    pipe = egolearn.build(plain)
    print(pipe.summary(), flush=True)

    if dry_run:
        print("[dry-run] pipeline valid.")
        return

    if record is not None:
        pipe.driver.hooks.append(record)

    if resume:
        path = str(resume)
        if path == "auto":
            run_dir = plain.get("run_dir", "runs/default")
            candidates = glob.glob(os.path.join(run_dir, "snapshot_*.pth"))
            if not candidates:
                raise SystemExit(f"resume=auto: no snapshot_*.pth in {run_dir}")
            path = max(candidates, key=os.path.getmtime)
        from egolearn.utils.snapshot import load_snapshot

        load_snapshot(path, pipe.driver)

    pipe.driver.fit()


if __name__ == "__main__":
    main()
