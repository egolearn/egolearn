"""Thin experiment entry point: shared training plus an exact random baseline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import egolearn
from egolearn.methods.vjepa2.loss import VJEPA2
from egolearn.run_record import record_run
from egolearn.training.hooks import CheckpointHook, WandbHook
from egolearn.utils.snapshot import load_snapshot


def run(config: dict):
    """Train with the ordinary driver, preserving baseline/EMA and run provenance."""
    config = deepcopy(config)
    if str(config.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V-JEPA 2, but no GPU is visible")
    with record_run(config, "train") as record:
        cfg = dict(record.config if record else config)
        resume = cfg.get("resume")
        if resume and cfg.get("init_from"):
            raise ValueError("init_from and resume are mutually exclusive")
        pipe = egolearn.build(cfg)
        print(pipe.summary(), flush=True)
        exit_code = 1
        try:
            if cfg.get("dry_run"):
                exit_code = 0
                return pipe
            losses = [loss for loss in pipe.losses if isinstance(loss, VJEPA2)]
            if not losses:
                raise ValueError("V-JEPA 2 runner requires loss:vjepa2")
            root = Path(cfg["run_dir"])
            if resume:
                if resume == "auto":
                    candidates = list(root.glob("snapshot_*.pth"))
                    if not candidates:
                        raise ValueError("resume=auto found no snapshots")
                    resume = str(max(candidates, key=lambda path: path.stat().st_mtime))
                load_snapshot(str(resume), pipe.driver)
                # Even a step-0 snapshot owns its teacher state.
                for loss in losses:
                    loss._synced = True
            else:
                if (root / "random.pth").exists() or (root / "final.pth").exists():
                    raise FileExistsError(
                        "run_dir already has checkpoints; resume or use a new directory"
                    )
                for loss in losses:
                    loss.initialize_target(pipe.model)
                CheckpointHook(str(root))._save(pipe.driver, "random.pth")
            if cfg.get("preflight", False):
                diagnostics = _preflight_inputs(pipe)
                if record is not None:
                    record.value["preflight"] = diagnostics
            pipe.driver.fit()
            if record is not None:
                record.value["clip_index"] = pipe.driver.index_stats
            exit_code = 0
            return pipe
        finally:
            try:
                pipe.loader.close()
            finally:
                for hook in pipe.driver.hooks:
                    if isinstance(hook, WandbHook) and hook._run is not None:
                        hook._run.finish(exit_code=exit_code)


def _preflight_inputs(pipe) -> dict[str, float]:
    """Check the next shuffled training batch without consuming the clock or RNG."""
    refs = pipe.driver._build_index()
    if len(refs) < pipe.driver.n:
        raise ValueError("preflight requires at least one complete batch of clips")
    batches_per_epoch = len(refs) // pipe.driver.n
    epoch, batch_index = divmod(pipe.ctx.clock.step, batches_per_epoch)
    order = list(range(len(refs)))
    pipe.ctx.fresh_rng(f"offline.shuffle.epoch{epoch}").shuffle(order)
    selected = pipe.driver._pick(refs, order, batch_index)
    batch = pipe.driver._compose(selected, seq=pipe.ctx.clock.compose_seq)
    diagnostics = {}
    for loss in pipe.losses:
        if isinstance(loss, VJEPA2):
            clips = batch[loss.slot]["views"]
            motion = (clips[:, :, 1:] - clips[:, :, :-1]).abs().flatten(3).amax(3)
            duplicate = motion == 0
            diagnostics[f"{loss.name}/identical_adjacent_fraction"] = float(
                duplicate.float().mean()
            )
            if duplicate.all(dim=2).any():
                raise ValueError("preflight found clips whose frames are all identical")
            values = loss.compute(batch, pipe.model.to(pipe.ctx.device), pipe.ctx)
            torch.stack(list(values.values())).sum().backward()
            for module in (
                pipe.model.encoders[loss.context],
                pipe.model.predictors[loss.predictor],
            ):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise ValueError("preflight found missing or nonfinite gradients")
                if not any(torch.count_nonzero(g) > 0 for g in grads):
                    raise ValueError("preflight found zero gradients")
            if any(p.grad is not None for p in pipe.model.encoders[loss.target].parameters()):
                raise ValueError("EMA target received gradients")
            if any(float(v) <= 0 for k, v in loss.diagnostics.items() if k.endswith("_std")):
                raise ValueError("preflight found zero representation spread")
    pipe.driver.optimizer.zero_grad(set_to_none=True)
    if pipe.ctx.device.type == "cuda":
        diagnostics["peak_cuda_allocated_gb"] = (
            torch.cuda.max_memory_allocated(pipe.ctx.device) / 1e9
        )
    print(
        f"[preflight] finite nonzero gradients, frozen target, nonzero spread: {diagnostics}",
        flush=True,
    )
    return diagnostics


@hydra.main(version_base="1.3", config_path="pkg://egolearn.configs", config_name=None)
def main(cfg: DictConfig) -> None:
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    run(plain)


if __name__ == "__main__":
    main()
