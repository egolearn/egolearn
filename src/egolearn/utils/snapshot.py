"""Save and resume training snapshots.

A snapshot captures everything a run needs to continue -- model, optimizer,
clocks/RNG states, memory tables, label stores, and the LR-schedule anchor.
`load_snapshot` restores all of it AND prepares the driver (the object that
runs the training loop) to continue:

  * streaming (LockstepDriver): the stream and the admission algorithms are
    deterministic, so the driver replays the first `clock.admitted`
    observations to rebuild the admission components' internal state (sliding
    windows, watermark, global-id handover) -- with event routing turned off
    during the replay, since the memories themselves come from the snapshot.
    The replay starts from fresh seed-derived RNGs and consumes them exactly
    like the original pass; the snapshot's saved RNG states are restored
    AFTER it, as are memory tables and label stores, so anything the replay
    touched ends up at the snapshot's authoritative values.
  * offline (OfflineDriver): no replay needed; fit() continues from
    `clock.step` (per-epoch shuffles are derived from (seed, epoch), so the
    epoch order regenerates exactly).

Restore order matters for devices too: the model moves to ctx.device before
the optimizer state loads, so torch casts optimizer state onto the right
device.

A possible future improvement is to rebuild memory directly from the log of
admitted events, with no stream replay at all -- the finalized-event
contract keeps that option open.
"""

from __future__ import annotations

import os

import torch

from egolearn.hub import resolve


def save_snapshot(path: str, driver) -> str:
    from egolearn.run_record import active_record

    record = active_record("train")
    composer = getattr(driver, "composer", None)
    label_stores = getattr(composer, "label_stores", None) or {}
    state = {
        "run_provenance": (
            record.training_provenance()
            if record is not None
            else getattr(driver, "run_provenance", None)
        ),
        "model": driver.model.state_dict(),
        "model_config": getattr(driver.model, "construction_config", None),
        "export_contract": (
            driver.model.export_contract() if hasattr(driver.model, "export_contract") else None
        ),
        "optimizer": driver.optimizer.state_dict(),
        # gradients accumulated but not yet applied (grad_accumulation > 1,
        # snapshot mid-window): without them a resumed run's next optimizer
        # step would see fewer micro-steps than the uninterrupted run's
        "grads": {
            n: p.grad.detach().cpu()
            for n, p in driver.model.named_parameters()
            if p.grad is not None
        },
        "driver": {
            "metadata": getattr(driver, "snapshot_metadata", None),
            "exhausted": bool(getattr(driver, "_exhausted", False)),
            "skipped_steps": getattr(driver, "skipped_steps", 0),
            "nonfinite_steps": getattr(driver, "nonfinite_steps", 0),
        },
        "context": driver.ctx.state_dict(),
        "memories": {
            name: mem.state_dict() for name, mem in getattr(driver, "memories", {}).items()
        },
        "label_stores": {name: store.state_dict() for name, store in label_stores.items()},
        "schedule": (
            driver.schedule.state_dict() if getattr(driver, "schedule", None) is not None else None
        ),
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    if record is not None:
        record.capture_output(os.path.basename(path), path)
    print(f"[snapshot] saved {path}", flush=True)
    return path


def load_snapshot(path: str, driver) -> None:
    """Restore a snapshot and prepare `driver` to continue training."""
    from egolearn.run_record import active_record
    from egolearn.utils.input_provenance import input_identity

    # weights_only=False: snapshots are our own files and hold more than
    # tensors (refs, RNG states, label stores)
    identity = None
    if getattr(driver, "run_config", None) is not None and driver.run_config.get(
        "record_json", "auto"
    ):
        record = active_record("train")
        if (
            record is not None
            and str(path) == record.config.get("resume")
            and "resume" in record.value["input_provenance"]
        ):
            # The CLI already froze this input. Retain its original Hub revision
            # instead of replacing it with the resolved local file's identity.
            identity = record.value["input_provenance"]["resume"]
        else:
            path, identity = input_identity(path, repo_type="model")
        driver.run_inputs = {**(driver.run_inputs or {}), "resume": identity}
        if record is not None:
            record.value["input_provenance"]["resume"] = identity
    state = torch.load(resolve(path, repo_type="model"), map_location="cpu", weights_only=False)
    if validate := getattr(driver, "validate_snapshot", None):
        validate(state)
    if identity is not None:
        identity["training_provenance"] = state.get("run_provenance") or "unavailable"

    # model to its device BEFORE the optimizer loads, so optimizer state is
    # cast onto the parameters' device by torch
    driver.model.to(driver.ctx.device)
    driver.model.load_state_dict(state["model"])
    # Resume restores lineage; construction metadata describes the live model.
    saved_config = state.get("model_config") or {}
    if "init_from" in saved_config:
        driver.model.construction_config = dict(
            getattr(driver.model, "construction_config", None) or {}
        )
        driver.model.construction_config["init_from"] = saved_config["init_from"]
    driver.optimizer.load_state_dict(state["optimizer"])
    if state.get("schedule") is not None and getattr(driver, "schedule", None) is not None:
        driver.schedule.load_state_dict(state["schedule"])

    # Order matters: clock first (the replay below needs to know how far to
    # go), RNG states LAST.  The replay re-runs admission from the start of
    # the stream, consuming RNGs exactly like the original pass did -- if the
    # saved states were restored first, the replay would consume them a
    # second time and the resumed run would diverge from an uninterrupted
    # one.
    driver.ctx.load_clock(state["context"])
    if hasattr(driver, "_resume_replay"):
        # exhausted: the snapshot was taken after the stream ran dry, so the
        # replay must also consume the final session-end flush -- otherwise
        # the resumed fit() would re-run it for real, duplicating events the
        # restored memories already hold and training an extra quantum the
        # uninterrupted run never had
        driver._resume_replay(exhausted=state.get("driver", {}).get("exhausted", False))
    driver.ctx.load_state_dict(state["context"])
    if state.get("grads"):
        params = dict(driver.model.named_parameters())
        for n, g in state["grads"].items():
            params[n].grad = g.to(driver.ctx.device)

    # memories/labels restored AFTER replay -- any label writes the replay
    # made are overwritten by the snapshot's saved values
    for name, st in state.get("memories", {}).items():
        driver.memories[name].load_state_dict(st)
    composer = getattr(driver, "composer", None)
    stores = getattr(composer, "label_stores", None) or {}
    for name, st in state.get("label_stores", {}).items():
        stores[name].load_state_dict(st)
    driver.skipped_steps = state.get("driver", {}).get("skipped_steps", 0)
    driver.nonfinite_steps = state.get("driver", {}).get("nonfinite_steps", 0)
    driver.run_provenance = state.get("run_provenance")
    print(
        f"[snapshot] resumed from {path} (step={driver.ctx.clock.step} "
        f"admitted={driver.ctx.clock.admitted})",
        flush=True,
    )
