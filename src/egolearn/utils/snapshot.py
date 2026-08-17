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
    Memory tables and label stores are restored AFTER the replay, so any
    label writes the replay makes are overwritten by the snapshot's saved
    values.
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


def save_snapshot(path: str, driver) -> str:
    composer = getattr(driver, "composer", None)
    label_stores = getattr(composer, "label_stores", None) or {}
    state = {
        "model": driver.model.state_dict(),
        "optimizer": driver.optimizer.state_dict(),
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
    print(f"[snapshot] saved {path}", flush=True)
    return path


def load_snapshot(path: str, driver) -> None:
    """Restore a snapshot and prepare `driver` to continue training."""
    # weights_only=False: snapshots are our own files and hold more than
    # tensors (refs, RNG states, label stores)
    state = torch.load(path, map_location="cpu", weights_only=False)

    # model to its device BEFORE the optimizer loads, so optimizer state is
    # cast onto the parameters' device by torch
    driver.model.to(driver.ctx.device)
    driver.model.load_state_dict(state["model"])
    driver.optimizer.load_state_dict(state["optimizer"])
    driver.ctx.load_state_dict(state["context"])
    if state.get("schedule") is not None and getattr(driver, "schedule", None) is not None:
        driver.schedule.load_state_dict(state["schedule"])

    # streaming: rebuild admission-internal state by deterministic replay
    if hasattr(driver, "_resume_replay"):
        driver._resume_replay()

    # memories/labels restored AFTER replay -- any label writes the replay
    # made are overwritten by the snapshot's saved values
    for name, st in state.get("memories", {}).items():
        driver.memories[name].load_state_dict(st)
    composer = getattr(driver, "composer", None)
    stores = getattr(composer, "label_stores", None) or {}
    for name, st in state.get("label_stores", {}).items():
        stores[name].load_state_dict(st)
    print(
        f"[snapshot] resumed from {path} (step={driver.ctx.clock.step} "
        f"admitted={driver.ctx.clock.admitted})",
        flush=True,
    )
