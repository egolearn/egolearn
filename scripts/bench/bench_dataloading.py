#!/usr/bin/env python
"""Benchmark of the data side only: how fast can the framework FEED a training run?

No model forward/backward runs here -- this isolates the dataloading path, so the
result answers "is the data pipeline the bottleneck?" independent of GPU speed.
It runs the real BabyCL data logic end to end, in three phases:

  phase 1  stream + admission   observations -> segmenter/AN-chunk emitters
                                -> routed events -> tiered memories
                                (includes the frozen-embedder work the online
                                segmenter does -- that IS part of the ingest cost)
  phase 2  batch composition    recipe sampling -> decode -> augment -> collate,
                                the exact loading path taken for every gradient
                                step (composer.compose, the same call the driver
                                makes)
  phase 3  micro breakdown      per-frame decode time vs. per-view augmentation
                                time, so a slow phase 2 can be attributed to one
                                or the other

Usage (defaults match the BabyCL default run):

    python scripts/bench/bench_dataloading.py \
        --config projects/babycl/configs/saycam_default.yaml \
        --batches 50 --device cpu

    # realistic warmup cost (fills segments.short to capacity, slow):
    python scripts/bench/bench_dataloading.py --config ... --full-warmup

Interpreting phase 2: with `rate.resample=true` every gradient step composes one
fresh batch, so `1000 / mean-ms-per-batch` is the ceiling on steps/s the data
pipeline can sustain single-threaded.  Compare against the model step time from
scripts/bench/bench_full_pipeline.py (reference: the original BabyCL run holds
~70% GPU utilization on 2x A100).
"""

from __future__ import annotations

import argparse
import json
import time

import egolearn
from egolearn.config import load_config
from egolearn.core import keys as K


def mem_count(mem) -> int:
    return sum(len(t) for t in mem.tiers.values())


def mem_capacity(mem) -> int:
    return sum(t.capacity for t in mem.tiers.values())


def required_items(recipe, fill_factor: int):
    """How many items each memory needs before phase 2 can fill every strict slot:
    the largest slot.n over the recipe (and its fallback), times a small
    oversampling factor so successive batches are not near-copies of each other."""
    need = {}
    r = recipe
    while r is not None:
        for slot in r.slots:
            src = slot.source
            if src.get("kind") in ("tier", "mix") and src.get("memory"):
                m = src["memory"]
                need[m] = max(need.get(m, 0), slot.n)
        r = r.fallback
    return {m: n * fill_factor for m, n in need.items()}


def batch_counts(batch):
    """(decoded frames, augmented view tensors) in one composed batch."""
    decoded = augmented = 0
    for slot in batch.values():
        v = slot["views"]
        if v.dim() == 6:  # clip slot [B, V, T, C, H, W]
            decoded += v.shape[0] * v.shape[2]
            augmented += v.shape[0] * v.shape[1] * v.shape[2]
        else:  # frame slot [B, V, C, H, W]
            decoded += v.shape[0]
            augmented += v.shape[0] * v.shape[1]
    return decoded, augmented


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--batches", type=int, default=50, help="batches to compose in phase 2")
    ap.add_argument(
        "--fill-factor",
        type=int,
        default=4,
        help="phase 1 stops once every memory holds fill_factor * (largest slot n) items",
    )
    ap.add_argument(
        "--full-warmup",
        action="store_true",
        help="instead run the config's real warmup condition "
        "(BabyCL: segments.short full) -- slow but realistic",
    )
    ap.add_argument(
        "--max-obs", type=int, default=500_000, help="phase 1 safety cap on observations"
    )
    ap.add_argument(
        "--micro-repeats", type=int, default=20, help="repeats per phase 3 micro measurement"
    )
    ap.add_argument(
        "--device",
        default=None,
        help="override cfg.device (embedder compute; use cpu on a login node)",
    )
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # data plane only: probes would instantiate eval benchmarks we never run
    cfg.pop("probes", None)
    cfg["hooks"] = {}
    if args.device:
        cfg["device"] = args.device

    pipe = egolearn.build(cfg)
    if pipe.mode != "streaming":
        raise SystemExit(
            "bench_dataloading measures the streaming data plane; "
            "for offline configs use bench_full_pipeline.py"
        )
    print(pipe.summary(), flush=True)
    drv = pipe.driver
    timer = drv.timer  # loader.timer is already attached to this

    report = {}

    # ---- phase 1: stream + admission -> memories ---------------------------
    if args.full_warmup:
        target_str = f"config warmup ({drv.warmup})"
        done = drv._warmup_done
    else:
        need = required_items(drv.recipe, args.fill_factor)
        need = {m: min(n, mem_capacity(drv.memories[m])) for m, n in need.items()}
        target_str = f"items per memory: {need}"
        done = lambda: all(mem_count(drv.memories[m]) >= n for m, n in need.items())

    print(f"\n=== phase 1: stream + admission (until {target_str}) ===", flush=True)
    timer.reset()
    events_by_schema = {}
    t0 = time.perf_counter()
    while not done():
        if drv.ctx.clock.admitted >= args.max_obs:
            print(
                f"[phase 1] hit --max-obs={args.max_obs} before targets; "
                "results below are for a partially filled memory"
            )
            break
        with timer.measure("stream.advance"):
            events = drv._step_stream()
        if events is None:
            print("[phase 1] stream exhausted")
            break
        for ev in events:
            s = ev[K.EVENT_SCHEMA]
            events_by_schema[s] = events_by_schema.get(s, 0) + 1
    wall1 = time.perf_counter() - t0
    obs = drv.ctx.clock.admitted

    rate = obs / max(wall1, 1e-9)
    fps = float(cfg["stream"].get("fps", 5.0))
    print(f"observations : {obs}  ({rate:.1f} obs/s = {rate / fps:.2f}x realtime at {fps} fps)")
    print("events       : %s" % (events_by_schema or "{}"))
    print("memories     : {}".format(" | ".join(m.stats() for m in drv.memories.values())))
    print(timer.summary())
    report["phase1_ingest"] = {
        "wall_s": wall1,
        "observations": obs,
        "obs_per_s": obs / max(wall1, 1e-9),
        "events": events_by_schema,
        "stages": timer.report(),
    }

    # ---- phase 2: batch composition (the per-step loading path) ------------
    print(f"\n=== phase 2: batch composition x {args.batches} ===", flush=True)
    timer.reset()
    decoded = augmented = composed = fell_back = 0
    slot_shapes = {}
    t0 = time.perf_counter()
    for _ in range(args.batches):
        with timer.measure("batch.compose"):
            batch = drv.composer.compose(
                drv.recipe, drv.memories, {s: list(d) for s, d in drv.stream_head.items()}, drv.ctx
            )
        if batch is None:
            continue
        composed += 1
        if set(batch) != {s.name for s in drv.recipe.slots}:
            fell_back += 1
        d, a = batch_counts(batch)
        decoded += d
        augmented += a
        for name, slot in batch.items():
            slot_shapes[name] = tuple(slot["views"].shape)
    wall2 = time.perf_counter() - t0

    if composed == 0:
        raise SystemExit(
            "[phase 2] no batch composed -- memories too empty? "
            "raise --fill-factor or check the stream paths"
        )
    ms = 1000.0 * wall2 / composed
    shapes = {n: "x".join(map(str, s)) for n, s in slot_shapes.items()}
    print(f"batches      : {composed} composed ({fell_back} via fallback recipe)  slots: {shapes}")
    print(f"per batch    : {ms:.1f} ms  ->  {1000.0 / ms:.2f} batches/s")
    print(
        f"frames       : decoded {decoded / max(wall2, 1e-9):.1f}/s   "
        f"augmented views {augmented / max(wall2, 1e-9):.1f}/s"
    )
    print(
        f"feed ceiling : {1000.0 / ms:.2f} gradient steps/s single-threaded "
        f"(rate.resample={drv.rate.get('resample')}: one fresh batch per step)"
    )
    print(timer.summary())
    report["phase2_compose"] = {
        "wall_s": wall2,
        "batches": composed,
        "fallback_batches": fell_back,
        "ms_per_batch": ms,
        "decoded_frames_per_s": decoded / max(wall2, 1e-9),
        "augmented_views_per_s": augmented / max(wall2, 1e-9),
        "slot_shapes": {n: list(s) for n, s in slot_shapes.items()},
        "stages": timer.report(),
    }

    # ---- phase 3: micro breakdown ------------------------------------------
    print(
        f"\n=== phase 3: micro (decode vs. augment, {args.micro_repeats} repeats) ===", flush=True
    )
    import random as _random

    rng = _random.Random(0)
    entry = None
    for mem in drv.memories.values():
        for t in mem.tiers.values():
            if len(t):
                entry = t.entries[0]
                break
        if entry is not None:
            break
    micro = {}
    if entry is None:
        print("(no memory entries -- skipped)")
    else:
        ref = entry[K.VIDEO_REF]
        idx = pipe.loader.pick_frame_indices(ref, 1, rng)[0]
        t0 = time.perf_counter()
        for _ in range(args.micro_repeats):
            pil = pipe.stream.open_frame(ref, idx)
        micro["decode"] = 1000.0 * (time.perf_counter() - t0) / args.micro_repeats
        for vname in pipe.views.names():
            t0 = time.perf_counter()
            for _ in range(args.micro_repeats):
                pipe.views.apply(pil, [vname])
            micro["aug:" + vname] = 1000.0 * (time.perf_counter() - t0) / args.micro_repeats
        width = max(len(k) for k in micro)
        for k, v in micro.items():
            print(f"  {k:<{width}} : {v:6.2f} ms/frame")
        print(
            "  (decode timings are warm-cache after the first read; "
            "cold NFS/squashfs reads will be slower)"
        )
    report["phase3_micro_ms"] = micro

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[report] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
