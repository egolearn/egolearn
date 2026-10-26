"""Benchmark fixed BabyCL frame requests across video and JPEG loaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from egolearn.core.refs import ItemRef
from egolearn.data.decoding.pyav import PyAVDecoder
from egolearn.data.loading import Loader
from egolearn.data.sources.base import Stream
from egolearn.data.sources.video import VideoStream
from egolearn.data.views import ViewBank


class IndividualVideoStream(VideoStream):
    """Serve the same requests without grouping reads within a batch."""

    def open_frame_batch(self, requests: list[tuple[ItemRef, int]]) -> list:
        return [self.open_frame(ref, index) for ref, index in requests]


def _jpeg_name(ref: ItemRef, index: int) -> str:
    path, start, _end = ref.locator
    prefix = hashlib.sha256(path.encode()).hexdigest()[:20]
    return f"{prefix}-{start + index:09d}.jpg"


class JPEGReference(Stream):
    """Read a bounded JPEG cache for the benchmark's exact video frame requests."""

    def __init__(self, root: str):
        self.root = Path(root)

    def open_frame(self, ref: ItemRef, index: int):
        with Image.open(self.root / _jpeg_name(ref, index)) as image:
            return image.convert("RGB")


def sample_batches(
    stream: VideoStream,
    *,
    batches: int,
    sessions: int,
    segments: int,
    frames: int,
    seconds: float,
    seed: int,
) -> tuple[list[list[tuple]], list[str]]:
    """Sample stratified frame requests from windows spread across recordings."""
    ids = stream.session_ids()
    selected = [ids[i] for i in np.linspace(0, len(ids) - 1, min(sessions, len(ids)), dtype=int)]
    rng = random.Random(seed)
    loader = Loader(stream, ViewBank({}))
    result = []
    for _ in range(batches):
        batch = []
        chosen = (
            rng.sample(selected, segments)
            if len(selected) >= segments
            else rng.choices(selected, k=segments)
        )
        for sid in chosen:
            meta = stream._session_meta(sid)
            fps, count = float(meta["fps"]), int(meta["num_frames"])
            if count < 1:
                raise ValueError(f"empty video: {sid}")
            span = min(count, max(1, round(seconds * fps)))
            start = rng.randrange(count - span + 1)
            ref = ItemRef(
                sid,
                "video",
                start / fps,
                (start + span - 1) / fps,
                (str(stream.root / sid), start, start + span),
            )
            for index in loader.pick_frame_indices(ref, frames, rng):
                batch.append(("frame", ref, index, ["v1", "v2"], rng.randrange(2**31)))
        result.append(batch)
    return result, selected


def prepare_jpegs(batches: list[list[tuple]], root: Path, workers: int = 4) -> dict:
    """Decode requested frames once and save full-resolution JPEGs at quality 95."""
    root.mkdir(parents=True, exist_ok=False)
    grouped: dict[str, dict[int, tuple[ItemRef, int]]] = defaultdict(dict)
    for batch in batches:
        for _, ref, index, _views, _seed in batch:
            grouped[ref.locator[0]][ref.locator[1] + index] = (ref, index)

    def prepare(entry):
        path, requests = entry
        decoder = PyAVDecoder(cache_handles=1)
        try:
            ordered = sorted(requests)
            for offset in range(0, len(ordered), 32):
                picks = ordered[offset : offset + 32]
                images = decoder.read_window(
                    path, picks[0], picks[-1] + 1, [i - picks[0] for i in picks]
                )
                for index, image in zip(picks, images, strict=True):
                    ref, relative = requests[index]
                    image.save(root / _jpeg_name(ref, relative), quality=95, subsampling=0)
        finally:
            decoder.close()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(prepare, grouped.items()))
    return {
        "wall_s": time.perf_counter() - started,
        "frames": sum(len(requests) for requests in grouped.values()),
        "bytes": sum(path.stat().st_size for path in root.iterdir()),
        "quality": 95,
        "resolution": "original",
    }


def measure(
    stream: Stream, batches: list[list[tuple]], *, workers: int, warmup: int, size: int = 224
) -> dict:
    """Measure loader-call latency after warmup, recording a batch tensor digest."""
    import egolearn.methods.babycl  # noqa: F401

    views = ViewBank(
        {
            "v1": {"name": "babycl_ms_aug", "size": size, "view_idx": 0, "scale_min": 0.2},
            "v2": {"name": "babycl_ms_aug", "size": size, "view_idx": 1, "scale_min": 0.2},
        }
    )
    loader = Loader(stream, views, workers=workers, imports=["egolearn.methods.babycl"])
    latencies = []
    digest = hashlib.sha256()
    try:
        for number, batch in enumerate(batches):
            started = time.perf_counter()
            values = loader.views_batch(batch)
            latencies.append(time.perf_counter() - started)
            # Hash outside the timed call; video variants must yield identical tensors.
            if number == warmup:
                for value in values:
                    digest.update(value.numpy().tobytes())
    finally:
        loader.close()
    measured = latencies[warmup:]
    count = sum(len(batch) for batch in batches[warmup:])
    return {
        "workers": workers,
        "startup_batch_s": latencies[0],
        "batch_s": measured,
        "mean_batch_s": float(np.mean(measured)),
        "p50_batch_s": float(np.median(measured)),
        "p95_batch_s": float(np.quantile(measured, 0.95)),
        "frame_requests": count,
        "frames_per_s": count / sum(measured),
        "views_per_s": 2 * count / sum(measured),
        "first_measured_batch_sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--jpeg-reference", action="store_true")
    parser.add_argument("--device", default="cpu", help="PyAV decoder device: cpu or cuda:<index>")
    parser.add_argument("--cache-handles", type=int, default=8)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--skip-individual", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batches, args.warmup, args.sessions, args.segments, args.frames, args.size) < 1:
        parser.error("batch, warmup, session, segment, frame, and size counts must be positive")
    if min(args.workers) < 0 or args.seconds <= 0:
        parser.error("workers must be nonnegative and seconds positive")
    torch.set_num_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    stream = VideoStream(args.root, PyAVDecoder(), manifest_json=args.manifest)
    batches, selected = sample_batches(
        stream,
        batches=args.batches + args.warmup,
        sessions=args.sessions,
        segments=args.segments,
        frames=args.frames,
        seconds=args.seconds,
        seed=args.seed,
    )
    workload = [
        [[list(ref.locator), index, seed] for _, ref, index, _, seed in batch] for batch in batches
    ]
    workload_json = json.dumps(workload, indent=2)
    (output / "requests.json").write_text(workload_json + "\n")
    report: dict = {
        "status": "running",
        "hostname": platform.node(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "args": vars(args),
        "sessions": selected,
        "workload_sha256": hashlib.sha256(workload_json.encode()).hexdigest(),
        "scope": (
            f"{args.device} decode and CPU BabyCL augmentation; "
            "OS cache uncontrolled; JPEG is lossy"
        ),
        "variants": [],
    }

    def save():
        temp = output / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output / "report.json")

    save()
    if args.jpeg_reference:
        print("Preparing the bounded JPEG reference", flush=True)
        report["jpeg_preparation"] = prepare_jpegs(batches, output / "jpeg")
        save()
    variants = [] if args.skip_individual else [("video_individual", 0)]
    variants += [("video_grouped", workers) for workers in dict.fromkeys(args.workers)]
    if args.jpeg_reference:
        variants += [("jpeg", workers) for workers in dict.fromkeys(args.workers)]
    reference_digest = None
    for name, workers in variants:
        source: Stream
        if name == "jpeg":
            source = JPEGReference(str(output / "jpeg"))
        else:
            source_cls = IndividualVideoStream if name == "video_individual" else VideoStream
            source = source_cls(
                args.root,
                PyAVDecoder(
                    cache_handles=args.cache_handles, seek_gap=args.seek_gap, device=args.device
                ),
                manifest_json=args.manifest,
            )
        print(f"Measuring {name}, workers={workers}", flush=True)
        result = {
            "kind": name,
            **measure(source, batches, workers=workers, warmup=args.warmup, size=args.size),
        }
        if name.startswith("video"):
            digest = result["first_measured_batch_sha256"]
            if reference_digest is not None and digest != reference_digest:
                raise RuntimeError("video variants produced different augmented tensors")
            reference_digest = digest
        report["variants"].append(result)
        save()
        print(
            f"{name} workers={workers}: {result['frames_per_s']:.2f} frames/s, "
            f"p50={result['p50_batch_s']:.3f}s p95={result['p95_batch_s']:.3f}s",
            flush=True,
        )
    report["status"] = "completed"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
