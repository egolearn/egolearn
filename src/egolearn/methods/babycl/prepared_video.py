"""Prepare and read lossless videos after BabyCL's fixed spatial preprocessing."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

from egolearn.config.registry import DECODERS
from egolearn.data.decoding.pyav import PyAVDecoder


def _recipe(size: int, gop: int) -> dict:
    import av
    import PIL
    import torchvision

    return {
        "version": 1,
        "size": size,
        "gop": gop,
        "transform": "PIL RGB -> Resize(size, BICUBIC) -> CenterCrop(size)",
        "codec": "libx264rgb",
        "crf": 0,
        "pillow": PIL.__version__,
        "torchvision": torchvision.__version__,
        "av": av.__version__,
        "ffmpeg": {k: list(v) for k, v in av.library_versions.items()},
    }


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _save(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def _prepare_one(source: Path, output: Path, recipe: dict) -> dict:
    import av
    import torch
    import torchvision.transforms as T

    torch.set_num_threads(1)
    started = time.perf_counter()
    sidecar = output.with_name(output.name + ".json")
    source_stat = _fingerprint(source)
    if output.exists() and sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if (
            previous["recipe"] == recipe
            and previous["source"] == source_stat
            and previous["output"] == _fingerprint(output)
        ):
            return previous
        raise ValueError(f"stale cache entry; use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + ".partial.mp4")
    size = recipe["size"]
    transform = T.Compose(
        [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
    )
    expected = hashlib.sha256()
    count = 0
    try:
        with av.open(str(source)) as inp, av.open(str(temp), "w") as out:
            src = inp.streams.video[0]
            src.codec_context.thread_count = 1
            fps = src.average_rate
            if not fps or not src.time_base or src.frames <= 0:
                raise ValueError(f"constant-rate video metadata required: {source}")
            dst = out.add_stream("libx264rgb", rate=fps)
            dst.width = dst.height = size
            dst.pix_fmt = "rgb24"
            dst.codec_context.thread_count = 1
            dst.codec_context.gop_size = recipe["gop"]
            dst.codec_context.max_b_frames = 0
            dst.options = {"crf": "0", "preset": "ultrafast", "sc_threshold": "0"}
            tick = src.time_base * fps
            origin = src.start_time or 0
            for frame in inp.decode(src):
                if frame.pts is None or abs((frame.pts - origin) * tick - count) > max(tick, 0.001):
                    raise ValueError(f"non-constant frame timeline: {source}, frame {count}")
                image = transform(frame.to_image())
                expected.update(image.tobytes())
                encoded = av.VideoFrame.from_image(image)
                encoded.pts = count
                encoded.time_base = 1 / fps
                for packet in dst.encode(encoded):
                    out.mux(packet)
                count += 1
            for packet in dst.encode():
                out.mux(packet)
            if count != src.frames:
                raise ValueError(f"decoded frame count differs from metadata: {source}")

        actual = hashlib.sha256()
        verified = 0
        with av.open(str(temp)) as check:
            video = check.streams.video[0]
            video.codec_context.thread_count = 1
            if video.frames != count or video.average_rate != fps:
                raise ValueError(f"prepared timeline differs: {source}")
            for frame in check.decode(video):
                actual.update(frame.to_image().tobytes())
                if frame.pts is None or frame.pts * video.time_base * fps != verified:
                    raise ValueError(f"prepared frame timestamp differs: {source}")
                verified += 1
        if verified != count or actual.digest() != expected.digest():
            raise ValueError(f"lossless RGB verification failed: {source}")
        if _fingerprint(source) != source_stat:
            raise ValueError(f"source changed during preparation: {source}")
        temp.replace(output)
        result = {
            "recipe": recipe,
            "source": source_stat,
            "output": _fingerprint(output),
            "num_frames": count,
            "fps": [fps.numerator, fps.denominator],
            "rgb_sha256": actual.hexdigest(),
            "wall_s": time.perf_counter() - started,
        }
        _save(sidecar, result)
        return result
    finally:
        temp.unlink(missing_ok=True)


def prepare(
    source_root: str,
    prepared_root: str,
    *,
    sessions: list[str] | None = None,
    workers: int = 1,
    size: int = 224,
    gop: int = 8,
) -> dict:
    """Convert constant-rate videos and verify every output RGB frame."""
    import multiprocessing as mp

    source, output = Path(source_root).resolve(), Path(prepared_root).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and prepared directories must be separate, non-nested trees")
    if min(workers, size, gop) < 1:
        raise ValueError("workers, size, and gop must be positive")
    names = sessions if sessions is not None else sorted(p.name for p in source.glob("*.mp4"))
    if not names or len(set(names)) != len(names):
        raise ValueError("provide a nonempty list of distinct video sessions")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"session must be a relative path within the source root: {name}")
        if not (source / name).resolve().is_relative_to(source):
            raise ValueError(f"session points outside the source root: {name}")
    output.mkdir(parents=True, exist_ok=True)
    recipe = _recipe(size, gop)
    with (output / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / "cache.json"
        if config.exists() and json.loads(config.read_text()) != recipe:
            raise ValueError("cache recipe differs; use a new output directory")
        _save(config, recipe)
        started = time.perf_counter()
        entries = {}
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {
                pool.submit(_prepare_one, source / name, output / name, recipe): name
                for name in names
            }
            for future in as_completed(futures):
                name = futures[future]
                entries[name] = future.result()
                print(f"verified {len(entries)}/{len(names)}: {name}", flush=True)
        report = {
            "status": "completed",
            "sessions": names,
            "num_frames": sum(e["num_frames"] for e in entries.values()),
            "bytes": sum(e["output"]["bytes"] for e in entries.values()),
            "wall_s": time.perf_counter() - started,
        }
        _save(output / "preparation.json", report)
        return report


@DECODERS.register("babycl_prepared_video")
class PreparedVideoDecoder(PyAVDecoder):
    """Resolve original video refs to verified, spatially prepared lossless videos.

    Only BabyCL's matching ``babycl_ms_aug`` and unflipped ``center_crop`` views
    are supported. Original paths remain in stream metadata and replay snapshots.
    Source and output fingerprints are checked when each file is first opened.
    """

    def __init__(
        self,
        source_root: str,
        prepared_root: str,
        cache_handles: int = 8,
        seek_gap: int = 8,
        threads: int = 1,
        device: str = "cpu",
    ):
        if device != "cpu":
            raise ValueError("lossless RGB prepared videos require CPU decoding")
        super().__init__(cache_handles, seek_gap, threads, device)
        self.source_root = Path(source_root).resolve()
        self.prepared_root = Path(prepared_root).resolve()
        self.recipe = json.loads((self.prepared_root / "cache.json").read_text())
        if self.recipe != _recipe(self.recipe["size"], self.recipe["gop"]):
            raise ValueError("cache preprocessing or library versions differ; rebuild the cache")
        self._validated: dict[str, dict] = {}

    def validate_views(self, specs: dict[str, dict]) -> None:
        for alias, spec in specs.items():
            if (
                spec.get("name") not in {"babycl_ms_aug", "center_crop"}
                or spec.get("size", 224) != self.recipe["size"]
                or spec.get("vertical_flip", False)
            ):
                raise ValueError(f"view {alias!r} is incompatible with the prepared video cache")

    def _open(self, path: str, *, metadata_only: bool = False):
        source = Path(path).resolve()
        name = source.relative_to(self.source_root)
        output = self.prepared_root / name
        if path not in self._validated:
            record = json.loads(output.with_name(output.name + ".json").read_text())
            if (
                record["recipe"] != self.recipe
                or record["source"] != _fingerprint(source)
                or record["output"] != _fingerprint(output)
            ):
                raise ValueError(f"stale or modified prepared video: {output}")
            self._validated[path] = record
        handle = super()._open(str(output), metadata_only=metadata_only)
        record = self._validated[path]
        if (
            handle.stream.frames != record["num_frames"]
            or handle.stream.average_rate != Fraction(*record["fps"])
            or (handle.stream.width, handle.stream.height) != (self.recipe["size"],) * 2
        ):
            handle.container.close()
            raise ValueError(f"prepared video metadata differs: {output}")
        return handle

    def __getstate__(self) -> dict:
        return {**super().__getstate__(), "_validated": {}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--gop", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(**vars(args)), indent=2), flush=True)
    return 0
