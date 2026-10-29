"""Prepare the original WalkingTours videos as ordered 3.75 FPS JPEG sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from PIL import Image


def prepare_video(source_root: Path, output_root: Path, video: dict, threads: int) -> dict:
    """Extract one complete session, publishing it only after validation."""
    source = source_root / video["filename"]
    stat = source.stat()
    if stat.st_size != video["bytes"]:
        raise ValueError(f"incomplete source video: {source}")
    ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    identity = {
        "source": video,
        "source_mtime_ns": stat.st_mtime_ns,
        "fps": 3.75,
        "width": 1280,
        "height": 720,
        "jpeg_quality": 2,
        "sampling": "fps=15/4:round=near",
        "ffmpeg": ffmpeg_version,
    }
    output = output_root / video["session"]
    if output.exists():
        record = json.loads((output / "preparation.json").read_text())
        if record["identity"] != identity:
            raise ValueError(f"prepared session differs: {output}")
        count = len(list(output.glob("frame_*.jpg")))
        if count != record["frames"]:
            raise ValueError(f"prepared session is incomplete: {output}")
        return record
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "md5").hexdigest() != video["md5"]:
            raise ValueError(f"source checksum differs from manifest: {source}")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,duration:format=duration",
                "-of",
                "json",
                str(source),
            ]
        )
    )
    stream = probe["streams"][0]
    duration = float(stream.get("duration", probe["format"]["duration"]))
    source_fps = float(Fraction(stream["avg_frame_rate"]))
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{video['session']}-", dir=output_root) as temp:
        staging = Path(temp)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-threads",
            str(threads),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "fps=15/4:round=near,scale=1280:720",
            "-filter_threads",
            "1",
            "-threads",
            "1",
            "-q:v",
            "2",
            "-start_number",
            "1",
            str(staging / "frame_%08d.jpg"),
        ]
        subprocess.run(command, check=True)
        files = sorted(staging.glob("frame_*.jpg"))
        if not files or abs(len(files) - duration * 3.75) > 2:
            raise ValueError(f"frame count disagrees with source duration: {source}")
        for index, path in enumerate(files, 1):
            if path.name != f"frame_{index:08d}.jpg" or path.stat().st_size == 0:
                raise ValueError(f"invalid extracted frame: {path}")
        for path in (files[0], files[-1]):
            with Image.open(path) as image:
                if image.size != (1280, 720):
                    raise ValueError(f"wrong frame geometry: {path}")
                image.verify()
        if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
            raise ValueError(f"source changed during preparation: {source}")
        record = {
            "identity": identity,
            "frames": len(files),
            "source_fps": source_fps,
            "duration_seconds": duration,
            "source_geometry": [stream["height"], stream["width"]],
        }
        (staging / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
        staging.rename(output)
    print(f"Prepared {video['session']}: {record['frames']} frames", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    manifest = json.loads(args.source_manifest.read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(
            pool.map(
                lambda video: prepare_video(
                    args.source_root, args.output_root, video, args.threads
                ),
                manifest["videos"],
            )
        )
    inventory = {
        "source_doi": manifest["doi"],
        "fps": 3.75,
        "sessions": [v["session"] for v in manifest["videos"]],
        "total_frames": sum(record["frames"] for record in records),
        "duration_seconds": sum(record["duration_seconds"] for record in records),
        "videos": records,
    }
    path = args.output_root / "inventory.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(inventory, indent=2) + "\n")
    temporary.replace(path)
    print(
        json.dumps({key: value for key, value in inventory.items() if key != "videos"}), flush=True
    )


if __name__ == "__main__":
    main()
