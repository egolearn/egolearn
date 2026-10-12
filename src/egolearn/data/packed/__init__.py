"""Packed frame store: pre-extracted frames concatenated into a few large shard
files plus a numpy offset index, so random access costs one `pread` per frame.

Why it exists.  The frame-directory adapters read one file per frame.  That is
fine when the page cache holds the frames, and poor when it does not: cold
random reads through a FUSE-mounted squashfs serialise in the daemon
(~200-1,000 frames/s however many workers ask), and per-file JPEGs on a
network filesystem pay a lookup per open and need millions of inodes.  Replay
in streaming and the first epoch in offline training are exactly that cold
random pattern.  Packed into ~256 MB shards on the same network scratch, the
same frames read at the decode ceiling (several thousand frames/s per core
pool), and a contiguous clip is one read instead of T opens.

Layout of a store directory (built by `pack_frame_dirs` or
`python -m egolearn.data.packed pack`):

    manifest.json   format tag, the source's fps/naming, the shard list and the
                    session table (id, first global frame, count) in stream order
    index.npy       int64 [N, 3]: (shard, byte offset, byte length) per frame
    times.npy       float64 [N]: media time (seconds) of each frame
    names.npy       bytes [N]: the source frame file names (storyboards and
                    other precomputed indexes are validated against them)
    shards/*.bin    the encoded bytes of consecutive frames, back to back

Frames keep their original encoded bytes, so a store decodes to exactly the
pixels of the frame directory it was built from, and frame i of a session is
the same frame in both layouts: refs minted by `stream:saycam` and by
`stream:packed_frames` (sources/packed.py) are interchangeable.  The index is
memory-mapped and only the store's path is pickled, so handing a store to
Loader workers costs nothing and every worker shares the same index pages.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT = "egolearn.packed_frames"
VERSION = 1
MANIFEST = "manifest.json"
INDEX = "index.npy"
TIMES = "times.npy"
NAMES = "names.npy"
SHARD_DIR = "shards"
DEFAULT_SHARD_BYTES = 256 << 20
# ranges closer than this are served by one read: cheaper than a second
# round trip on a network filesystem, and a clip with stride > 1 stays one read
DEFAULT_MERGE_GAP = 256 << 10


@dataclass(frozen=True)
class SessionSpan:
    id: str
    first: int  # global index of the session's first frame
    count: int


def decode_frame(data: bytes) -> Image.Image:
    """Encoded frame bytes -> RGB PIL image (the same decode as the frame-dir path)."""
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
class PackedFrameStore:
    """Read side of a store directory.

    Everything is opened lazily on first use, and pickling keeps only the root
    path, so a store travels to worker processes as a few bytes and each worker
    maps the index from the page cache.  Shard file descriptors stay open in a
    per-process LRU: a store has one shard file per session chunk (about 1,300
    for the SAYCam S child), and reopening one costs a network round trip, so
    the cap defaults to a quarter of the process's open-file limit (at least
    64, at most 4,096), which keeps every shard open under a job step's limit
    and degrades to an LRU under an interactive shell's 1024.
    """

    def __init__(self, root: str | os.PathLike, max_open_shards: int | None = None):
        self.root = os.fspath(root)
        self.max_open_shards = max(1, int(max_open_shards or _default_open_shards()))
        self._reset()

    def _reset(self) -> None:
        self._manifest: dict | None = None
        self._index: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._names: np.ndarray | None = None
        self._spans: dict[str, SessionSpan] | None = None
        self._fds: OrderedDict[int, int] = OrderedDict()

    def __getstate__(self) -> dict:
        return {"root": self.root, "max_open_shards": self.max_open_shards}

    def __setstate__(self, state: dict) -> None:
        self.root = state["root"]
        self.max_open_shards = state["max_open_shards"]
        self._reset()

    def __del__(self) -> None:
        # fallback for stores that are dropped without close(): release the fds
        for fd in getattr(self, "_fds", {}).values():
            with contextlib.suppress(OSError):
                os.close(fd)

    # ---- metadata ---------------------------------------------------------
    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            path = os.path.join(self.root, MANIFEST)
            try:
                with open(path) as f:
                    m = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{self.root!r} is not a packed frame store (no {MANIFEST}); build one "
                    "with `python -m egolearn.data.packed pack <frame_root> <store>`"
                ) from None
            if m.get("format") != FORMAT or int(m.get("version", -1)) != VERSION:
                raise ValueError(
                    f"{path}: expected {FORMAT} v{VERSION}, found "
                    f"{m.get('format')!r} v{m.get('version')!r}"
                )
            self._manifest = m
        return self._manifest

    @property
    def fps(self) -> float:
        return float(self.manifest["fps"])

    @property
    def naming(self) -> str:
        return str(self.manifest["naming"])

    def __len__(self) -> int:
        return int(self.manifest["frame_count"])

    @property
    def index(self) -> np.ndarray:
        if self._index is None:
            arr = np.load(os.path.join(self.root, INDEX), mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != len(self):
                raise ValueError(f"{INDEX}: shape {arr.shape} does not match the manifest")
            self._index = arr
        return self._index

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            arr = np.load(os.path.join(self.root, TIMES), mmap_mode="r")
            if arr.shape != (len(self),):
                raise ValueError(f"{TIMES}: shape {arr.shape} does not match the manifest")
            self._times = arr
        return self._times

    @property
    def names(self) -> np.ndarray:
        if self._names is None:
            arr = np.load(os.path.join(self.root, NAMES), mmap_mode="r")
            if arr.shape != (len(self),):
                raise ValueError(f"{NAMES}: shape {arr.shape} does not match the manifest")
            self._names = arr
        return self._names

    def sessions(self) -> list[SessionSpan]:
        """Sessions in the order they were packed (the source stream's order)."""
        return [
            SessionSpan(s["id"], int(s["first"]), int(s["count"]))
            for s in self.manifest["sessions"]
        ]

    def span(self, sid: str) -> SessionSpan:
        if self._spans is None:
            self._spans = {s.id: s for s in self.sessions()}
        try:
            return self._spans[sid]
        except KeyError:
            raise KeyError(f"session {sid!r} is not in the packed store {self.root!r}") from None

    def session_times(self, sid: str) -> np.ndarray:
        s = self.span(sid)
        return np.asarray(self.times[s.first : s.first + s.count])

    def session_names(self, sid: str) -> list[str]:
        """The source frame file names of one session, in frame order."""
        s = self.span(sid)
        return [n.decode() for n in self.names[s.first : s.first + s.count]]

    def empty_sessions(self) -> list[str]:
        """Sessions that were selected at build time but held no frames."""
        return list(self.manifest.get("empty_sessions", []))

    # ---- bytes ------------------------------------------------------------
    def _fd(self, shard: int) -> int:
        fd = self._fds.get(shard)
        if fd is not None:
            self._fds.move_to_end(shard)
            return fd
        path = os.path.join(self.root, self.manifest["shards"][shard]["path"])
        fd = os.open(path, os.O_RDONLY)
        self._fds[shard] = fd
        while len(self._fds) > self.max_open_shards:
            _, old = self._fds.popitem(last=False)
            os.close(old)
        return fd

    def _pread(self, shard: int, offset: int, length: int) -> bytes:
        data = os.pread(self._fd(shard), length, offset)
        if len(data) != length:
            path = self.manifest["shards"][shard]["path"]
            raise OSError(f"short read from {path} at {offset}: {len(data)} of {length} bytes")
        return data

    def _row(self, index: int) -> tuple[int, int, int]:
        n = len(self)
        if not 0 <= index < n:
            raise IndexError(f"frame {index} outside the packed store (0..{n - 1})")
        shard, offset, length = (int(v) for v in self.index[index])
        return shard, offset, length

    def read(self, index: int) -> bytes:
        """Encoded bytes of global frame `index`: one pread."""
        return self._pread(*self._row(index))

    def read_many(self, indices: Sequence[int], merge_gap: int = DEFAULT_MERGE_GAP) -> list[bytes]:
        """Encoded bytes of several frames, in the requested order (repeats
        allowed).  Frames that sit in the same shard within `merge_gap` bytes of
        each other are fetched by one read, so a contiguous clip is one pread."""
        if len(indices) == 0:
            return []
        out: dict[int, bytes] = {}
        run: list[tuple[int, int, int, int]] = []  # (index, shard, offset, length)

        def flush() -> None:
            shard = run[0][1]
            start = run[0][2]
            end = run[-1][2] + run[-1][3]
            blob = self._pread(shard, start, end - start)
            for i, _, off, n in run:
                out[i] = blob[off - start : off - start + n]
            run.clear()

        for i in sorted({int(i) for i in indices}):
            shard, offset, length = self._row(i)
            if run and (shard != run[-1][1] or offset - (run[-1][2] + run[-1][3]) > merge_gap):
                flush()
            run.append((i, shard, offset, length))
        if run:
            flush()
        return [out[int(i)] for i in indices]

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _pack_session(
    root: str, out: str, sid: str, files: list[str], shard_bytes: int
) -> tuple[str, list[tuple[str, int]], np.ndarray]:
    """Append one session's frames to new shard files (`shards/<sid>.<k>.bin`),
    starting a new shard once the current one would exceed `shard_bytes`.
    Returns (sid, [(shard relpath, bytes)], int64 [n, 3] (local shard, offset, length))."""
    shards: list[tuple[str, int]] = []
    rows: list[tuple[int, int, int]] = []
    k = -1
    cur = None
    cur_rel = ""
    cur_size = 0
    try:
        for fn in files:
            with open(os.path.join(root, sid, fn), "rb") as f:
                data = f.read()
            if cur is None or (cur_size > 0 and cur_size + len(data) > shard_bytes):
                if cur is not None:
                    cur.close()
                    shards.append((cur_rel, cur_size))
                k += 1
                cur_rel = f"{SHARD_DIR}/{sid}.{k:03d}.bin"
                # the handle outlives this iteration; closed on shard change and in finally
                cur = open(os.path.join(out, cur_rel), "wb", buffering=1 << 20)  # noqa: SIM115
                cur_size = 0
            cur.write(data)
            rows.append((k, cur_size, len(data)))
            cur_size += len(data)
    finally:
        if cur is not None:
            cur.close()
    if cur is not None:
        shards.append((cur_rel, cur_size))
    return sid, shards, np.array(rows, dtype=np.int64).reshape(-1, 3)


def _default_open_shards() -> int:
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):  # pragma: no cover - non-POSIX
        return 64
    return max(64, min(4096, soft // 4))


def _build_files(out: Path) -> list[Path]:
    """Everything a build (complete or interrupted) may have left under `out`."""
    files = [out / name for name in (MANIFEST, INDEX, TIMES, NAMES) if (out / name).exists()]
    files += sorted((out / SHARD_DIR).glob("*.bin")) if (out / SHARD_DIR).is_dir() else []
    return files


def pack_frame_dirs(
    root: str | os.PathLike,
    out: str | os.PathLike,
    *,
    include: str | None = None,
    sessions: list[str] | None = None,
    naming: str = "timestamp",
    fps: float = 5.0,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    workers: int = 0,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Build a packed store under `out` from the frame directories under `root`.

    Sessions, their order and each session's frame order are exactly what
    `stream:saycam` would produce for the same `root`/`include`/`sessions`/
    `naming`/`fps` arguments (the listing rules live in SaycamStream), so the
    refs of both adapters agree.  Each session is packed by one process
    (`workers` > 0 fans sessions out to a process pool); the manifest is
    written last, so a store without one is incomplete.  Returns the manifest.
    """
    from egolearn.data.sources.saycam import SaycamStream

    root_p, out_p = Path(root), Path(out)
    leftovers = _build_files(out_p)
    if leftovers:
        if not force:
            raise FileExistsError(
                f"{out_p} already holds a packed store or an interrupted build (use force=True)"
            )
        for path in leftovers:  # only files a build of this store writes
            path.unlink()
    src = SaycamStream(str(root_p), fps=fps, naming=naming, sessions=sessions, include=include)
    listed = [(sid, src.frame_files(sid)) for sid in src.session_ids()]
    jobs = [(sid, files) for sid, files in listed if files]
    empty = [sid for sid, files in listed if not files]
    if not jobs:
        raise ValueError(f"no frame directories to pack under {root_p}")
    (out_p / SHARD_DIR).mkdir(parents=True, exist_ok=True)
    say = log or (lambda _msg: None)
    say(f"packing {len(jobs)} sessions from {root_p} into {out_p}")

    args = [(str(root_p), str(out_p), sid, files, int(shard_bytes)) for sid, files in jobs]
    t0 = time.perf_counter()
    if workers > 0:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            results = pool.map(_pack_session, *zip(*args, strict=True))
            packed = list(_progress(results, len(jobs), say, t0))
    else:
        packed = list(_progress(map(_pack_session, *zip(*args, strict=True)), len(jobs), say, t0))

    shards: list[dict] = []
    session_table: list[dict] = []
    index_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    names: list[bytes] = []
    first = 0
    for (sid, files), (sid2, session_shards, rows) in zip(jobs, packed, strict=True):
        assert sid == sid2
        base = len(shards)
        shards.extend(
            {"path": rel, "bytes": nbytes, "session": sid} for rel, nbytes in session_shards
        )
        rows = rows.copy()
        rows[:, 0] += base
        index_parts.append(rows)
        time_parts.append(np.asarray(src.frame_times(files), dtype=np.float64))
        names.extend(fn.encode() for fn in files)
        session_table.append({"id": sid, "first": first, "count": int(rows.shape[0])})
        first += int(rows.shape[0])

    index = np.concatenate(index_parts) if index_parts else np.zeros((0, 3), dtype=np.int64)
    times = np.concatenate(time_parts) if time_parts else np.zeros((0,), dtype=np.float64)
    np.save(out_p / INDEX, index)
    np.save(out_p / TIMES, times)
    np.save(out_p / NAMES, np.array(names, dtype="S") if names else np.zeros(0, dtype="S1"))
    manifest: dict = {
        "format": FORMAT,
        "version": VERSION,
        "fps": float(fps),
        "naming": naming,
        "frame_count": int(index.shape[0]),
        "total_bytes": int(sum(s["bytes"] for s in shards)),
        "source_root": str(root_p),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shards": shards,
        "sessions": session_table,
        "empty_sessions": empty,
    }
    tmp = out_p / (MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, out_p / MANIFEST)
    say(
        f"done: {manifest['frame_count']} frames, {len(shards)} shards, "
        f"{manifest['total_bytes'] / 1e9:.2f} GB in {time.perf_counter() - t0:.0f} s"
    )
    return manifest


def _progress(results, total: int, say: Callable[[str], None], t0: float):
    for i, item in enumerate(results, 1):
        sid, session_shards, rows = item
        say(
            f"[{i}/{total}] {sid}: {rows.shape[0]} frames, "
            f"{sum(n for _, n in session_shards) / 1e6:.1f} MB, {time.perf_counter() - t0:.0f} s"
        )
        yield item


def verify_store(
    store_root: str | os.PathLike,
    frame_root: str | os.PathLike,
    *,
    sample: int | None = None,
    seed: int = 0,
) -> int:
    """Check a store against the frame directories it was built from: the same
    sessions with the same frame names and times, and byte-identical frames
    (all of them, or a random `sample`).  Raises ValueError on the first
    mismatch; returns the number of frames compared."""
    from egolearn.data.sources.saycam import SaycamStream

    store = PackedFrameStore(store_root)
    spans = store.sessions()
    src = SaycamStream(
        os.fspath(frame_root), fps=store.fps, naming=store.naming, sessions=[s.id for s in spans]
    )
    checks: list[tuple[int, str]] = []
    for s in spans:
        files = src.frame_files(s.id)
        if len(files) != s.count:
            raise ValueError(f"session {s.id}: {len(files)} frames on disk, {s.count} packed")
        if files != store.session_names(s.id):
            raise ValueError(f"session {s.id}: frame names or order differ from the packed store")
        if not np.array_equal(store.session_times(s.id), np.asarray(src.frame_times(files))):
            raise ValueError(f"session {s.id}: frame times differ from the packed store")
        checks.extend(
            (s.first + i, os.path.join(os.fspath(frame_root), s.id, fn))
            for i, fn in enumerate(files)
        )
    if sample is not None and sample < len(checks):
        checks = sorted(random.Random(seed).sample(checks, sample))
    try:
        for g, path in checks:
            with open(path, "rb") as f:
                expected = f.read()
            if store.read(g) != expected:
                raise ValueError(f"frame {g} ({path}) differs from the packed bytes")
    finally:
        store.close()
    return len(checks)


# ---------------------------------------------------------------------------
# command line: python -m egolearn.data.packed {pack,verify,info}
# ---------------------------------------------------------------------------
def _parse_bytes(text: str) -> int:
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    t = text.strip().upper()
    if t and t[-1] in units:
        return int(float(t[:-1]) * units[t[-1]])
    return int(t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m egolearn.data.packed",
        description="Build, verify, or describe a packed frame store.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="pack frame directories into a store")
    p.add_argument("frame_root", help="root holding <session>/<frame files> directories")
    p.add_argument("store", help="output store directory")
    p.add_argument("--include", default=None, help="fnmatch pattern on session names (e.g. 'S_*')")
    p.add_argument(
        "--sessions", nargs="*", default=None, help="explicit session list (wins over --include)"
    )
    p.add_argument("--naming", choices=["timestamp", "frame_index"], default="timestamp")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--shard-bytes", default="256M", help="target shard size, e.g. 256M")
    p.add_argument(
        "--workers", type=int, default=0, help="sessions packed in parallel (0 = in-process)"
    )
    p.add_argument(
        "--force", action="store_true", help="replace an existing store at the output path"
    )
    v = sub.add_parser("verify", help="compare a store with its source frame directories")
    v.add_argument("store")
    v.add_argument("frame_root")
    v.add_argument("--sample", type=int, default=None, help="compare only this many random frames")
    i = sub.add_parser("info", help="print a store's manifest summary")
    i.add_argument("store")
    args = ap.parse_args(argv)

    if args.cmd == "pack":
        pack_frame_dirs(
            args.frame_root,
            args.store,
            include=args.include,
            sessions=args.sessions,
            naming=args.naming,
            fps=args.fps,
            shard_bytes=_parse_bytes(args.shard_bytes),
            workers=args.workers,
            force=args.force,
            log=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    elif args.cmd == "verify":
        n = verify_store(args.store, args.frame_root, sample=args.sample)
        print(f"ok: {n} frames byte-identical to {args.frame_root}")
    else:
        store = PackedFrameStore(args.store)
        m = store.manifest
        print(
            f"{args.store}: {m['frame_count']} frames, {len(m['sessions'])} sessions, "
            f"{len(m['shards'])} shards, {m['total_bytes'] / 1e9:.2f} GB, "
            f"fps={m['fps']} naming={m['naming']} created={m['created']}"
        )
    return 0
