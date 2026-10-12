"""Packed frame store: byte-identical frames, refs interchangeable with the
frame-dir adapter, one-read clips, cheap pickling, and the build/verify CLI."""

import json
import os
import pickle
import random

import numpy as np
import pytest
import torch
from PIL import Image

from egolearn.config.instantiate import instantiate
from egolearn.core.refs import ItemRef
from egolearn.data.loading import Loader
from egolearn.data.packed import PackedFrameStore, main, pack_frame_dirs, verify_store
from egolearn.data.sources.packed import PackedFrameStream
from egolearn.data.sources.saycam import SaycamStream
from egolearn.data.views import ViewBank

S1, S2, S3 = "S_20130101_0001_01", "S_20130102_0002_01", "S_20130103_0003_01"
N1, N2, N3 = 37, 5, 64


def _write_session(root, sid, n, rng, t0=0.0, fps=5.0, naming="timestamp"):
    d = root / sid
    d.mkdir(parents=True)
    for i in range(n):
        img = Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8))
        fn = f"{sid}_{t0 + i / fps:.2f}.jpg" if naming == "timestamp" else f"frame_{i + 1:08d}.jpg"
        img.save(d / fn, quality=int(rng.integers(40, 95)))  # varied byte sizes


def _make_frame_root(base):
    rng = np.random.default_rng(0)
    root = base / "frames"
    _write_session(root, S1, N1, rng)
    _write_session(root, S2, N2, rng, t0=3.0)
    _write_session(root, S3, N3, rng)
    _write_session(root, "A_20130104_0004_01", 9, rng)
    (root / "S_20130105_empty").mkdir()  # no frames: both adapters skip it
    (root / "notes.txt").write_text("not a session")
    transcript = base / "expand.json"
    transcript.write_text(
        json.dumps(
            [
                {
                    "video_filename": f"{S2}.mp4",
                    "utterance": "look",
                    "utterance_num": 1,
                    "timestamps": [3.2, 3.6],
                },
                {
                    "video_filename": f"{S3}.mp4",
                    "utterance": "a ball",
                    "utterance_num": 2,
                    "timestamps": [1.0, 2.0, 2.4],
                    "clip_importance": 0.5,
                },
            ]
        )
    )
    return root, transcript


@pytest.fixture
def frame_root(tmp_path):
    return _make_frame_root(tmp_path)


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    base = tmp_path_factory.mktemp("packed")
    root, transcript = _make_frame_root(base)
    out = base / "store"
    # tiny shards: every session spans several files and clips cross shard edges
    manifest = pack_frame_dirs(root, out, include="S_*", shard_bytes=4096, workers=2)
    return root, transcript, out, manifest


def _pixels(stream, ref, i):
    return np.asarray(stream.open_frame(ref, i))


def test_store_is_byte_identical_and_refs_interchangeable(packed):
    root, transcript, out, manifest = packed
    assert [s["id"] for s in manifest["sessions"]] == [S1, S2, S3]
    assert len(manifest["shards"]) > 3
    store = PackedFrameStore(out)
    assert len(store) == N1 + N2 + N3
    src = SaycamStream(str(root), transcript_json=str(transcript), include="S_*")
    dst = PackedFrameStream(str(out), transcript_json=str(transcript))
    assert dst.session_ids() == [s.sid for s in src.sessions()] == [S1, S2, S3]
    assert dst.total_observations() == src.total_observations() == N1 + N2 + N3 + 2
    for a, b in zip(src.sessions(), dst.sessions(), strict=True):
        assert a.sid == b.sid and list(a.channels) == list(b.channels)
        va, vb = a.channels["video"], b.channels["video"]
        assert va.times == vb.times
        files = src.frame_files(a.sid)
        for i in range(len(va)):
            assert va.ref_at(i) == vb.ref_at(i)
            # a ref minted by either adapter reads the same pixels from the other
            assert np.array_equal(_pixels(dst, va.ref_at(i), 0), _pixels(src, vb.ref_at(i), 0))
            assert store.read(store.span(a.sid).first + i) == (root / a.sid / files[i]).read_bytes()
        if "transcript" in a.channels:
            obs_a = [(o.t, o.ref, o.payload) for o in a.channels["transcript"]]
            obs_b = [(o.t, o.ref, o.payload) for o in b.channels["transcript"]]
            assert obs_a == obs_b
    dst.close()
    store.close()


def test_windows_and_clips_read_the_same_frames(packed):
    root, _, out, _ = packed
    src = SaycamStream(str(root), include="S_*")
    dst = PackedFrameStream(str(out))
    ref = ItemRef(S3, "video", 2.0, 11.8, (S3, 10, 60))
    picks = [0, 3, 3, 19, 49, 48, 0]  # repeats, gaps and out of order, across shard edges
    got = dst.open_frames(ref, picks)
    want = src.open_frames(ref, picks)
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(IndexError):
        dst.open_frames(ref, [50])
    keys = [dst.store.span(S3).first + 10 + i for i in picks]
    assert dst.store.read_many(keys, merge_gap=0) == dst.store.read_many(keys)
    assert dst.store.read_many([]) == []

    views = ViewBank(
        {
            "ce": {"name": "clip_center_crop", "size": 16},
            "cs": {"name": "clip_simclr", "size": 16, "view_idx": 0},
        }
    )
    ta = Loader(src, views).clip_views(ref, ["ce", "cs"], 8, 3, random.Random(0), seed=5)
    tb = Loader(dst, views).clip_views(ref, ["ce", "cs"], 8, 3, random.Random(0), seed=5)
    assert ta.shape == (2, 8, 3, 16, 16) and torch.equal(ta, tb)
    short = ItemRef(S2, "video", 3.0, 3.4, (S2, 0, 3))
    ta = Loader(src, views).clip_views(short, ["ce"], 6, 1, random.Random(1), seed=2)
    tb = Loader(dst, views).clip_views(short, ["ce"], 6, 1, random.Random(1), seed=2)
    assert torch.equal(ta, tb) and torch.equal(tb[0, 2], tb[0, 5])  # clamped repeats
    dst.close()


def test_stream_pickles_small_and_workers_match_in_process(packed):
    _, transcript, out, _ = packed
    dst = PackedFrameStream(str(out), transcript_json=str(transcript))
    blob = pickle.dumps(dst)
    assert len(blob) < 4096  # the store travels as its path; no listing cache
    again = pickle.loads(blob)
    ref = ItemRef(S1, "video", 1.0, 1.0, (S1, 5))
    assert np.array_equal(np.asarray(again.open_frame(ref, 0)), np.asarray(dst.open_frame(ref, 0)))

    views = ViewBank(
        {
            "v": {"name": "simclr_strong", "size": 16, "view_idx": 0},
            "c": {"name": "clip_simclr", "size": 16, "view_idx": 1},
        }
    )
    items: list[tuple] = [
        ("frame", ItemRef(S3, "video", i / 5, i / 5, (S3, i)), 0, ["v"], 100 + i)
        for i in range(0, N3, 9)
    ]
    items.append(("clip", ItemRef(S1, "video", 0.0, 7.2, (S1, 0, N1)), 30, ["c"], 4, 2, 7))
    expected = Loader(dst, views).views_batch(items)
    pooled = Loader(dst, views, workers=2)
    try:
        actual = pooled.views_batch(items)
    finally:
        pooled.close()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    dst.close()


def test_session_selection_and_registry(packed):
    _, _, out, _ = packed
    assert PackedFrameStream(str(out), sessions=[S3, S1]).session_ids() == [S3, S1]
    assert PackedFrameStream(str(out), include="S_2013010[12]*").session_ids() == [S1, S2]
    with pytest.raises(KeyError, match="not in the packed store"):
        PackedFrameStream(str(out), sessions=["S_nope"]).session_ids()
    stream = instantiate(
        {"_target_": "stream:packed_frames", "root_dir": str(out), "include": "S_*"}
    )
    assert isinstance(stream, PackedFrameStream)


def test_build_cli_force_verify_and_corruption(frame_root, tmp_path, capsys):
    root, _ = frame_root
    out = tmp_path / "store2"
    argv = [
        "pack",
        str(root),
        str(out),
        "--include",
        "S_*",
        "--shard-bytes",
        "8K",
        "--workers",
        "0",
    ]
    assert main(argv) == 0
    assert len(PackedFrameStore(out)) == N1 + N2 + N3
    with pytest.raises(FileExistsError):
        pack_frame_dirs(root, out, include="S_*")
    assert main(["pack", str(root), str(out), "--sessions", S2, S1, "--force"]) == 0
    store = PackedFrameStore(out)
    assert [s.id for s in store.sessions()] == [S2, S1]
    leftover = sorted(p.name for p in (out / "shards").iterdir())
    assert leftover and all(name.startswith((S1, S2)) for name in leftover)  # old shards gone
    assert verify_store(out, root) == N1 + N2
    assert verify_store(out, root, sample=10) == 10
    assert main(["info", str(out)]) == 0
    assert main(["verify", str(out), str(root), "--sample", "5"]) == 0
    printed = capsys.readouterr().out
    assert f"{N1 + N2} frames" in printed and "ok: 5 frames" in printed

    shard = out / store.manifest["shards"][0]["path"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="differs"):
        verify_store(out, root)
    shard.write_bytes(bytes(data[:-10]))  # truncated shard: a short read must not pass silently
    rows = np.asarray(store.index)
    last = int(np.argmax(np.where(rows[:, 0] == 0, rows[:, 1], -1)))
    with pytest.raises(OSError, match="short read"):
        store.read(last)
    store.close()
    with pytest.raises(FileNotFoundError, match="not a packed frame store"):
        _ = PackedFrameStore(tmp_path / "nowhere").manifest
    with pytest.raises(ValueError, match="no frame directories"):
        pack_frame_dirs(root, tmp_path / "store3", include="Z_*")


def test_frame_index_naming(tmp_path):
    rng = np.random.default_rng(1)
    root = tmp_path / "bv"
    _write_session(root, "bv_001", 7, rng, naming="frame_index")
    out = tmp_path / "bv_store"
    pack_frame_dirs(root, out, naming="frame_index", fps=2.0)
    stream = PackedFrameStream(str(out))
    (session,) = list(stream.sessions())
    assert session.channels["video"].times == [i / 2.0 for i in range(7)]
    assert stream.fps == 2.0 and stream.naming == "frame_index"
    assert verify_store(out, root) == 7


def test_frame_names_serve_storyboard_validation_and_empty_sessions(frame_root, tmp_path):
    from egolearn.methods.babycl.offline import build_offline_index

    root, transcript = frame_root
    out = tmp_path / "store4"
    selected = [S3, "S_20130105_empty", S2]  # an explicit list with an empty directory
    pack_frame_dirs(root, out, sessions=selected)
    store = PackedFrameStore(out)
    assert [s.id for s in store.sessions()] == [S3, S2]
    assert store.empty_sessions() == ["S_20130105_empty"]
    src = SaycamStream(str(root), transcript_json=str(transcript), sessions=selected)
    dst = PackedFrameStream(str(out), transcript_json=str(transcript), sessions=selected)
    assert dst.session_ids() == [s.sid for s in src.sessions()] == [S3, S2]
    assert dst.frame_files(S3) == src.frame_files(S3) == store.session_names(S3)

    storyboard = tmp_path / "storyboard.json"
    videos = [
        {"video_dir": sid, "frames": src.frame_files(sid), "segments": [{"start": 0, "end": n}]}
        for sid, n in ((S3, N3), (S2, N2))
    ]
    storyboard.write_text(json.dumps({"fps": 5.0, "root_dir": "unused", "videos": videos}))
    a = build_offline_index(src, str(storyboard))
    b = build_offline_index(dst, str(storyboard))
    assert (a.num_frames, a.segments, a.pairs) == (b.num_frames, b.segments, b.pairs)
    assert b.num_frames == N3 + N2 and len(b.segments) == 2 and b.pairs
    videos[0]["frames"] = list(reversed(videos[0]["frames"]))
    storyboard.write_text(json.dumps({"fps": 5.0, "root_dir": "unused", "videos": videos}))
    with pytest.raises(ValueError, match="frame order differs"):
        build_offline_index(dst, str(storyboard))


def test_stale_refs_raise_on_both_adapters(packed):
    root, _, out, _ = packed
    src = SaycamStream(str(root), include="S_*")
    dst = PackedFrameStream(str(out))
    for ref, index in (
        (ItemRef(S1, "video", 0.0, 0.0, (S1, N1)), 0),  # point ref past the session's end
        (ItemRef(S1, "video", 0.0, 1.0, (S1, N1 - 2, N1 + 5)), 3),  # interval ref overrunning it
    ):
        with pytest.raises(IndexError):
            src.open_frame(ref, index)
        with pytest.raises(IndexError):
            dst.open_frame(ref, index)
        with pytest.raises(IndexError):
            dst.open_frames(ref, [index])


def test_interrupted_build_is_refused_without_force_and_cleaned_with_it(frame_root, tmp_path):
    root, _ = frame_root
    out = tmp_path / "store5"
    (out / "shards").mkdir(parents=True)
    stray = out / "shards" / "S_old.000.bin"
    stray.write_bytes(b"leftover from a build that never wrote its manifest")
    with pytest.raises(FileExistsError, match="interrupted build"):
        pack_frame_dirs(root, out, include="S_*")
    pack_frame_dirs(root, out, include="S_*", force=True)
    assert not stray.exists()
    assert verify_store(out, root) == N1 + N2 + N3


def test_shard_descriptors_are_capped_and_released(packed):
    import gc

    _, _, out, _ = packed
    fd_dir = "/proc/self/fd"
    count = (lambda: len(os.listdir(fd_dir))) if os.path.isdir(fd_dir) else (lambda: 0)
    gc.collect()
    base = count()
    store = PackedFrameStore(out, max_open_shards=1)  # LRU of one: every shard change reopens
    keys = list(range(0, N1 + N2 + N3, 7))
    assert store.read_many(keys) == [store.read(k) for k in keys]
    assert len(store._fds) == 1
    assert PackedFrameStore(out).max_open_shards >= 64
    if os.path.isdir(fd_dir):
        assert count() > base  # a shard fd plus the memory-mapped index
    del store
    gc.collect()
    assert count() == base  # the finalizer released them without an explicit close()
