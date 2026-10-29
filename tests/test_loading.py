"""Throughput options must preserve pixels, randomness and item order."""

import os
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from egolearn.core.refs import ItemRef
from egolearn.data.loading import Loader, _decode_clip, _open_frame
from egolearn.data.sources.base import Stream
from egolearn.data.views import ViewBank


class CountingStream(Stream):
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)

    def num_frames(self, ref):
        return 4

    def open_frame(self, ref, index):
        with (self.log_dir / f"{os.getpid()}.txt").open("a") as f:
            f.write(f"{ref.key}:{index}\n")
        pixels = np.arange(36 * 40 * 3, dtype=np.uint8).reshape(36, 40, 3)
        return Image.fromarray(np.roll(pixels, index + ref.locator[0], axis=0))


class DenormalCheckingStream(CountingStream):
    def __init__(self, log_dir, flush_denormal):
        super().__init__(log_dir)
        self.flush_denormal = flush_denormal

    def open_frame(self, ref, index):
        assert (torch.tensor(1e-40).item() == 0) == self.flush_denormal
        return super().open_frame(ref, index)


def items(start=0, seed=11):
    return [
        ("clip", ItemRef("session", "video", 0, 1, (i,)), 0, ["clip"], 4, 1, seed + i)
        for i in range(start, start + 4)
    ]


def bank():
    return ViewBank(
        {
            "clip": {"name": "clip_rrc", "size": 32, "scale_min": 0.4},
            "frame": {"name": "center_crop", "size": 24},
        }
    )


def count_decodes(path):
    return sum(len(p.read_text().splitlines()) for p in path.glob("*.txt"))


def test_decode_cache_is_bounded_and_keeps_pristine_pixels(tmp_path):
    stream = CountingStream(tmp_path)
    cache: OrderedDict = OrderedDict()
    ref = items()[0][1]
    expected = _decode_clip(stream, ref, 0, 6, 1)
    actual = _decode_clip(stream, ref, 0, 6, 1, cache, 4)
    assert torch.equal(actual, expected)
    assert count_decodes(tmp_path) == 10  # six uncached + four distinct frames
    actual.zero_()
    assert torch.equal(_decode_clip(stream, ref, 0, 6, 1, cache, 4), expected)
    assert count_decodes(tmp_path) == 10
    _open_frame(stream, replace(ref, locator=(5,)), 0, cache, 4)
    assert len(cache) == 4
    _open_frame(stream, ref, 0, cache, 4)
    assert count_decodes(tmp_path) == 12  # oldest frame was evicted


@pytest.mark.parametrize(
    "packed,affinity", [(False, False), (True, False), (False, True), (True, True)]
)
def test_parallel_loading_matches_serial_and_reuses_live_clips(tmp_path, packed, affinity):
    serial_dir, worker_dir = tmp_path / "serial", tmp_path / "worker"
    serial_dir.mkdir()
    worker_dir.mkdir()
    serial = Loader(CountingStream(serial_dir), bank())
    pooled = Loader(
        CountingStream(worker_dir),
        bank(),
        workers=2,
        packed=packed,
        affinity=affinity,
        cache_frames=16,
    )
    try:
        for n, work in enumerate((items(), items(seed=101)[::-1], items(start=2, seed=201))):
            assert torch.equal(pooled.tensor_batch(work), serial.tensor_batch(work))
            if affinity:
                assert count_decodes(worker_dir) == (16 if n < 2 else 24)
                assert len(pooled._owners) == 4
        # Mixed frame/clip slots cannot be stacked inside a worker.
        mixed = [items()[0], ("frame", items()[1][1], 1, ["frame"], 14)]
        for actual, expected in zip(
            pooled.views_batch(mixed), serial.views_batch(mixed), strict=True
        ):
            assert torch.equal(actual, expected)
        frames = [("frame", work[1], 0, ["frame"], 21 + i) for i, work in enumerate(items())]
        assert torch.equal(pooled.tensor_batch(frames), serial.tensor_batch(frames))
        # Cold caches and ownership after resume must produce the same next batch.
        pooled.close()
        assert torch.equal(
            pooled.tensor_batch(items(seed=401)), serial.tensor_batch(items(seed=401))
        )
        assert serial.frame_requests == pooled.frame_requests == 73
    finally:
        pooled.close()
        serial.close()


@pytest.mark.parametrize("affinity", [False, True])
def test_cached_worker_pools_honor_denormal_setting(tmp_path, affinity):
    expected = Loader(CountingStream(tmp_path), bank()).tensor_batch(items())
    for flush_denormal in (False, True):
        loader = Loader(
            DenormalCheckingStream(tmp_path, flush_denormal),
            bank(),
            1,
            None,
            flush_denormal,  # Preserve the positional argument already available on main.
            packed=True,
            affinity=affinity,
            cache_frames=4,
        )
        try:
            assert torch.equal(loader.tensor_batch(items()), expected)
        finally:
            loader.close()


def test_frame_view_mutation_cannot_poison_worker_cache(tmp_path, monkeypatch):
    from egolearn.data import loading

    monkeypatch.setitem(loading._WORKER, "stream", CountingStream(tmp_path))
    monkeypatch.setitem(loading._WORKER, "cache", OrderedDict())
    monkeypatch.setitem(loading._WORKER, "cache_frames", 4)

    class MutatingViews:
        def apply(self, image, names):
            result = torch.tensor(np.array(image))
            image.paste((0, 0, 0), (0, 0, image.width, image.height))
            return result

    monkeypatch.setitem(loading._WORKER, "views", MutatingViews())
    work = ("frame", items()[0][1], 0, ["mutating"], 17)
    assert torch.equal(loading._pool_task(work), loading._pool_task(work))
    assert count_decodes(tmp_path) == 1


@pytest.mark.parametrize("options", [{"cache_frames": -1}, {"cache_frames": 4}, {"affinity": True}])
def test_invalid_loading_options_fail(tmp_path, options):
    with pytest.raises(ValueError):
        Loader(CountingStream(tmp_path), bank(), **options)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_clip_conversion_exactly_matches_original_to_tensor_stack(tmp_path, dtype, mode):
    class Source(CountingStream):
        def open_frame(self, ref, index):
            return super().open_frame(ref, index).convert(mode)

    stream, ref = Source(tmp_path), items()[0][1]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        expected = torch.stack([TF.to_tensor(stream.open_frame(ref, i)) for i in (1, 3, 3, 3)])
        actual = _decode_clip(stream, ref, 1, 4, 2)
        assert actual.is_contiguous()
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.gpu
@pytest.mark.parametrize("affinity", [False, True])
def test_packed_batch_is_pinned_and_matches_serial(tmp_path, affinity):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA pinned allocation")
    stream = CountingStream(tmp_path)
    serial = Loader(stream, bank())
    pooled = Loader(
        stream, bank(), workers=2, packed=True, pin_memory=True, affinity=affinity, cache_frames=16
    )
    try:
        actual = pooled.tensor_batch(items())
        assert actual.is_pinned()
        expected = serial.tensor_batch(items())
        assert torch.equal(actual, expected)
        assert torch.equal(actual.to("cuda", non_blocking=True).cpu(), expected)
    finally:
        pooled.close()


def test_affinity_balances_new_clips_and_prunes_old_owners(tmp_path):
    loader = Loader(CountingStream(tmp_path), bank(), workers=2, affinity=True)
    assert loader._affinity_groups(items()) == [[0, 2], [1, 3]]
    original = dict(loader._owners)
    assert loader._affinity_groups(items()[::-1]) == [[1, 3], [0, 2]]
    loader._affinity_groups(items(start=2))
    assert len(loader._owners) == 4
    for key in original.keys() & loader._owners.keys():
        assert loader._owners[key] == original[key]
    assert sorted(loader._owners.values()) == [0, 0, 1, 1]
    loader.close()
    assert not loader._owners
