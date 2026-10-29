"""Throughput knobs must preserve MAE pixels, seeds, ordering, and gradients."""

import random
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.core.refs import ItemRef
from egolearn.data import loading
from egolearn.data.sources.saycam import SaycamStream
from egolearn.data.views import ViewBank
from egolearn.methods.mae import MAE, MAEDecoder, MAEEncoder
from egolearn.methods.mae.views import mae
from egolearn.model.model import ComposedModel


def test_uint8_pipeline_preserves_normalized_pixels_and_loss():
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (43, 67, 3), dtype=np.uint8))
    torch.manual_seed(13)
    normalized = mae(size=32)(image)
    torch.manual_seed(13)
    raw = mae(size=32, uint8=True)(image)
    assert raw.dtype == torch.uint8
    model = ComposedModel(
        encoders={"vision": MAEEncoder(image_size=32, patch_size=8, dim=32, depth=1, heads=4)},
        predictors={
            "reconstruction": {
                "inputs": ["vision"],
                "module": MAEDecoder(
                    image_size=32, patch_size=8, encoder_dim=32, dim=32, depth=1, heads=4
                ),
            }
        },
    )
    loss = MAE("frames", "vision", "reconstruction")
    values = []
    gradients = []
    for pixels in (normalized, raw):
        model.zero_grad(set_to_none=True)
        value = loss.compute(
            {"frames": {K.SLOT_VIEWS: pixels[None, None]}}, model, Context(device="cpu")
        )["mse"]
        value.backward()
        values.append(value.detach())
        gradients.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_packed_workers_preserve_order_and_seeded_views(tmp_path):
    root = tmp_path / "S_20200101"
    root.mkdir()
    for i in range(5):
        pixels = np.random.default_rng(i).integers(0, 256, (43, 67, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"frame_{i}.jpg")
    stream = SaycamStream(str(tmp_path))
    refs = [ItemRef(root.name, "video", float(i), float(i), (root.name, i)) for i in range(5)]
    items = [("frame", ref, 0, ["v"], i + 10) for i, ref in enumerate(refs)]
    bank = ViewBank({"v": {"name": "mae", "size": 32, "uint8": True}})
    serial = loading.Loader(stream, bank)
    expected = serial.tensor_batch(items)
    packed = loading.Loader(
        stream, bank, workers=2, packed=True, cache_frames=3, imports=["egolearn.methods.mae"]
    )
    try:
        for _ in range(2):
            torch.testing.assert_close(packed.tensor_batch(items), expected, rtol=0, atol=0)
    finally:
        packed.close()


def test_decoded_cache_is_bounded_and_does_not_cache_augmentations(tmp_path, monkeypatch):
    class Source(SaycamStream):
        calls = 0

        def open_frame(self, ref, index):
            self.calls += 1
            return Image.new("RGB", (4, 4), (10, 20, 30))

    class MutatingView:
        def apply(self, image, names):
            original = image.getpixel((0, 0))
            image.putpixel((0, 0), (255, 255, 255))
            return torch.tensor(original)

    source = Source(str(tmp_path))
    monkeypatch.setattr(
        loading,
        "_WORKER",
        {
            "stream": source,
            "views": MutatingView(),
            "cache": OrderedDict(),
            "cache_frames": 1,
        },
    )
    refs = [ItemRef("s", "video", i, i, (i,)) for i in range(2)]
    rng_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            for i in (0, 0, 1, 0):
                result = loading._pool_task(("frame", refs[i], 0, ["v"], 10))
                assert result.tolist() == [10, 20, 30]
    finally:
        random.setstate(rng_state)
    assert source.calls == 3
    assert len(loading._WORKER["cache"]) == 1


def test_requested_pinning_falls_back_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = loading.Loader(SaycamStream(str(tmp_path)), ViewBank({}), pin_memory=True)
    assert not loader.pin_memory
