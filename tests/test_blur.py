"""Numerical and RNG contracts of the Gaussian blur implementations and of the
worker-side flush-denormal mode."""

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from egolearn.core.refs import ItemRef
from egolearn.data.blur import SeparableGaussianBlur, SkipGaussianBlur, gaussian_blur
from egolearn.data.loading import Loader
from egolearn.data.sources.toy import ToyStream
from egolearn.data.views import ViewBank, simclr_strong
from egolearn.methods.babycl.data import babycl_ms_aug


def test_default_implementation_is_separable():
    assert isinstance(gaussian_blur(23), SeparableGaussianBlur)
    assert isinstance(gaussian_blur(23, "torchvision"), T.GaussianBlur)
    assert isinstance(gaussian_blur(23, "none"), SkipGaussianBlur)
    with pytest.raises(ValueError, match="blur_impl"):
        gaussian_blur(23, "fast")


@pytest.mark.parametrize("mode", ["RGB", "L"])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 1.0, 2.0])
@pytest.mark.parametrize("kernel", [3, 23, (23, 7)])
def test_separable_matches_reference_filter_and_rng(mode, sigma, kernel):
    shape = (32, 40, 3) if mode == "RGB" else (32, 40)
    pixels = np.random.default_rng(5).integers(0, 256, shape, dtype=np.uint8)
    image = Image.fromarray(pixels)
    torch.manual_seed(7)
    reference = np.asarray(T.GaussianBlur(kernel, (sigma, sigma))(image))
    reference_rng = torch.get_rng_state()
    torch.manual_seed(7)
    actual = np.asarray(SeparableGaussianBlur(kernel, (sigma, sigma))(image))
    assert torch.equal(torch.get_rng_state(), reference_rng)
    error = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    assert error.max() <= 1
    assert np.mean(error != 0) < 0.005
    assert np.array_equal(np.asarray(image), pixels)


def test_separable_preserves_constant_images_and_rejects_unsupported_input():
    transform = SeparableGaussianBlur(23)
    image = Image.new("RGB", (32, 32), (0, 127, 255))
    assert np.array_equal(np.asarray(transform(image)), np.asarray(image))
    with pytest.raises(TypeError, match="RGB or L"):
        transform(Image.new("F", (32, 32)))


@pytest.mark.parametrize("view_idx", [0, 1])
@pytest.mark.parametrize("builder", [simclr_strong, babycl_ms_aug])
@pytest.mark.parametrize("size", [64, 224])
def test_default_view_tracks_reference_within_one_level_and_same_draws(view_idx, builder, size):
    pixels = np.random.default_rng(19).integers(0, 256, (80, 96, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    reference = builder(size=size, view_idx=view_idx, blur_impl="torchvision")
    default = builder(size=size, view_idx=view_idx)
    for seed in range(8):
        torch.manual_seed(seed)
        expected = reference(image)
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        actual = default(image)
        assert torch.equal(state, torch.get_rng_state())
        torch.testing.assert_close(actual, expected, rtol=0, atol=1 / (255 * 0.224) + 1e-6)
    with pytest.raises(ValueError, match="blur_impl"):
        builder(blur_impl="unknown")


@pytest.mark.parametrize("view_idx", [0, 1])
@pytest.mark.parametrize("builder", [simclr_strong, babycl_ms_aug])
def test_no_blur_preserves_random_draws_and_skips_filter(view_idx, builder, monkeypatch):
    image = Image.fromarray(np.random.default_rng(23).integers(0, 256, (80, 96, 3), dtype=np.uint8))
    enabled = builder(size=64, view_idx=view_idx)
    disabled = builder(size=64, view_idx=view_idx, blur_impl="none")
    expected_states = []
    for seed in range(12):
        torch.manual_seed(seed)
        enabled(image)
        expected_states.append(torch.get_rng_state())

    def forbidden_filter(*args, **kwargs):
        raise AssertionError("blur filtering ran despite blur_impl=none")

    monkeypatch.setattr(T.functional, "gaussian_blur", forbidden_filter)
    monkeypatch.setattr(Image.Image, "filter", forbidden_filter)
    for seed, expected in enumerate(expected_states):
        torch.manual_seed(seed)
        actual = disabled(image)
        assert actual.shape == (3, 64, 64)
        assert torch.equal(torch.get_rng_state(), expected)


# ---------------------------------------------------------------------------
# flush-denormal: a throughput setting that must not touch a single pixel
# ---------------------------------------------------------------------------
def _adversarial_images():
    """Images whose blurred pixels are dominated by the kernel's tail weights:
    isolated bright pixels and lines on black, so most output pixels sum only
    subnormal contributions (the case where flushing could change a result)."""
    rng = np.random.default_rng(3)
    black = np.zeros((48, 56, 3), dtype=np.uint8)
    dot = black.copy()
    dot[20, 30] = 255
    line = black.copy()
    line[10, :] = (255, 128, 1)
    column = black.copy()
    column[:, 40] = 255
    sparse = black.copy()
    sparse[rng.integers(0, 48, 12), rng.integers(0, 56, 12)] = rng.integers(1, 256, (12, 3))
    noise = rng.integers(0, 256, (48, 56, 3), dtype=np.uint8)
    return [Image.fromarray(a) for a in (dot, line, column, sparse, noise)]


def _tail_weights_are_subnormal(kernel: int, sigma: float) -> bool:
    # torchvision's kernel: exp(-0.5 (x / sigma)^2) over the centred taps
    x = torch.linspace(-(kernel - 1) / 2, (kernel - 1) / 2, kernel)
    pdf = torch.exp(-0.5 * (x / sigma) ** 2)
    tiny = torch.finfo(torch.float32).tiny
    return bool(((pdf > 0) & (pdf < tiny)).any())


@pytest.mark.parametrize("impl", ["separable", "torchvision"])
def test_flush_denormal_leaves_blurred_pixels_identical(impl):
    if not torch.set_flush_denormal(False):
        pytest.skip("this CPU has no flush-denormal mode")
    kernel = 23
    sigmas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0]
    assert any(_tail_weights_are_subnormal(kernel, s) for s in sigmas)
    images = _adversarial_images()
    try:
        for sigma in sigmas:
            blur = type(gaussian_blur(kernel, impl))(kernel, sigma=(sigma, sigma))
            torch.set_flush_denormal(False)
            kept = [np.asarray(blur(im)) for im in images]
            torch.set_flush_denormal(True)
            flushed = [np.asarray(blur(im)) for im in images]
            for a, b in zip(kept, flushed, strict=True):
                assert np.array_equal(a, b), (impl, sigma)
    finally:
        torch.set_flush_denormal(False)


@pytest.mark.parametrize(
    ("view_name", "blur_impl"),
    [("simclr_strong", "separable"), ("babycl_ms_aug", "separable"), ("simclr_strong", "none")],
)
def test_views_match_across_worker_counts(view_name, blur_impl):
    # workers (which flush denormals) must load the same tensors as in-process
    stream = ToyStream(n_sessions=1, img_size=80, noise=80)
    views = ViewBank(
        {
            f"v{i}": {"name": view_name, "size": 64, "view_idx": i, "blur_impl": blur_impl}
            for i in range(2)
        }
    )
    items = [
        ("frame", ItemRef("toy_00", "video", i, i, ("toy_00", i)), 0, ["v0", "v1"], 123 + i)
        for i in range(8)
    ]
    expected = Loader(stream, views).views_batch(items)
    pooled = Loader(stream, views, workers=2, imports=["egolearn.methods.babycl"])
    assert pooled.flush_denormal
    try:
        actual = pooled.views_batch(items)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    finally:
        pooled.close()
