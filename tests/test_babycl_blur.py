"""Pixel and RNG compatibility of BabyCL's Pillow radius augmentation."""

import random

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from egolearn.methods.babycl.data import PILRadiusBlur


@pytest.fixture(autouse=True)
def restore_python_rng():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.mark.parametrize("mode", ["RGB", "L"])
@pytest.mark.parametrize("sigma", [None, (0.3, 1.25), (0.75, 0.75), (0.0, 0.0)])
def test_pil_radius_blur_matches_seeded_pillow_pixels_and_rng(mode, sigma):
    shape = (21, 29, 3) if mode == "RGB" else (21, 29)
    pixels = np.random.default_rng(5).integers(0, 256, shape, dtype=np.uint8)
    image = Image.fromarray(pixels)
    blur = PILRadiusBlur() if sigma is None else PILRadiusBlur(sigma=sigma)
    bounds = (0.1, 2.0) if sigma is None else sigma
    torch_state = torch.get_rng_state()

    for seed in (0, 17, 1337):
        expected_rng = random.Random(seed)
        random.seed(seed)
        for _ in range(3):
            radius = expected_rng.uniform(*bounds)
            expected = image.filter(ImageFilter.GaussianBlur(radius=radius))
            actual = blur(image)
            assert actual.mode == image.mode
            assert actual.size == image.size
            assert actual.tobytes() == expected.tobytes()
            assert random.getstate() == expected_rng.getstate()

    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.asarray(image), pixels)
