"""Standard MAE behavior and pipeline integration, using synthetic images only."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.evaluation import load_model
from egolearn.methods.mae import MAE, MAEDecoder, MAEEncoder, mae
from egolearn.methods.mae.loss import patchify, random_mask, reconstruction_loss
from egolearn.model.build import build_model
from egolearn.model.losses.base import own_forward_slots
from egolearn.training.solver import ProgressCosine, build_optimizer
from egolearn.utils.snapshot import load_snapshot, save_snapshot


@pytest.fixture
def config(tmp_path):
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="toy_mae")
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    plain["run_dir"] = str(tmp_path)
    plain["hooks"] = {}
    return plain


def test_patchify_preserves_spatial_and_rgb_order():
    images = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
    patches = patchify(images, 2)
    expected = torch.stack(
        [
            images[:, :, row : row + 2, col : col + 2].permute(0, 2, 3, 1).reshape(2, -1)
            for row in (0, 2)
            for col in (0, 2, 4)
        ],
        dim=1,
    )
    torch.testing.assert_close(patches, expected)
    with pytest.raises(ValueError, match="multiples"):
        patchify(images, 3)


def test_masks_are_per_image_seeded_and_have_exact_counts():
    visible, hidden = random_mask(8, 16, 0.75, generator=torch.Generator().manual_seed(7))
    torch.rand(100)  # unrelated global randomness must not change the mask
    again = random_mask(8, 16, 0.75, generator=torch.Generator().manual_seed(7))
    assert torch.equal(visible, again[0]) and torch.equal(hidden, again[1])
    assert visible.shape == (8, 4)
    assert hidden.sum(1).tolist() == [12] * 8
    assert not hidden.gather(1, visible).any()
    assert all(len(set(row.tolist())) == 4 for row in visible)
    assert torch.unique(hidden, dim=0).size(0) > 1


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1, float("nan")])
def test_invalid_mask_ratios_fail_early(ratio):
    with pytest.raises(ValueError, match="mask_ratio"):
        MAE("frames", "vision", "reconstruction", mask_ratio=ratio)
    with pytest.raises(ValueError, match="mask_ratio"):
        random_mask(2, 16, ratio, generator=torch.Generator())


def test_mask_requires_visible_and_hidden_patches():
    with pytest.raises(ValueError, match="at least one"):
        random_mask(2, 1, 0.75, generator=torch.Generator())
    with pytest.raises(ValueError, match="at least one"):
        random_mask(2, 4, 0.99, generator=torch.Generator())


def test_encoder_cannot_read_hidden_pixels():
    encoder = MAEEncoder(image_size=16, patch_size=4, dim=32, depth=2, heads=4)
    images = torch.randn(2, 3, 16, 16)
    visible = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    changed = images.clone()
    changed[:, :, 4:, :] += 100  # only hidden rows of patches
    sequence_lengths = []
    hook = encoder.blocks.register_forward_pre_hook(
        lambda module, args: sequence_lengths.append(args[0].size(1))
    )
    first = encoder(images, visible=visible)
    second = encoder(changed, visible=visible)
    hook.remove()
    torch.testing.assert_close(first.tokens, second.tokens, rtol=0, atol=0)
    assert sequence_lengths == [5, 5]  # four visible patches plus CLS
    assert first.spatial is None
    full = encoder(images)
    assert full.tokens is not None and full.spatial is not None
    assert full.pooled.shape == (2, 32)
    assert full.tokens.shape == (2, 16, 32)
    assert full.spatial.shape == (2, 32, 4, 4)
    torch.testing.assert_close(full.pooled, full.aux["cls_token"][:, 0])
    encoder.pooling = "mean"
    torch.testing.assert_close(encoder(images).pooled, full.tokens.mean(dim=1))
    assert not encoder.pos_embed.requires_grad


def test_decoder_restores_positions_independent_of_visible_order():
    encoder = MAEEncoder(image_size=16, patch_size=4, dim=32, depth=1, heads=4)
    decoder = MAEDecoder(image_size=16, patch_size=4, encoder_dim=32, dim=16, depth=1, heads=4)
    images = torch.randn(2, 3, 16, 16)
    indices = torch.tensor([[7, 0, 10, 2], [13, 6, 1, 8]])
    output = encoder(images, visible=indices)
    assert output.tokens is not None
    expected = decoder(output.tokens, cls_token=output.aux["cls_token"], visible=indices)
    # Simultaneously permuting tokens and their locations must preserve reconstruction.
    perm = torch.tensor([2, 0, 3, 1])
    actual = decoder(
        output.tokens[:, perm], cls_token=output.aux["cls_token"], visible=indices[:, perm]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 16, 48)


@pytest.mark.parametrize("normalize", [False, True])
def test_reconstruction_loss_uses_only_masked_patches(normalize):
    # Every patch contains RGB values 1, 2, 3. For normalized targets the
    # sample variance convention gives mean square (N-1)/N for N=12 pixels.
    images = torch.tensor([1.0, 2.0, 3.0])[None, :, None, None].expand(1, 3, 2, 4)
    images = images.clone().requires_grad_()
    predictions = torch.zeros(1, 2, 12)
    predictions[:, 0] = 100
    predictions.requires_grad_()
    mask = torch.tensor([[False, True]])
    loss = reconstruction_loss(images, predictions, mask, patch_size=2, norm_pix_loss=normalize)
    expected = (11 / 12) / (1 + 1e-6 * 11 / 8) if normalize else 14 / 3
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert predictions.grad is not None
    assert predictions.grad[:, 0].count_nonzero() == 0
    assert predictions.grad[:, 1].count_nonzero() > 0
    assert images.grad is None  # targets must not receive gradients


def test_constant_targets_and_empty_masks():
    images = torch.ones(1, 3, 4, 4)
    predictions = torch.zeros(1, 4, 12)
    assert (
        reconstruction_loss(
            images, predictions, torch.ones(1, 4, dtype=torch.bool), patch_size=2
        ).item()
        == 0
    )
    with pytest.raises(ValueError, match="masked patch"):
        reconstruction_loss(images, predictions, torch.zeros(1, 4, dtype=torch.bool), patch_size=2)


@pytest.mark.parametrize(
    "kwargs", [{"image_size": 15}, {"patch_size": 0}, {"dim": 30}, {"heads": 5}, {"depth": 0}]
)
def test_invalid_model_geometry(kwargs):
    args: dict[str, Any] = dict(image_size=16, patch_size=4, dim=32, depth=1, heads=4)
    args.update(kwargs)
    with pytest.raises(ValueError):
        MAEEncoder(**args)


def test_loss_gradients_reach_both_transformers_and_skip_shared_forward(config):
    model = build_model(config["model"], imports=config["_imports_"])
    loss = MAE("frames", "vision", "reconstruction")
    batch = {"frames": {K.SLOT_VIEWS: torch.randn(2, 2, 3, 32, 32), K.SLOT_MODALITY: "image"}}
    ctx = Context(seed=10, device="cpu")
    model.precompute_feats(batch, ctx.device, skip=own_forward_slots([loss]))
    assert K.SLOT_FEATS not in batch["frames"]
    result = loss.compute(batch, model, ctx)["mse"]
    torch.rand(50)
    torch.testing.assert_close(result, loss.compute(batch, model, ctx)["mse"], rtol=0, atol=0)
    ctx.clock.step += 1
    assert result.item() != loss.compute(batch, model, ctx)["mse"].item()
    result.backward()
    parameters = dict(model.named_parameters())
    for name in (
        "encoders.vision.patch_embed.weight",
        "encoders.vision.blocks.0.self_attn.in_proj_weight",
        "encoders.vision.cls_token",
        "predictors.reconstruction.mask_token",
        "predictors.reconstruction.blocks.0.self_attn.in_proj_weight",
        "predictors.reconstruction.pixels.weight",
    ):
        parameter = parameters[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_mae_view_preserves_constant_color_up_to_normalization():
    view = mae(size=32)
    image = Image.new("RGB", (48, 64), (128, 64, 32))
    actual = view(image)
    expected = torch.tensor([128, 64, 32]) / 255 - torch.tensor([0.485, 0.456, 0.406])
    expected /= torch.tensor([0.229, 0.224, 0.225])
    assert actual.shape == (3, 32, 32)
    torch.testing.assert_close(actual, expected[:, None, None].expand_as(actual))


def test_streaming_resume_and_evaluation_reload_match_uninterrupted(config, tmp_path):
    config["driver"]["max_steps"] = 4
    reference = egolearn.build(config)
    reference.driver.fit()
    partial_cfg = deepcopy(config)
    partial_cfg["driver"]["max_steps"] = 2
    interrupted = egolearn.build(partial_cfg)
    interrupted.driver.fit()
    checkpoint = tmp_path / "mae.pth"
    save_snapshot(str(checkpoint), interrupted.driver)
    restored = egolearn.build(config)
    load_snapshot(str(checkpoint), restored.driver)
    restored.driver.fit()
    assert restored.ctx.clock.step == 4
    for name, want in reference.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], want, rtol=0, atol=0)
    # Standalone load uses saved construction metadata, including method imports.
    loaded = load_model(checkpoint, device="cpu", trusted_snapshot=True)
    images = torch.randn(2, 3, 32, 32)
    interrupted.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.encode("vision", images).pooled,
            interrupted.model.encode("vision", images).pooled,
            rtol=0,
            atol=0,
        )


def test_same_components_train_in_offline_driver(config):
    for key in ("memories", "admission", "routes", "recipe"):
        config.pop(key)
    config["driver"] = {
        "mode": "offline",
        "max_steps": 2,
        "batch": {"slot": "frames", "n": 4, "views": ["reconstruction"]},
    }
    pipe = egolearn.build(config)
    before = pipe.model.encoders["vision"].patch_embed.weight.detach().clone()
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2
    assert not torch.equal(before, pipe.model.encoders["vision"].patch_embed.weight)


@pytest.mark.parametrize(
    "name,dim,heads,lr", [("mae_small", 384, 6, 1.6e-4), ("mae_base", 768, 12, 8e-5)]
)
def test_project_config_composition(name, dim, heads, lr, tmp_path):
    directory = Path(__file__).resolve().parents[1] / "projects" / "streammae" / "configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        cfg = compose(config_name=name, overrides=[f"stream.root_dir={tmp_path}"])
    assert cfg.model.encoders.vision.dim == cfg.model.predictors.reconstruction.encoder_dim == dim
    assert cfg.model.encoders.vision.heads == heads
    assert cfg.solver.groups[0].lr == lr
    assert cfg.memories.frames.tiers.window.capacity == cfg.recipe.slots[0].n == 512
    assert cfg.driver.warmup.admitted == 511


def test_adam_betas_decay_exemptions_and_warmup(config):
    pipe = egolearn.build(config)
    assert all(group["betas"] == (0.9, 0.95) for group in pipe.driver.optimizer.param_groups)
    decay = {
        id(p): group["weight_decay"]
        for group in pipe.driver.optimizer.param_groups
        for p in group["params"]
    }
    for name, parameter in pipe.model.named_parameters():
        if "bias" in name or "norm" in name:
            assert decay[id(parameter)] == 0
    assert decay[id(pipe.model.encoders["vision"].patch_embed.weight)] == 0.05
    assert decay[id(pipe.model.encoders["vision"].cls_token)] == 0.05
    assert decay[id(pipe.model.predictors["reconstruction"].mask_token)] == 0.05
    ctx = Context(device="cpu", total_observations=100)
    schedule = ProgressCosine(warmup_frac=0.1, warmup_min_steps=0, floor=1.0, anchor=False)
    values = []
    for admitted in (0, 5, 10, 50, 100):
        ctx.clock.admitted = admitted
        values.append(schedule.multiplier(ctx))
    assert values == [0, 0.5, 1, 1, 1]


@pytest.mark.parametrize("opt", ["adam", "adamw"])
def test_adam_default_betas_are_unchanged(opt):
    optimizer = build_optimizer(torch.nn.Linear(2, 2), [{"match": ".*", "lr": 0.001}], opt=opt)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
