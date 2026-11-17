"""Numerical and batch-label contracts for supervised contrastive learning."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.model.losses.ntxent import _nt_xent
from egolearn.model.losses.supcon import SupervisedContrastive, _supcon
from egolearn.model.model import ComposedModel


def test_unique_global_labels_match_ntxent_values_and_gradients():
    generator = torch.Generator().manual_seed(17)
    z1 = torch.randn(3, 4, generator=generator, dtype=torch.double, requires_grad=True)
    z2 = torch.randn(3, 4, generator=generator, dtype=torch.double, requires_grad=True)
    labels = torch.tensor([7, 9001, 2**40])
    actual = _supcon(z1, z2, labels, 0.2)
    expected = _nt_xent(z1, z2, 0.2)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, (z1, z2)),
        torch.autograd.grad(expected, (z1, z2)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)


def test_supcon_keeps_float32_numerics_inside_autocast():
    generator = torch.Generator().manual_seed(23)
    z1 = torch.randn(5, 4, generator=generator).to(torch.bfloat16).requires_grad_()
    z2 = torch.randn(5, 4, generator=generator).to(torch.bfloat16).requires_grad_()
    labels = torch.tensor([0, 0, 7, 7, 9001])
    expected = _supcon(z1, z2, labels, 0.2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = _supcon(z1, z2, labels, 0.2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(
        torch.autograd.grad(actual, (z1, z2)),
        torch.autograd.grad(expected, (z1, z2)),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_multiple_positives_average_log_probabilities_and_ignore_id_numbering():
    generator = torch.Generator().manual_seed(19)
    z1 = torch.randn(5, 4, generator=generator, dtype=torch.double, requires_grad=True)
    z2 = torch.randn(5, 4, generator=generator, dtype=torch.double, requires_grad=True)
    labels = torch.tensor([9001, 12, 9001, 9001, 12])
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    repeated = labels.tolist() * 2
    terms = []
    for i, label in enumerate(repeated):
        others = [j for j in range(len(repeated)) if j != i]
        probabilities = F.log_softmax(z[i] @ z[others].t() / 0.2, dim=0)
        positives = [k for k, j in enumerate(others) if repeated[j] == label]
        terms.append(-probabilities[positives].mean())
    expected = torch.stack(terms).mean()
    actual = _supcon(z1, z2, labels, 0.2)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, (z1, z2)),
        torch.autograd.grad(expected, (z1, z2)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)
    permutation = torch.tensor([4, 2, 0, 1, 3])
    renamed = torch.tensor([2**40, 3, 2**40, 2**40, 3])
    torch.testing.assert_close(
        _supcon(z1[permutation], z2[permutation], renamed[permutation], 0.2), actual
    )


@pytest.mark.parametrize("batch_size,temperature", [(1, 0.001), (3, 0.001), (3, 1e-8)])
def test_single_segment_excludes_self_and_has_finite_gradients(batch_size, temperature):
    z1 = torch.ones(batch_size, 4, dtype=torch.float16, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_()
    loss = _supcon(z1, z2, torch.full((batch_size,), 9001), temperature)
    assert loss.item() == pytest.approx(math.log(2 * batch_size - 1), abs=1e-4)
    loss.backward()
    assert z1.grad is not None and z2.grad is not None
    assert torch.isfinite(z1.grad).all() and torch.isfinite(z2.grad).all()


def test_loss_shares_global_positives_across_slots_and_supports_precomputed_features():
    ctx = Context(seed=21, device="cpu")
    encoder = torch.nn.Linear(3, 5)
    head = torch.nn.Linear(5, 4)
    calls = []

    def encode(modality, values):
        calls.append(modality)
        return SimpleNamespace(pooled=encoder(values))

    model = SimpleNamespace(encode=encode, apply_head=lambda name, values: head(values))
    views = torch.randn(4, 2, 3)
    labels = torch.tensor([7000, 12, 7000, 12])
    batch = {
        name: {K.SLOT_VIEWS: views[indices], K.SLOT_LABELS: labels[indices]}
        for name, indices in (("short", slice(0, 2)), ("long", slice(2, 4)))
    }
    objective = SupervisedContrastive(slots=["short", "long"], head="segment")
    raw = objective.compute(batch, model, ctx)["supcon"]
    assert calls == ["image"] * 4
    calls.clear()
    for slot in batch.values():
        slot[K.SLOT_FEATS] = encoder(slot.pop(K.SLOT_VIEWS))
    cached = objective.compute(batch, model, ctx)["supcon"]
    assert calls == []
    expected = _supcon(head(encoder(views[:, 0])), head(encoder(views[:, 1])), labels, 0.2)
    torch.testing.assert_close(raw, expected)
    torch.testing.assert_close(cached, expected)
    parameters = tuple(encoder.parameters()) + tuple(head.parameters())
    for got, want in zip(
        torch.autograd.grad(raw, parameters),
        torch.autograd.grad(cached, parameters),
        strict=True,
    ):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("cached_slots", [("short",), ("short", "long")])
def test_batchnorm_features_gradients_and_buffers_match_cached_path(cached_slots):
    ctx = Context(seed=3, device="cpu")

    class Encoder(torch.nn.Sequential):
        modality = "image"
        out_dim = 5

    model = ComposedModel(
        encoders={
            "vision": Encoder(torch.nn.Linear(3, 5), torch.nn.BatchNorm1d(5), torch.nn.ReLU())
        },
        heads={"segment": {"encoder": "vision", "module": torch.nn.Linear(5, 4)}},
    )
    cached_model = copy.deepcopy(model)
    views = torch.randn(8, 2, 3)
    views[:4] += 20
    views[4:] -= 20
    labels = torch.tensor([7000, 12, 7000, 12] * 2)
    batch = {
        name: {
            K.SLOT_VIEWS: views[indices],
            K.SLOT_LABELS: labels[indices],
            K.SLOT_MODALITY: "image",
        }
        for name, indices in (("short", slice(0, 4)), ("long", slice(4, 8)))
    }
    cached_batch = copy.deepcopy(batch)
    cached_model.precompute_feats({name: cached_batch[name] for name in cached_slots}, ctx.device)
    for name in cached_slots:
        del cached_batch[name][K.SLOT_VIEWS]
    objective = SupervisedContrastive(slots=["short", "long"], head="segment")
    raw = objective.compute(batch, model, ctx)["supcon"]
    cached = objective.compute(cached_batch, cached_model, ctx)["supcon"]
    torch.testing.assert_close(raw, cached)
    raw.backward()
    cached.backward()
    for parameter, cached_parameter in zip(
        model.parameters(), cached_model.parameters(), strict=True
    ):
        assert parameter.grad is not None and cached_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, cached_parameter.grad)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, cached_model.state_dict()[name])


def test_invalid_supervision_fails_before_model_forward():
    ctx = Context(device="cpu")
    objective = SupervisedContrastive(slots=["segments"], head="segment")
    slot = {K.SLOT_FEATS: torch.zeros(3, 2, 4)}
    with pytest.raises(KeyError, match="carries no labels"):
        objective.compute({"segments": slot}, None, ctx)
    slot[K.SLOT_LABELS] = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="one label per frame"):
        objective.compute({"segments": slot}, None, ctx)
    slot[K.SLOT_LABELS] = torch.zeros(3, dtype=torch.long)
    slot[K.SLOT_FEATS] = torch.zeros(3, 1, 4)
    with pytest.raises(ValueError, match=">=2 views"):
        objective.compute({"segments": slot}, None, ctx)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="temperature"):
            SupervisedContrastive(slots=["segments"], head="segment", temperature=temperature)
