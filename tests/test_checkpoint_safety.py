"""Restricted checkpoint loading and explicit snapshot trust at both entry points."""

from pathlib import Path
from pickle import UnpicklingError

import pytest
import torch

from egolearn.evaluation import load_model
from egolearn.model.init import load_pretrained_weights
from egolearn.utils.checkpoint import read_checkpoint


class PickleSideEffect:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


@pytest.mark.parametrize("entry_point", ["evaluation", "initialization"])
@pytest.mark.parametrize("trust", [False, "false"])
def test_checkpoint_objects_are_rejected_without_executing_pickle(tmp_path, entry_point, trust):
    marker = tmp_path / "executed"
    checkpoint = tmp_path / "untrusted.pth"
    torch.save({"model": {}, "extra": PickleSideEffect(marker)}, checkpoint)
    with pytest.raises(UnpicklingError if trust is False else TypeError):
        if entry_point == "evaluation":
            load_model(checkpoint, trusted_snapshot=trust)
        else:
            load_pretrained_weights(torch.nn.Linear(2, 2), str(checkpoint), trusted_snapshot=trust)
    assert not marker.exists()


@pytest.mark.parametrize("state", [[], {}, {"model": None}, {"model": []}])
def test_checkpoint_requires_model_mapping(tmp_path, state):
    checkpoint = tmp_path / "invalid.pth"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="'model' mapping"):
        read_checkpoint(checkpoint)


def test_snapshot_weights_need_explicit_trust(toy_config, tmp_path):
    import egolearn
    from egolearn.utils.snapshot import save_snapshot

    pipe = egolearn.build(toy_config(tmp_path / "run"))
    pipe.driver.fit()
    snapshot = tmp_path / "snapshot.pth"
    save_snapshot(str(snapshot), pipe.driver)
    with pytest.raises(UnpicklingError):
        load_model(snapshot)
    with pytest.raises(UnpicklingError):
        load_pretrained_weights(pipe.model, str(snapshot))
    loaded = load_model(snapshot, trusted_snapshot=True)
    assert all(
        torch.equal(tensor, loaded.state_dict()[name])
        for name, tensor in pipe.model.state_dict().items()
    )


def test_checkpoint_null_encoders_reports_model_validation(tmp_path):
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": {}, "model_config": {"model": {"encoders": None}}}, checkpoint)
    with pytest.raises(ValueError, match="model needs at least one entry"):
        load_model(checkpoint)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_snapshot_trust_requires_a_boolean_before_reading(tmp_path, value):
    # A quoted YAML/environment value must never silently opt into unsafe pickle.
    with pytest.raises(TypeError, match="trusted_snapshot must be a bool"):
        read_checkpoint(tmp_path / "not-read.pth", trusted_snapshot=value)
