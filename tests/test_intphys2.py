"""IntPhys 2 probing: uniform frame sampling, pairing, and V-JEPA scoring."""

import csv
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("av")
import av
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from egolearn.evaluation import evaluate
from egolearn.evaluation.benchmarks.intphys2 import IntPhys2Probe
from egolearn.evaluation.benchmarks.intphys2 import probe as intphys2
from egolearn.evaluation.evaluator import EvaluationContext
from egolearn.evaluation.voe import read_video
from egolearn.model.build import build_model

pytestmark = pytest.mark.video

TYPES = ("1_Possible", "1_Impossible", "2_Possible", "2_Impossible")


def _write_video(path, values, size=32):
    """One constant-gray frame per value."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for v in values:
            img = np.full((size, size, 3), v, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        container.mux(stream.encode())


def _intphys2_root(tmp_path, monkeypatch, brightness=None):
    """One IntPhys 2 scene: four distinct 8-frame videos and their metadata.csv rows."""
    (tmp_path / "Videos").mkdir()
    rows = []
    for k, kind in enumerate(TYPES):
        name = f"Videos/{kind}.mp4"
        values = [(brightness or {}).get(kind, 20 * i + 9 * k) for i in range(8)]
        _write_video(tmp_path / name, values)
        rows.append({"SceneIndex": 0, "type": kind, "file_name": name})
    with open(tmp_path / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SceneIndex", "type", "file_name"])
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256((tmp_path / "metadata.csv").read_bytes()).hexdigest()
    monkeypatch.setattr(intphys2, "METADATA_SHA256", digest)
    return tmp_path


def test_read_video_samples_uniformly_over_the_whole_video(tmp_path):
    path = tmp_path / "ramp.mp4"
    _write_video(path, [6 * i for i in range(40)])
    clip = read_video(str(path), 4, T.PILToTensor())
    assert clip.shape == (4, 3, 32, 32)
    # frames 0, 13, 26, 39 -- the first and last frames included
    assert clip.float().mean(dim=(1, 2, 3)).tolist() == pytest.approx([0, 78, 156, 234], abs=4)


def test_intphys2_pairs_count_the_impossible_twin_as_more_surprising(tmp_path, monkeypatch):
    brightness = {"1_Possible": 50, "1_Impossible": 200, "2_Possible": 200, "2_Impossible": 50}
    root = _intphys2_root(tmp_path, monkeypatch, brightness)
    probe = IntPhys2Probe(name="intphys2", data_root=str(root), num_frames=4, context_frames=2)
    probe.prepare()
    assert probe.pairs == [
        (str(root / "Videos/1_Possible.mp4"), str(root / "Videos/1_Impossible.mp4")),
        (str(root / "Videos/2_Possible.mp4"), str(root / "Videos/2_Impossible.mp4")),
    ]
    assert probe.metadata() == {"n_pairs": 2, "metadata_sha256": intphys2.METADATA_SHA256}
    # brighter = more surprising: only the first pair has its impossible video brighter
    monkeypatch.setattr(probe, "surprise", lambda model, clips: {2: clips.mean(dim=(1, 2, 3, 4))})
    ctx = EvaluationContext(seed=0, device=torch.device("cpu"))
    assert probe.run(None, ctx) == {"voe_acc_ctx2": 0.5, "voe_acc": 0.5}


def test_voe_acc_pools_surprise_over_context_lengths(tmp_path, monkeypatch):
    root = _intphys2_root(tmp_path, monkeypatch)
    probe = IntPhys2Probe(name="intphys2", data_root=str(root), num_frames=8, context_frames=[2, 4])
    # [possible, impossible] surprise per pair; pair 1 is wrong at ctx 4 by more than it is
    # right at ctx 2, so its pooled surprise is wrong too
    surprise = iter(
        [
            {2: torch.tensor([1.0, 2.0]), 4: torch.tensor([3.0, 0.0])},
            {2: torch.tensor([1.0, 2.0]), 4: torch.tensor([0.0, 3.0])},
        ]
    )
    monkeypatch.setattr(probe, "surprise", lambda model, clips: next(surprise))
    ctx = EvaluationContext(seed=0, device=torch.device("cpu"))
    assert probe.run(None, ctx) == {"voe_acc_ctx2": 1.0, "voe_acc_ctx4": 0.5, "voe_acc": 0.5}


def test_intphys2_verifies_metadata(tmp_path):
    (tmp_path / "metadata.csv").write_text("SceneIndex,type,file_name\n")
    probe = IntPhys2Probe(name="intphys2", data_root=str(tmp_path))
    with pytest.raises(ValueError, match="unexpected SHA-256"):
        probe.prepare()


def test_intphys2_reports_missing_videos(tmp_path, monkeypatch):
    root = _intphys2_root(tmp_path, monkeypatch)
    (root / "Videos/2_Impossible.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="1 of 4"):
        IntPhys2Probe(name="intphys2", data_root=str(root)).prepare()


class _GridModel:
    """A (4, 2, 2) token grid behind encode/predict: records the split, predicts zeros."""

    def __init__(self, tokens):
        self.tokens, self.visible, self.predicted = tokens, [], []

    def encode(self, name, x, visible=None):
        if visible is None:
            return SimpleNamespace(tokens=self.tokens, aux={"grid": torch.tensor([4, 2, 2])})
        self.visible.append(visible[0].tolist())
        return SimpleNamespace(tokens=self.tokens.gather(1, visible[..., None].expand(-1, -1, 8)))

    def predict(self, name, tokens, *, ctx_pos, tgt_pos, grid):
        self.predicted.append((tgt_pos[0].tolist(), grid))
        return torch.zeros(*tgt_pos.shape, self.tokens.size(-1))


@pytest.mark.parametrize("distance", ["l1", "l2"])
def test_surprise_splits_the_token_grid_after_the_context(distance):
    tokens = torch.randn(2, 16, 8, generator=torch.Generator().manual_seed(0))
    model = _GridModel(tokens)
    probe = IntPhys2Probe(
        name="intphys2", data_root="", num_frames=8, context_frames=[2, 6], distance=distance
    )
    surprise = probe.surprise(model, torch.zeros(2, 8, 3, 32, 32))
    # tubelet 2 over 8 frames -> 4 time slices of 2x2 tokens: 2 frames = 4 tokens
    assert model.visible == [list(range(4)), list(range(12))]
    assert model.predicted == [(list(range(4, 16)), (4, 2, 2)), (list(range(12, 16)), (4, 2, 2))]
    targets = F.layer_norm(tokens, (8,))  # predictions are zero, so the error is the target
    error = targets.abs() if distance == "l1" else targets**2
    assert torch.allclose(surprise[2], error[:, 4:].mean(dim=(1, 2)))
    assert torch.allclose(surprise[6], error[:, 12:].mean(dim=(1, 2)))


def test_intphys2_scores_a_vjepa_model(tmp_path, monkeypatch):
    root = _intphys2_root(tmp_path, monkeypatch)
    vit = {"_target_": "encoder:video_vit", "dim": 32, "depth": 1, "heads": 2, "tubelet": 2}
    model = build_model(
        {
            "encoders": {"video": vit, "video_target": {**vit, "frozen": True}},
            "predictors": {
                "pred": {
                    "inputs": ["video"],
                    "_target_": "predictor:vjepa",
                    "encoder_dim": 32,
                    "dim": 16,
                    "depth": 1,
                    "heads": 2,
                }
            },
        }
    )
    spec = {
        "_target_": "probe:intphys2",
        "name": "intphys2",
        "data_root": str(root),
        "num_frames": 8,
        "context_frames": [2, 4, 6],
        "size": 32,
    }
    imports = ["egolearn.evaluation.benchmarks.intphys2"]
    result = evaluate(model, benchmarks=[spec], imports=imports).results["intphys2"]

    # the evaluator reports exactly the decisions of the probe's own surprise
    kwargs: dict[str, Any] = {k: v for k, v in spec.items() if k != "_target_"}
    probe = IntPhys2Probe(**kwargs)
    single = IntPhys2Probe(**{**kwargs, "context_frames": 4})
    probe.prepare()
    assert probe.pairs is not None
    expected = dict.fromkeys(result, 0.0)
    with torch.no_grad():
        for pair in probe.pairs:
            clips = torch.stack([read_video(p, 8, probe.transform) for p in pair])
            surprise = probe.surprise(model.eval(), clips)
            # sweeping scores each context length exactly as that length alone
            assert torch.equal(single.surprise(model, clips)[4], surprise[4])
            scores = {f"voe_acc_ctx{c}": s for c, s in surprise.items()}
            scores["voe_acc"] = torch.stack(list(surprise.values())).mean(dim=0)
            for key, (possible, impossible) in scores.items():
                assert possible != impossible  # distinct videos -> distinct surprise
                expected[key] += float(impossible > possible) / len(probe.pairs)
    assert result == expected

    # L2 surprise scores the same context lengths
    l2 = evaluate(model, benchmarks=[{**spec, "distance": "l2"}], imports=imports)
    assert l2.results["intphys2"].keys() == result.keys()
    with pytest.raises(ValueError, match="distance"):
        evaluate(model, benchmarks=[{**spec, "distance": "cosine"}], imports=imports)

    # the context/target split must fall on a tubelet boundary
    with pytest.raises(ValueError, match="tubelet"):
        evaluate(model, benchmarks=[{**spec, "context_frames": [2, 3]}], imports=imports)
