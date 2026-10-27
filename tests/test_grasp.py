"""GRASP Level 2 probing: manifest preparation, pairing by name, grouped VoE scoring."""

import hashlib
import json
import sys

import numpy as np
import pytest

pytest.importorskip("av")
import av
import torch

from egolearn.evaluation import evaluate
from egolearn.evaluation.benchmarks.grasp import GRASPProbe, prepare
from egolearn.evaluation.benchmarks.grasp.manifest import (
    CONCEPTS,
    TESTS,
    metric_label,
    pair_groups,
)
from egolearn.evaluation.evaluator import EvaluationContext
from egolearn.evaluation.voe import auroc
from egolearn.hub import sha256_file
from egolearn.model.build import build_model

pytestmark = pytest.mark.video


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


# brightness of every frame of each video; the implausible twin of the first Gravity
# pair is brighter (correct), of the second darker (wrong); both GravityInertia correct
BRIGHTNESS = {
    ("Gravity", "000", "P"): 50,
    ("Gravity", "000", "IP"): 200,
    ("Gravity", "001", "P"): 200,
    ("Gravity", "001", "IP"): 50,
    ("GravityInertia", "000", "P"): 50,
    ("GravityInertia", "000", "IP"): 200,
    ("GravityInertia", "001", "P"): 50,
    ("GravityInertia", "001", "IP"): 200,
}


def _level2(tmp_path, frames=8, first_frame_mismatch=()):
    """A tiny Level 2 tree: two tests, two scenes each, in P_/IP_ folders.

    Like real twins, both videos of a scene open on the same frame and differ
    afterwards; `first_frame_mismatch` names the scenes whose twins differ from frame 0.
    """
    level = tmp_path / "level2"
    for (test, scene, kind), value in BRIGHTNESS.items():
        folder = level / f"{kind}_{test}"
        folder.mkdir(parents=True, exist_ok=True)
        values = [128] + [value] * (frames - 1)
        if (test, scene) in first_frame_mismatch and kind == "IP":
            values[0] = 0
        _write_video(folder / f"{scene}.mp4", values)
    return level


def _manifest(tmp_path, **kwargs):
    level = _level2(tmp_path, **kwargs)
    manifest = tmp_path / "manifest.json"
    prepare.write_manifest(prepare.build_manifest(level), manifest)
    return level, manifest


def test_catalog_has_sixteen_tests_over_eight_concepts():
    assert len(TESTS) == 16
    assert CONCEPTS == (
        "collision",
        "continuity",
        "gravity",
        "inertia",
        "object_permanence",
        "solidity",
        "support",
        "unchangeableness",
    )
    assert metric_label("Gravity") == "gravity"
    assert metric_label("GravityInertia2") == "gravity_inertia2"
    assert metric_label("ObjectPermanence3") == "object_permanence3"
    assert pair_groups("GravitySupport") == (
        "test_gravity_support",
        "concept_gravity",
        "concept_support",
    )


def test_prepare_pairs_twins_by_name_and_records_first_frame_match(tmp_path):
    level = _level2(tmp_path, first_frame_mismatch=[("GravityInertia", "001")])
    manifest = prepare.build_manifest(level)
    assert manifest["benchmark"] == "grasp" and manifest["level"] == "level2"
    assert manifest["tests"] == {
        "Gravity": {"concepts": ["gravity"], "n_pairs": 2},
        "GravityInertia": {"concepts": ["gravity", "inertia"], "n_pairs": 2},
    }
    assert [(p["test"], p["scene"], p["first_frame_match"]) for p in manifest["pairs"]] == [
        ("Gravity", "000", True),
        ("Gravity", "001", True),
        ("GravityInertia", "000", True),
        ("GravityInertia", "001", False),
    ]
    assert manifest["pairs"][0]["possible"] == "P_Gravity/000.mp4"
    assert manifest["pairs"][0]["impossible"] == "IP_Gravity/000.mp4"
    assert len(manifest["videos"]) == 8
    for rel, video in manifest["videos"].items():
        assert video == {"sha256": sha256_file(level / rel), "frames": 8}


def test_prepare_rejects_unknown_tests_missing_twins_and_name_mismatches(tmp_path):
    level = _level2(tmp_path)
    (level / "P_Levitation").mkdir()
    with pytest.raises(ValueError, match="unknown GRASP tests \\['Levitation'\\]"):
        prepare.discover_pairs(level)
    (level / "P_Levitation").rmdir()

    (level / "IP_GravityInertia").rename(level / "IP_GravityInertia_old")
    with pytest.raises(FileNotFoundError, match="no IP_GravityInertia twin"):
        prepare.discover_pairs(level)
    (level / "IP_GravityInertia_old").rename(level / "IP_GravityInertia")

    (level / "IP_Gravity/001.mp4").rename(level / "IP_Gravity/002.mp4")
    with pytest.raises(ValueError, match="Gravity: P_/IP_ video names differ"):
        prepare.discover_pairs(level)

    with pytest.raises(FileNotFoundError, match="no P_<Test> folders"):
        prepare.discover_pairs(tmp_path)


def test_prepare_rejects_videos_without_a_frame_count(tmp_path):
    level = _level2(tmp_path)
    for kind in ("P", "IP"):  # Matroska stores no frame count
        (level / f"{kind}_Gravity/000.mp4").unlink()
        _write_video(level / f"{kind}_Gravity/000.mkv", [128] * 8)
    with pytest.raises(ValueError, match=r"P_Gravity/000\.mkv: the container reports no frame"):
        prepare.build_manifest(level)


def test_prepare_main_writes_the_manifest_once(tmp_path, monkeypatch, capsys):
    level = _level2(tmp_path)
    output = tmp_path / "out/manifest.json"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--data_root", str(level), "--output", str(output)]
    )
    prepare.main()
    assert "Wrote 4 pairs over 2 tests" in capsys.readouterr().out
    assert len(json.loads(output.read_text())["pairs"]) == 4
    with pytest.raises(SystemExit):  # never overwrite a manifest
        prepare.main()


def test_auroc_counts_ties_as_one_half():
    assert auroc([1.0, 2.0, 3.0], [3.0, 3.0, 4.0]) == pytest.approx(8 / 9)
    assert auroc([0.0], [1.0]) == 1.0
    assert auroc([1.0], [0.0]) == 0.0
    assert auroc([1.0, 2.0], [1.0, 2.0]) == 0.5
    with pytest.raises(ValueError, match="at least one"):
        auroc([], [1.0])


def test_probe_groups_pairs_by_test_and_concept(tmp_path, monkeypatch):
    level, manifest = _manifest(tmp_path)
    probe = GRASPProbe(
        name="grasp", manifest=str(manifest), data_root=str(level), num_frames=4, context_frames=2
    )
    probe.prepare()
    assert probe.pairs == [
        (str(level / "P_Gravity/000.mp4"), str(level / "IP_Gravity/000.mp4")),
        (str(level / "P_Gravity/001.mp4"), str(level / "IP_Gravity/001.mp4")),
        (str(level / "P_GravityInertia/000.mp4"), str(level / "IP_GravityInertia/000.mp4")),
        (str(level / "P_GravityInertia/001.mp4"), str(level / "IP_GravityInertia/001.mp4")),
    ]
    assert probe.groups[0] == ("test_gravity", "concept_gravity")
    assert probe.groups[2] == ("test_gravity_inertia", "concept_gravity", "concept_inertia")
    assert probe.metadata() == {
        "n_pairs": 4,
        "tests": ["Gravity", "GravityInertia"],
        "n_tests": 2,
        "first_frame_mismatches": 0,
    }
    # brighter = more surprising, see BRIGHTNESS
    monkeypatch.setattr(probe, "surprise", lambda model, clips: {2: clips.mean(dim=(1, 2, 3, 4))})
    ctx = EvaluationContext(seed=0, device=torch.device("cpu"))
    assert probe.run(None, ctx) == {
        "voe_acc_ctx2": 0.75,
        "voe_acc": 0.75,
        "voe_auroc": 0.75,
        "voe_acc_concept_gravity": 0.75,
        "voe_auroc_concept_gravity": 0.75,
        "voe_acc_concept_inertia": 1.0,
        "voe_auroc_concept_inertia": 1.0,
        "voe_acc_test_gravity": 0.5,
        "voe_auroc_test_gravity": 0.5,
        "voe_acc_test_gravity_inertia": 1.0,
        "voe_auroc_test_gravity_inertia": 1.0,
    }


def test_probe_filters_tests_and_guards_the_videos(tmp_path):
    level, manifest = _manifest(tmp_path, first_frame_mismatch=[("Gravity", "000")])

    def probe_for(**kwargs):
        return GRASPProbe("grasp", str(manifest), str(level), **kwargs)

    probe = probe_for(tests="Gravity")
    probe.prepare()
    assert probe.metadata()["tests"] == ["Gravity"]
    assert probe.metadata()["n_pairs"] == 2
    assert probe.metadata()["first_frame_mismatches"] == 1

    with pytest.raises(ValueError, match="unknown GRASP tests"):
        probe_for(tests=["Gravity", "Levitation"])
    with pytest.raises(ValueError, match="without duplicates"):
        probe_for(tests=["Gravity", "Gravity"])
    with pytest.raises(ValueError, match="not in the manifest"):
        probe_for(tests=["Collision"]).prepare()

    (level / "IP_GravityInertia/001.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="1 of 8 GRASP videos are missing"):
        probe_for().prepare()
    probe_for(tests=["Gravity"]).prepare()  # the filtered-out test is not checked

    (level / "P_Gravity/000.mp4").write_bytes(b"corrupt")
    probe_for(tests=["Gravity"]).prepare()  # existence only by default
    with pytest.raises(ValueError, match="1 of 4 GRASP videos differ"):
        probe_for(tests=["Gravity"], verify_videos=True).prepare()

    doc = json.loads(manifest.read_text())
    doc["videos"]["IP_Gravity/001.mp4"]["frames"] = 0
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="1 of 4 GRASP videos report no frame count"):
        probe_for(tests=["Gravity"]).prepare()

    other = tmp_path / "other.json"
    other.write_text(json.dumps({"benchmark": "intphys2"}))
    with pytest.raises(ValueError, match="not a GRASP manifest"):
        GRASPProbe("grasp", str(other), str(level)).prepare()


def test_grasp_scores_a_vjepa_model(tmp_path):
    level, manifest = _manifest(tmp_path)
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
        "_target_": "probe:grasp",
        "name": "grasp",
        "manifest": str(manifest),
        "data_root": str(level),
        "num_frames": 8,
        "context_frames": [2, 4],
        "size": 32,
    }
    result = evaluate(model, benchmarks=[spec], imports=["egolearn.evaluation.benchmarks.grasp"])
    metrics = result.results["grasp"]
    assert set(metrics) == {
        "voe_acc_ctx2",
        "voe_acc_ctx4",
        "voe_acc",
        "voe_auroc",
        "voe_acc_concept_gravity",
        "voe_auroc_concept_gravity",
        "voe_acc_concept_inertia",
        "voe_auroc_concept_inertia",
        "voe_acc_test_gravity",
        "voe_auroc_test_gravity",
        "voe_acc_test_gravity_inertia",
        "voe_auroc_test_gravity_inertia",
    }
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    # the two tests have equally many pairs, so the overall accuracy is their mean
    tests = (metrics["voe_acc_test_gravity"] + metrics["voe_acc_test_gravity_inertia"]) / 2
    assert metrics["voe_acc"] == pytest.approx(tests)
    # every GravityInertia pair is also a gravity pair
    assert metrics["voe_acc_concept_inertia"] == metrics["voe_acc_test_gravity_inertia"]
    assert result.metadata["grasp"]["n_tests"] == 2
    assert result.configs["grasp"]["tests"] is None  # defaults are recorded
    assert result.configs["grasp"]["num_frames"] == 8

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest  # the manifest file is what provenance identifies; see input_provenance
