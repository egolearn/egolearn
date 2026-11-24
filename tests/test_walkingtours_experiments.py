"""Validate the shared WT input, visual BabyCL recipe, and frame preparation."""

import hashlib
import json
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

import egolearn
from egolearn.data import prepare_walkingtours
from egolearn.data.prepare_walkingtours import prepare_video
from egolearn.methods.babycl.data import FrozenResNeXt50Embedder


def recipe(project, name, root):
    directory = Path(__file__).resolve().parents[1] / "projects" / project / "configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        cfg = compose(config_name=name, overrides=[f"stream.root_dir={root}"])
    return OmegaConf.to_container(cfg, resolve=True)


def test_wt_recipes_share_all_ten_videos_and_sampling(tmp_path):
    mae = recipe("streammae", "full_wt_original_streammae", tmp_path)
    baby = recipe("babycl", "walkingtours", tmp_path)
    source = json.loads(
        files("egolearn.data").joinpath("manifests/walkingtours.json").read_text(encoding="utf-8")
    )
    assert mae["stream"] == baby["stream"]
    assert baby["stream"]["sessions"] == [v["session"] for v in source["videos"]]
    assert len(baby["stream"]["sessions"]) == 10
    assert baby["stream"]["sessions"][-1] == "wildlife"
    assert baby["admission"][0]["fps"] == 3.75
    assert baby["admission"][0]["embed_stride"] / 3.75 == 12
    assert [loss["_target_"] for loss in baby["losses"]] == ["loss:ntxent", "loss:supcon"]
    assert baby["driver"]["max_steps"] is None
    assert mae["run_control"]["stride"] == 1


@pytest.mark.parametrize("custom_manifest", [False, True])
def test_preparer_loads_manifest_outside_checkout(tmp_path, monkeypatch, custom_manifest):
    monkeypatch.chdir(tmp_path)
    expected = json.loads(
        files("egolearn.data").joinpath("manifests/walkingtours.json").read_text(encoding="utf-8")
    )
    arguments = [
        "prepare_walkingtours",
        "--source-root",
        "raw",
        "--output-root",
        "frames",
        "--workers",
        "1",
    ]
    if custom_manifest:
        expected = {"doi": "test-release", "videos": [{"session": "custom"}]}
        Path("custom.json").write_text(json.dumps(expected))
        arguments.extend(["--source-manifest", "custom.json"])
    monkeypatch.setattr(sys, "argv", arguments)
    prepared = []

    def prepare(source_root, output_root, video, threads):
        assert source_root == Path("raw")
        output_root.mkdir(exist_ok=True)
        prepared.append(video)
        return {"frames": 8, "duration_seconds": 2.0}

    monkeypatch.setattr(prepare_walkingtours, "prepare_video", prepare)
    prepare_walkingtours.main()
    inventory = json.loads(Path("frames/inventory.json").read_text())
    assert prepared == expected["videos"]
    assert inventory["source_doi"] == expected["doi"]
    assert inventory["sessions"] == [video["session"] for video in expected["videos"]]
    assert inventory["total_frames"] == 8 * len(expected["videos"])
    assert inventory["duration_seconds"] == 2.0 * len(expected["videos"])


def test_babycl_wt_trains_without_transcripts(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    folder = tmp_path / "frames" / "venice"
    folder.mkdir(parents=True)
    for index in range(1, 49):
        Image.new("RGB", (48, 32), (index * 4, 70, 120)).save(folder / f"frame_{index:08d}.png")
    cfg = recipe("babycl", "walkingtours", folder.parent)
    cfg.update(
        device="cpu",
        run_dir=str(tmp_path / "run"),
        record_json=None,
        loader={"workers": 0},
        hooks={"log_every": 1000, "checkpoint_every": 1000},
    )
    cfg["stream"]["sessions"] = ["venice"]
    cfg["admission"][0].update(
        avg_seg_frames=4,
        window_scale=2,
        embed_stride=1,
        compute_stride_frames=1,
        seg_extend_min_sec=0,
        seg_extend_max_sec=0,
    )
    cfg["memories"]["segments"]["tiers"]["short"]["capacity"] = 2
    cfg["memories"]["segments"]["tiers"]["long"]["capacity"] = 4
    cfg["recipe"]["slots"][0].update(n=2, frames_per_item=2)
    cfg["driver"]["prefetch"] = False
    cfg["driver"]["rate"]["steps_per_admit"] = 1
    for view in cfg["views"].values():
        view["size"] = 32
    cfg["model"]["encoders"]["vision"] = {"_target_": "encoder:toy_cnn", "dim": 16}
    for head in cfg["model"]["heads"].values():
        head["sizes"][:2] = [16, 16]
    monkeypatch.setattr(
        FrozenResNeXt50Embedder,
        "_lazy_model",
        lambda self: torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()),
    )
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    assert pipe.driver._exhausted
    assert pipe.ctx.clock.step > 0
    assert pipe.driver.nonfinite_steps == 0
    assert (tmp_path / "run/final.pth").is_file()


@pytest.fixture
def synthetic_video(tmp_path):
    source = tmp_path / "synthetic.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=30",
            "-frames:v",
            "64",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )
    return {
        "session": "synthetic",
        "filename": source.name,
        "bytes": source.stat().st_size,
        "md5": hashlib.md5(source.read_bytes()).hexdigest(),
    }


@pytest.mark.video
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_wt_preparation_resamples_checksums_and_reuses_complete_output(tmp_path, synthetic_video):
    video = synthetic_video
    output = tmp_path / "frames"
    record = prepare_video(tmp_path, output, video, 1)
    assert record["frames"] == 8
    assert record["source_fps"] == 30
    assert prepare_video(tmp_path, output, video, 1) == record
    (output / "synthetic/frame_00000008.jpg").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        prepare_video(tmp_path, output, video, 1)
    with pytest.raises(ValueError, match="checksum"):
        prepare_video(tmp_path, tmp_path / "other", {**video, "md5": "0" * 32}, 1)


@pytest.mark.video
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_known_source_truncation_keeps_duration_and_checksum_checks(
    tmp_path, synthetic_video, monkeypatch
):
    check_output = subprocess.check_output

    def long_header(command, **kwargs):
        result = check_output(command, **kwargs)
        if command[0] == "ffprobe":
            probe = json.loads(result)
            probe["streams"][0]["duration"] = "12.133333"
            result = json.dumps(probe).encode()
        return result

    monkeypatch.setattr(subprocess, "check_output", long_header)
    with pytest.raises(ValueError, match="found 8, expected about 45"):
        prepare_video(tmp_path, tmp_path / "unannotated", synthetic_video, 1)
    video = {
        **synthetic_video,
        "usable_duration_seconds": 64 / 30,
        "duration_note": "Test source has a known incorrect container duration.",
    }
    output = tmp_path / "annotated"
    record = prepare_video(tmp_path, output, video, 1)
    assert record["frames"] == 8
    assert record["duration_seconds"] == 64 / 30
    assert record["reported_duration_seconds"] == 12.133333
    assert record["duration_note"] == video["duration_note"]
    assert prepare_video(tmp_path, output, video, 1) == record
    with pytest.raises(ValueError, match="checksum"):
        prepare_video(tmp_path, tmp_path / "bad-source", {**video, "md5": "0" * 32}, 1)
    with pytest.raises(ValueError, match="frame count disagrees"):
        prepare_video(
            tmp_path, tmp_path / "bad-duration", {**video, "usable_duration_seconds": 6}, 1
        )
    with pytest.raises(ValueError, match="requires a duration_note"):
        prepare_video(tmp_path, tmp_path / "no-note", {**video, "duration_note": ""}, 1)
