"""Direct-video loading against sequential decode, using synthetic H.264."""

import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egolearn.data.decoding.pyav import PyAVDecoder
from egolearn.data.loading import Loader
from egolearn.data.sources.video import VideoStream
from egolearn.data.views import ViewBank

av = pytest.importorskip("av")
pytestmark = pytest.mark.video


def _write_video(path, *, variable=False, offset=0):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(30000, 1001))
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 12
        stream.codec_context.max_b_frames = 2
        rng = np.random.default_rng(2)
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), format="rgb24"
            )
            frame.pts = offset + i + (i // 3 if variable else 0)
            frame.time_base = Fraction(1001, 30000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def videos(tmp_path):
    for name in ("20140901.mp4", "20140902.mp4"):
        _write_video(tmp_path / name, offset=7)
    return tmp_path


def _reference(path):
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def test_seek_windows_sequential_cache_and_pickle(videos):
    path = str(videos / "20140901.mp4")
    expected = _reference(path)
    decoder = PyAVDecoder(cache_handles=1, seek_gap=8)
    assert decoder.probe(path) == (pytest.approx(30000 / 1001), 48)
    picks = [47, 0, 17, 17, 12, 46, 3]
    for i, image in zip(picks, decoder.read_window(path, 0, 48, picks), strict=True):
        np.testing.assert_array_equal(image, expected[i])
    for i, image in decoder.iter_frames(path, 11):
        np.testing.assert_array_equal(image, expected[i])
    # A live native handle must never cross a spawn boundary.
    worker_decoder = pickle.loads(pickle.dumps(decoder))
    assert not worker_decoder._handles
    np.testing.assert_array_equal(worker_decoder.read_frame(path, 19), expected[19])
    decoder.read_frame(str(videos / "20140902.mp4"), 4)
    assert list(decoder._handles) == [str(videos / "20140902.mp4")]
    np.testing.assert_array_equal(decoder.read_frame(path, 9), expected[9])
    with pytest.raises(IndexError):
        decoder.read_frame(path, 48)
    with pytest.raises(IndexError):
        decoder.read_window(path, 3, 6, [-1])
    assert decoder.read_window(path, 0, 0) == []
    decoder.close()
    worker_decoder.close()
    assert not decoder._handles


def test_variable_rate_is_rejected(tmp_path):
    path = tmp_path / "variable.mp4"
    _write_video(path, variable=True)
    decoder = PyAVDecoder()
    try:
        with pytest.raises(ValueError, match="constant-rate"):
            decoder.read_window(str(path), 0, 48)
    finally:
        decoder.close()


def test_stream_groups_files_and_preserves_refs(videos, monkeypatch):
    stream = VideoStream(str(videos), PyAVDecoder())
    assert stream.session_ids() == ["20140901.mp4", "20140902.mp4"]
    assert stream.total_observations() == 96
    channels = [s.channels["video"] for s in stream.sessions()]
    assert channels[0].ref_at(30).t0 == pytest.approx(1.001)
    requests = [(channels[0].interval_ref(5, 20), 8), (channels[1].ref_at(3), 0)]
    requests += [(channels[0].ref_at(13), 0), (channels[0].ref_at(2), 0)]
    calls = []
    read_window = stream.decoder.read_window

    def record(*args):
        calls.append(args)
        return read_window(*args)

    monkeypatch.setattr(stream.decoder, "read_window", record)
    got = stream.open_frame_batch(requests)
    assert len(calls) == 2
    for image, (ref, index) in zip(got, requests, strict=True):
        np.testing.assert_array_equal(image, stream.open_frame(ref, index))
    with pytest.raises(IndexError):
        stream.open_frames(channels[0].interval_ref(5, 20), [15])
    assert stream.open_frame_batch([]) == []
    stream.close()


def test_loader_workers_preserve_augmented_frames_and_padded_clips(videos):
    stream = VideoStream(str(videos), PyAVDecoder())
    channel = next(iter(stream.sessions())).channels["video"]
    views = ViewBank(
        {
            "v": {"name": "simclr_strong", "size": 32},
            "clip": {"name": "clip_center_crop", "size": 32},
        }
    )
    items: list[tuple] = [("frame", channel.ref_at(i % 48), 0, ["v"], i) for i in range(60, 12, -1)]
    items.insert(3, ("clip", channel.interval_ref(46, 48), 0, ["clip"], 4, 1, 81))
    serial = Loader(stream, views)
    expected = serial.views_batch(items)
    pooled = Loader(stream, views, workers=2)
    try:
        actual = pooled.views_batch(items)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert actual[3].shape == (1, 4, 3, 32, 32)
        assert torch.equal(actual[3][:, 1], actual[3][:, 3])
    finally:
        pooled.close()
        serial.close()
    assert serial.frame_requests == pooled.frame_requests == 52
    assert not stream.decoder._handles


@pytest.mark.parametrize("config_name", ["streaming", "offline"])
def test_krishnacam_configs_train_from_video(videos, tmp_path, config_name):
    import egolearn

    config_dir = Path(__file__).resolve().parents[1] / "projects/krishnacam/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name)
    cfg.stream.root_dir = str(videos)
    cfg.device = "cpu"
    cfg.loader.workers = 0
    cfg.driver.prefetch = False
    cfg.driver.max_steps = 4
    cfg.run_dir = str(tmp_path / config_name)
    cfg.record_json = None
    cfg.views.v1.size = cfg.views.v2.size = 32
    if config_name == "streaming":
        cfg.driver.warmup.admitted = 4
        cfg.driver.advance.n = 24
        cfg.recipe.slots[0].n = 4
    else:
        cfg.driver.batch.n = 4
    pipe = egolearn.build(OmegaConf.to_container(cfg, resolve=True))
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 4
    assert not pipe.stream.decoder._handles


def test_discovery_and_manifest_validation(videos, tmp_path):
    parent = VideoStream(str(videos.parent), PyAVDecoder(), recursive=True)
    assert all(name.endswith(".mp4") for name in parent.session_ids())
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no video files"):
        VideoStream(str(empty), PyAVDecoder()).session_ids()
    explicit = VideoStream(str(videos), PyAVDecoder(), sessions=["20140902.mp4"])
    assert explicit.session_ids() == ["20140902.mp4"]


def test_video_benchmark_compares_identical_requests_and_bounded_jpegs(videos, tmp_path):
    import json

    from egolearn.data.video_benchmark import main

    output = tmp_path / "benchmark"
    assert (
        main(
            [
                "--root",
                str(videos),
                "--output-dir",
                str(output),
                "--workers",
                "0",
                "--batches",
                "1",
                "--warmup",
                "1",
                "--sessions",
                "2",
                "--segments",
                "2",
                "--frames",
                "2",
                "--seconds",
                "0.5",
                "--size",
                "32",
                "--jpeg-reference",
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    individual, grouped, jpeg = report["variants"]
    assert individual["first_measured_batch_sha256"] == grouped["first_measured_batch_sha256"]
    assert jpeg["kind"] == "jpeg"
    assert all(result["frame_requests"] == 4 for result in report["variants"])
    assert report["jpeg_preparation"]["frames"] <= 8
    assert report["jpeg_preparation"]["bytes"] > 0
    assert all(result["frames_per_s"] > 0 for result in report["variants"])


def test_babycl_krishnacam_trains_without_transcripts(videos, tmp_path):
    import torch.nn.functional as F

    import egolearn
    from egolearn.config.registry import EMBEDDERS
    from egolearn.memory.admission.base import EmbeddingProvider

    @EMBEDDERS.register("test_video_mean")
    class MeanEmbedder(EmbeddingProvider):
        def embed_refs(self, refs, normalize=True):
            values = torch.stack(
                [self.loader.frame_views(ref, 0, ["eval"])[0].mean(dim=(1, 2)) for ref in refs]
            )
            return F.normalize(values, dim=1) if normalize else values

    config_dir = Path(__file__).resolve().parents[1] / "projects/babycl/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="krishnacam_pilot")
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    assert plain["model"]["encoders"]["vision"]["_target_"] == "encoder:resnext50"
    assert len(plain["losses"]) == 2 and "tokenizer" not in plain
    plain.update(device="cpu", record_json=None, run_dir=str(tmp_path / "run"))
    plain["stream"]["root_dir"] = str(videos)
    plain["loader"]["workers"] = 0
    plain["embedders"] = {"cluster": {"_target_": "embedder:test_video_mean"}}
    for view in plain["views"].values():
        view["size"] = 32
    admission = plain["admission"][0]
    admission.update(
        fps=30000 / 1001,
        avg_seg_frames=12,
        embed_stride=3,
        compute_stride_frames=3,
        window_scale=2,
        seg_extend_min_sec=0.1,
        seg_extend_max_sec=0.2,
    )
    plain["memories"]["segments"]["tiers"]["short"]["capacity"] = 2
    slot = plain["recipe"]["slots"][0]
    slot.update(n=2, frames_per_item=2)
    plain["model"] = {
        "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": 16}},
        "heads": {
            "proj128": {"encoder": "vision", "_target_": "head:mlp", "sizes": [16, 16, 8]},
            "cls": {"encoder": "vision", "_target_": "head:linear", "in_dim": 16, "out_dim": 2000},
        },
    }
    plain["driver"].update(prefetch=False, max_steps=2)
    plain["driver"]["rate"]["steps_per_admit"] = 1
    plain["hooks"] = {}
    pipe = egolearn.build(plain)
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2
    assert pipe.ctx.clock.events_emitted > 0
    assert pipe.loader.frame_requests > 8  # training frames plus segmentation embeddings
    assert pipe.driver.nonfinite_steps == 0
