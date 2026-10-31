"""Ego4D NLQ benchmark: annotation loading, items, metrics and challenge predictions."""

import json

import pytest

pytest.importorskip("av")

from egolearn.evaluation import evaluate
from egolearn.evaluation.benchmarks.ego4d_nlq import Ego4DNLQBenchmark
from egolearn.evaluation.benchmarks.ego4d_nlq.data import load_split
from egolearn.model import ComposedModel
from egolearn.model.generative import GenerativeComponent

pytestmark = pytest.mark.video

REPLIES = {"where is the cup?": "1.0 - 2.0", "where is the key?": "0 - 0.5", "who came?": "?"}


class Windows(GenerativeComponent):
    input_modalities = ("text", "video")

    def generate(self, texts=None, images=None, videos=None):
        self.check_inputs(texts, images, videos)
        assert videos is not None and all(v[0].fps == pytest.approx(0.5) for v in videos)
        return [REPLIES[text.split("Query: ")[1].split("\n")[0]] for text in texts]


def _annotations(annotated=True):
    def query(text, start, end):
        return {"query": text} | (
            {"clip_start_sec": start, "clip_end_sec": end} if annotated else {}
        )

    return {
        "videos": [
            {
                "video_uid": "video-1",
                "clips": [
                    {
                        "clip_uid": "clip-1",
                        "video_start_sec": 10.0,
                        "video_end_sec": 14.0,
                        "annotations": [
                            {
                                "annotation_uid": "ann-1",
                                "language_queries": [
                                    query("where is the cup?", 1.0, 2.0),
                                    {"query": ""},
                                    query("where is the key?", 2.0, 3.0),
                                ],
                            }
                        ],
                    },
                    {
                        "clip_uid": "clip-2",
                        "video_start_sec": 0.0,
                        "video_end_sec": 4.0,
                        "annotations": [
                            {
                                "annotation_uid": "ann-2",
                                "language_queries": [query("who came?", 0.0, 4.0)],
                            }
                        ],
                    },
                ],
            }
        ]
    }


@pytest.fixture
def ego4d_root(tmp_path, counter_video):
    root = tmp_path / "ego4d"
    (root / "clips").mkdir(parents=True)
    (root / "nlq_val.json").write_text(json.dumps(_annotations()))
    (root / "nlq_test_unannotated.json").write_text(json.dumps(_annotations(annotated=False)))
    for uid in ("clip-1", "clip-2"):
        counter_video(root / "clips" / f"{uid}.mp4", seconds=4.0)
    return root


def _benchmark(root, **overrides):
    settings = dict(
        name="ego4d_nlq",
        annotations_json=str(root / "nlq_val.json"),
        clip_root=str(root / "clips"),
        workers=1,
    )
    return Ego4DNLQBenchmark(**{**settings, **overrides})


def _model():
    return ComposedModel({}, generative={"vlm": Windows()})


def test_loader_keeps_query_indices_of_skipped_queries(ego4d_root):
    split = load_split(ego4d_root / "nlq_val.json", "val")
    assert [(q.annotation_uid, q.query_idx) for q in split.queries] == [
        ("ann-1", 0),
        ("ann-1", 2),
        ("ann-2", 0),
    ]
    assert split.clips["clip-1"].duration == 4.0


def test_val_split_reports_official_metrics(ego4d_root, tmp_path):
    out = tmp_path / "out"
    result = evaluate(
        _model(), benchmarks=[_benchmark(ego4d_root, predictions_dir=str(out))], record_json=None
    )
    metrics = result.results["ego4d_nlq"]
    assert metrics["n_queries"] == 3.0 and metrics["unparsed"] == pytest.approx(1 / 3)
    assert metrics["r1_iou0.3"] == pytest.approx(1 / 3) and metrics["r5_iou0.5"] == pytest.approx(
        1 / 3
    )
    assert metrics["miou"] == pytest.approx(1 / 3)
    assert result.metadata["ego4d_nlq"]["n_clips"] == 2
    assert result.metadata["ego4d_nlq"]["video_frames_mean"] == 2.0  # 4 s at 0.5 fps
    predictions = json.loads((out / "predictions.json").read_text())
    assert predictions["challenge"] == "ego4d_nlq_challenge"
    first, second, third = predictions["results"]
    assert (first["clip_uid"], first["annotation_uid"], first["query_idx"]) == (
        "clip-1",
        "ann-1",
        0,
    )
    assert first["predicted_times"] == [[1.0, 2.0]] * 5
    assert second["query_idx"] == 2
    assert third["predicted_times"] == [[0.0, 0.0]] * 5


def test_test_split_writes_predictions_without_metrics(ego4d_root, tmp_path):
    out = tmp_path / "out"
    benchmark = _benchmark(
        ego4d_root,
        split="test",
        annotations_json=str(ego4d_root / "nlq_test_unannotated.json"),
        predictions_dir=str(out),
        max_clips=1,
    )
    metrics = evaluate(_model(), benchmarks=[benchmark], record_json=None).results["ego4d_nlq"]
    assert set(metrics) == {"n_queries", "unparsed"} and metrics["n_queries"] == 2.0
    assert len(json.loads((out / "predictions.json").read_text())["results"]) == 2


def test_benchmark_refuses_missing_clips_and_unknown_splits(ego4d_root):
    (ego4d_root / "clips" / "clip-2.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="1 of 2 NLQ clips"):
        _benchmark(ego4d_root).prepare()
    with pytest.raises(ValueError, match="split must be one of"):
        _benchmark(ego4d_root, split="train")


def test_packaged_config_composes():
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="ego4d_nlq_eval", overrides=["vlm_url=http://host:8000/v1"])
    assert cfg.checkpoint is None and cfg.benchmarks[0].split == "val"
    assert cfg.benchmarks[0].annotations_json.endswith("annotations/nlq_val.json")
