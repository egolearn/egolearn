# _official_iou and _official_nlq_metrics are adapted from EGO4D/episodic-memory
# NLQ/VSLNet/utils/evaluate_ego4d_nlq.py at 5275b9570a3d77884e8685bad474309f7778db1f
# (MIT, Copyright (c) 2021 EGO4D Dataset and Benchmark Suite), as a reference for
# the grounding metrics; the rest of this file is EgoLearn's own.
"""Video input sampling and the generic multiple-choice and temporal-grounding scorers."""

import pytest

pytest.importorskip("av")

import numpy as np
import torch

from egolearn.config.registry import GENERATIVE, PARSERS, PROMPTS
from egolearn.evaluation import load_model
from egolearn.evaluation.evaluator import EvaluationContext
from egolearn.evaluation.multiple_choice import (
    LETTERS,
    MultipleChoiceItem,
    MultipleChoiceScorer,
    format_prompt,
    parse_choice,
)
from egolearn.evaluation.temporal_grounding import (
    GroundingItem,
    TemporalGroundingScorer,
    compute_iou,
    parse_windows,
)
from egolearn.evaluation.video_input import VideoClip, VideoReader, VideoSampling, read_video
from egolearn.model import ComposedModel, build_model
from egolearn.model.encoders.image import ToyCNNEncoder
from egolearn.model.generative import GenerativeComponent

pytestmark = pytest.mark.video

CTX = EvaluationContext(seed=0, device=torch.device("cpu"))


class Scripted(GenerativeComponent):
    """Replies with ``answer(text)`` and keeps every request's videos."""

    input_modalities = ("text", "video")

    def __init__(self, answer):
        super().__init__()
        self.answer = answer
        self.videos = []
        self.texts = []

    def generate(self, texts=None, images=None, videos=None):
        self.check_inputs(texts, images, videos)
        self.videos.extend(videos or [])
        self.texts.extend(texts)
        return [self.answer(text) for text in texts]


def _model(answer):
    component = Scripted(answer)
    return ComposedModel({}, generative={"vlm": component}), component


# ---- video input ---------------------------------------------------------------------------


def test_sampling_spaces_frames_from_the_clip_start_and_caps_their_number():
    assert VideoSampling(fps=1.0, max_frames=64).times(4.0) == [0.0, 1.0, 2.0, 3.0]
    times = VideoSampling(fps=1.0, max_frames=4).times(180.0)
    assert times == [0.0, 45.0, 90.0, 135.0]
    assert VideoSampling(fps=0.1).times(3.0) == [0.0]
    with pytest.raises(ValueError, match="positive"):
        VideoSampling(fps=0.0)
    with pytest.raises(ValueError, match="duration"):
        VideoSampling().times(0.0)


def test_read_video_takes_the_frame_at_each_time_and_reports_the_rate(
    tmp_path, counter_video, frame_index
):
    path = tmp_path / "clip.mp4"
    counter_video(path, seconds=6.0, fps=10, size=(64, 48))
    video = read_video(VideoClip(str(path)), VideoSampling(fps=0.5, max_frames=64, max_side=32))
    assert video.timestamps == (0.0, 2.0, 4.0)
    assert video.fps == pytest.approx(0.5)
    assert [frame.size for frame in video.frames] == [(32, 24)] * 3
    assert [frame_index(f) for f in video.frames] == [0, 20, 40]


def test_read_video_window_counts_times_from_its_start(tmp_path, counter_video, frame_index):
    path = tmp_path / "clip.mp4"
    counter_video(path, seconds=6.0, fps=10)
    clip = VideoClip(str(path), start_sec=3.0, end_sec=5.0)
    video = read_video(clip, VideoSampling(fps=2.0, max_side=None))
    assert video.timestamps == (0.0, 0.5, 1.0, 1.5)
    assert video.frames[0].size == (64, 48)
    assert [frame_index(f) for f in video.frames] == [30, 35, 40, 45]


def test_read_video_seeks_across_long_gaps(tmp_path, counter_video, frame_index):
    path = tmp_path / "clip.mp4"
    counter_video(path, seconds=20.0, fps=10)  # keyframes every 2 s; frames 5 s apart
    video = read_video(VideoClip(str(path)), VideoSampling(fps=1.0, max_frames=4))
    assert video.timestamps == (0.0, 5.0, 10.0, 15.0)
    assert [frame_index(f) for f in video.frames] == [0, 50, 100, 150]


def test_read_video_rejects_a_file_shorter_than_the_clip(tmp_path, counter_video, frame_index):
    path = tmp_path / "clip.mp4"
    counter_video(path, seconds=6.0, fps=10)  # last frame at 5.9 s
    near = read_video(VideoClip(str(path), end_sec=6.5), VideoSampling(fps=2.0))
    assert frame_index(near.frames[-1]) == 59  # 6.0 s, within a second of the end
    with pytest.raises(ValueError, match=r"ends at 5\.9 s, before the frame at 7\.0 s"):
        read_video(VideoClip(str(path), end_sec=8.0), VideoSampling(fps=1.0))


def test_reader_decodes_each_clip_once_while_cached(tmp_path, counter_video, monkeypatch):
    import egolearn.evaluation.video_input as video_input

    paths = []
    for name in ("a", "b"):
        paths.append(tmp_path / f"{name}.mp4")
        counter_video(paths[-1], seconds=2.0)
    calls = []
    original = video_input.read_video

    def counting(clip, sampling):
        calls.append(clip)
        return original(clip, sampling)

    monkeypatch.setattr(video_input, "read_video", counting)
    reader = VideoReader(VideoSampling(), workers=2, cache_size=1)
    a, b = VideoClip(str(paths[0])), VideoClip(str(paths[1]))
    first = reader.read([a, a, b])
    assert first[0] is first[1] and len(calls) == 2
    reader.read([b])
    assert len(calls) == 2  # b is cached
    reader.read([a])
    assert len(calls) == 3  # a was evicted


# ---- multiple choice -----------------------------------------------------------------------


def test_prompt_lists_lettered_options_and_the_instruction():
    assert format_prompt(" Why? ", ["x", " y "]) == (
        "Question: Why?\nOptions:\nA. x\nB. y\n"
        "Answer with the option's letter from the given choices directly."
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("B", 1),
        ("(C)", 2),
        ("D. turn off the stove", 3),
        ("  a", None),
        ("C is washing dishes", None),
        ("Answer: B", 1),
        ("the answer is a dog", None),
        ("The answer is (E).", 4),
        ("cut onions.", 0),
        ("F", None),
        ("I cannot tell.", None),
    ],
)
def test_parse_choice(reply, expected):
    options = ["cut onions", "b", "c", "turn off the stove", "e"]
    assert parse_choice(reply, options) == expected


def test_multiple_choice_scorer_reports_accuracy_interval_and_letter_shares():
    replies = iter(["A", "B", "B", "nonsense"])
    model, _ = _model(lambda text: next(replies))
    items = [
        MultipleChoiceItem(f"q{i}", "Why?", ("x", "y", "z"), answer=answer)
        for i, answer in enumerate([0, 1, 0, 2])
    ]
    scorer = MultipleChoiceScorer("mcq", batch_size=3, bootstrap_resamples=2000)
    metrics, rows = scorer.evaluate(model, CTX, items)
    assert metrics["acc"] == 0.5 and metrics["n_answered"] == 4.0
    assert metrics["acc_ci_low"] <= 0.5 <= metrics["acc_ci_high"]
    assert metrics["unparsed"] == 0.25
    assert (metrics["pred_frac_A"], metrics["pred_frac_B"], metrics["pred_frac_C"]) == (
        0.25,
        0.5,
        0.0,
    )
    assert [row["pred"] for row in rows] == [0, 1, 1, None]


def test_multiple_choice_scorer_reads_each_item_video(tmp_path, counter_video):
    path = tmp_path / "v.mp4"
    counter_video(path, seconds=3.0)
    model, component = _model(lambda text: "A")
    item = MultipleChoiceItem("q", "Why?", ("x", "y"), video=VideoClip(str(path)), answer=None)
    scorer = MultipleChoiceScorer("mcq", sampling=VideoSampling(fps=1.0))
    metrics, rows = scorer.evaluate(model, CTX, [item])
    assert "acc" not in metrics and metrics["n_scored"] == 1.0
    (video,) = component.videos[0]
    assert video.timestamps == (0.0, 1.0, 2.0)
    assert (rows[0]["frames"], rows[0]["fps"]) == (3, 1.0)
    assert scorer.video_summary == {"video_frames_mean": 3.0, "video_fps_mean": 1.0}
    config = scorer.configuration()
    assert config["fps"] == 1.0
    assert (config["prompt"], config["parser"]) == ("lettered_options", "option_letter")


@PROMPTS.register("test_bare_question")
def bare_question(question, options):
    return question


@PARSERS.register("test_last_letter")
def last_letter(reply, options):
    named = [c for c in reply if c in LETTERS[: len(options)]]
    return LETTERS.index(named[-1]) if named else None


def test_multiple_choice_scorer_uses_the_named_prompt_and_parser():
    model, component = _model(lambda text: "I think A, no, B")
    item = MultipleChoiceItem("q", "Why?", ("x", "y"), answer=1)
    metrics, _ = MultipleChoiceScorer("mcq").evaluate(model, CTX, [item])
    assert metrics["unparsed"] == 1.0  # the default parser reads a leading letter only
    scorer = MultipleChoiceScorer("mcq", prompt="test_bare_question", parser="test_last_letter")
    metrics, _ = scorer.evaluate(model, CTX, [item])
    assert metrics["acc"] == 1.0 and component.texts[-1] == "Why?"
    assert scorer.configuration()["parser"] == "test_last_letter"
    with pytest.raises(KeyError, match="unknown component parser:nope"):
        MultipleChoiceScorer("mcq", parser="nope")


# ---- temporal grounding --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("12.5 - 20", [[12.5, 20.0]]),
        ("The moment is from 3s to 7.5 seconds.", [[3.0, 7.5]]),
        ("1:05 - 1:10", [[65.0, 70.0]]),
        ("40 - 30", [[30.0, 40.0]]),
        ("90 - 500", [[90.0, 100.0]]),
        ("1. 10.0 - 12.0\n2. 50 to 60", [[10.0, 12.0], [50.0, 60.0]]),
        ("around 12 seconds", []),
    ],
)
def test_parse_windows(reply, expected):
    assert parse_windows(reply, duration=100.0) == expected


def test_compute_iou_uses_the_hull_as_union():
    overlap = compute_iou([[0, 10], [0, 1]], [[5, 15]])
    assert overlap.shape == (2, 1)
    assert overlap[:, 0].tolist() == pytest.approx([5 / 15, 0.0])


def test_grounding_scorer_counts_recall_strictly_above_threshold(tmp_path, counter_video):
    path = tmp_path / "clip.mp4"
    counter_video(path, seconds=2.0)
    clip = VideoClip(str(path))
    windows = {"a": "0 - 10", "b": "0 - 5", "c": "no idea", "d": "90 - 100\n0 - 10"}
    model, component = _model(lambda text: windows[text.split("Query: ")[1].split("\n")[0]])
    items = [
        GroundingItem("a", "a", clip, duration=100.0, window=(0.0, 10.0)),  # IoU 1
        GroundingItem("b", "b", clip, duration=100.0, window=(0.0, 10.0)),  # IoU 0.5
        GroundingItem("c", "c", clip, duration=100.0, window=(0.0, 10.0)),  # unparsed
        GroundingItem("d", "d", clip, duration=100.0, window=(0.0, 10.0)),  # hit at rank 2
        GroundingItem("e", "a", clip, duration=100.0, window=None),  # hidden answer
    ]
    metrics, rows = TemporalGroundingScorer("nlq", batch_size=2).evaluate(model, CTX, items)
    assert metrics["n_queries"] == 5.0 and metrics["n_scored"] == 4.0
    assert metrics["r1_iou0.3"] == 0.5 and metrics["r1_iou0.5"] == 0.25
    assert metrics["r5_iou0.3"] == 0.75 and metrics["r5_iou0.5"] == 0.5
    assert metrics["miou"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)
    assert metrics["unparsed"] == 0.2
    assert rows[3]["windows"] == [[90.0, 100.0], [0.0, 10.0]]
    assert component.texts[0].startswith("The video is 100.0 seconds long.\nQuery: a\n")
    assert len(component.videos[0][0].frames) == 2  # 2 s at the default 1 fps


# ---- a component with trained parts --------------------------------------------------------


@GENERATIVE.register("test_encoder_readout")
class EncoderReadout(GenerativeComponent):
    """Stands in for an EgoLearn encoder + projector + language model: it encodes the
    frames it is given with an EgoLearn encoder and reads a reply out of the features."""

    input_modalities = ("text", "video")

    def __init__(self, dim: int = 8):
        super().__init__()
        self.encoder = ToyCNNEncoder(dim=dim)
        self.projector = torch.nn.Linear(dim, 2)

    def generate(self, texts=None, images=None, videos=None):
        self.check_inputs(texts, images, videos)
        replies = []
        for text, (video,) in zip(texts, videos, strict=True):
            pixels = np.stack([np.asarray(frame) for frame in video.frames])
            frames = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255
            score = self.projector(self.encoder(frames).mean(0))
            if "start - end" in text:  # a grounding prompt: the whole sampled span
                replies.append(f"{video.timestamps[0]:.1f} - {video.timestamps[-1]:.1f}")
            else:
                replies.append("AB"[int(score.argmax())])
        return replies


def test_a_component_wrapping_an_encoder_is_scored_and_reloaded(tmp_path, counter_video):
    spec = {"generative": {"vlm": {"_target_": "generative:test_encoder_readout", "dim": 8}}}
    model = build_model(spec)
    assert model.capabilities() == frozenset({"text_generation"})
    checkpoint = tmp_path / "readout.pth"
    state = {
        "model": model.state_dict(),
        "model_config": model.construction_config,
        "export_contract": model.export_contract(),
    }
    torch.save(state, checkpoint)
    loaded = load_model(checkpoint)
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value), name

    path = tmp_path / "v.mp4"
    counter_video(path, seconds=4.0)
    clip = VideoClip(str(path))
    item = MultipleChoiceItem("q", "Why?", ("x", "y"), video=clip, answer=0)
    metrics, _ = MultipleChoiceScorer("mcq").evaluate(loaded, CTX, [item])
    assert metrics["unparsed"] == 0.0 and metrics["n_answered"] == 1.0
    query = GroundingItem("g", "Where?", clip, duration=4.0, window=(0.0, 3.0))
    metrics, rows = TemporalGroundingScorer("nlq").evaluate(loaded, CTX, [query])
    assert rows[0]["windows"] == [[0.0, 3.0]] and metrics["r1_iou0.5"] == 1.0


def _official_iou(pred, gt):
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    return 1.0 * inter / union


def _official_nlq_metrics(predictions, truths, thresholds=(0.3, 0.5), topk=(1, 5)):
    results: list[list[list[bool]]] = [[[] for _ in topk] for _ in thresholds]
    average_iou = []
    for windows, truth in zip(predictions, truths, strict=True):
        overlap = _official_iou(windows, [list(truth)])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))
        for tt, threshold in enumerate(thresholds):
            for rr, k in enumerate(topk):
                results[tt][rr].append((overlap > threshold)[:k].any())
    return np.array(results).mean(axis=-1), float(np.mean(average_iou))


def test_grounding_metrics_match_the_official_evaluation(tmp_path, counter_video):
    rng = np.random.default_rng(0)
    path = tmp_path / "clip.mp4"
    counter_video(path, seconds=2.0)
    clip = VideoClip(str(path))
    windows, truths, items = [], [], []
    for i in range(200):  # 1 to 5 windows per query, inside [0, 100]
        starts = rng.uniform(0, 90, int(rng.integers(1, 6))).round(1)
        ends = (starts + rng.uniform(0.5, 10, starts.size)).round(1)
        windows.append([[float(s), float(e)] for s, e in zip(starts, ends, strict=True)])
        start = round(float(rng.uniform(0, 90)), 1)
        truths.append((start, round(start + float(rng.uniform(0.5, 10)), 1)))
        items.append(GroundingItem(str(i), f"q{i}", clip, duration=100.0, window=truths[-1]))
    replies = {f"q{i}": "\n".join(f"{s} - {e}" for s, e in w) for i, w in enumerate(windows)}
    model, _ = _model(lambda text: replies[text.split("Query: ")[1].split("\n")[0]])
    metrics, rows = TemporalGroundingScorer("nlq", batch_size=50).evaluate(model, CTX, items)
    assert [row["windows"] for row in rows] == windows
    recall, miou = _official_nlq_metrics(windows, truths)
    for tt, threshold in enumerate((0.3, 0.5)):
        for rr, k in enumerate((1, 5)):
            assert metrics[f"r{k}_iou{threshold}"] == pytest.approx(recall[tt][rr])
    assert metrics["miou"] == pytest.approx(miou)
    assert metrics["r5_iou0.3"] > metrics["r1_iou0.3"]  # several windows do matter here
