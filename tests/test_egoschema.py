"""EgoSchema benchmark: loaders, items, submission files and the packaged config end to end."""

import csv
import json
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_module

pytest.importorskip("av")

from egolearn.evaluation import evaluate
from egolearn.evaluation.benchmarks.egoschema import EgoSchemaBenchmark
from egolearn.evaluation.benchmarks.egoschema.data import load_answers, load_questions
from egolearn.model import ComposedModel
from egolearn.model.generative import GenerativeComponent

pytestmark = pytest.mark.video

UIDS = ["q-alpha", "q-beta", "q-gamma"]


class Letters(GenerativeComponent):
    """Answers B to every question whose text mentions "beta", else A."""

    input_modalities = ("text", "video")

    def generate(self, texts=None, images=None, videos=None):
        self.check_inputs(texts, images, videos)
        assert videos is not None and all(len(v[0].frames) == 3 for v in videos)
        return ["B" if "beta" in text else "A" for text in texts]


@pytest.fixture
def egoschema_root(tmp_path, counter_video):
    root = tmp_path / "egoschema"
    (root / "videos").mkdir(parents=True)
    rows = [
        {"q_uid": uid, "question": f"What happens in {uid}?"}
        | {f"option {i}": f"{uid} option {i}" for i in range(5)}
        for uid in UIDS
    ]
    (root / "questions.json").write_text(json.dumps(rows))
    (root / "subset_answers.json").write_text(json.dumps({"q-alpha": 0, "q-beta": 0}))
    for uid in UIDS:
        counter_video(root / "videos" / f"{uid}.mp4", seconds=3.0)
    return root


def _benchmark(root, **overrides):
    settings = dict(
        name="egoschema",
        questions_json=str(root / "questions.json"),
        answers_json=str(root / "subset_answers.json"),
        video_root=str(root / "videos"),
        workers=1,
        verify=False,
    )
    return EgoSchemaBenchmark(**{**settings, **overrides})


def _model():
    return ComposedModel({}, generative={"vlm": Letters()})


def test_loaders_verify_the_release_files(egoschema_root):
    with pytest.raises(ValueError, match="unexpected SHA-256"):
        load_questions(egoschema_root / "questions.json")
    questions = load_questions(egoschema_root / "questions.json", verify=False)
    assert [q.q_uid for q in questions] == UIDS and len(questions[0].options) == 5
    assert load_answers(egoschema_root / "subset_answers.json", verify=False) == {
        "q-alpha": 0,
        "q-beta": 0,
    }


def test_subset_scores_public_answers(egoschema_root, tmp_path):
    out = tmp_path / "out"
    benchmark = _benchmark(egoschema_root, submission_dir=str(out))
    result = evaluate(_model(), benchmarks=[benchmark], record_json=None)
    metrics = result.results["egoschema"]
    assert metrics["n_scored"] == 2.0 and metrics["acc"] == 0.5
    assert metrics["pred_frac_A"] == 0.5 and metrics["pred_frac_B"] == 0.5
    assert result.metadata["egoschema"]["n_answered"] == 2
    assert result.metadata["egoschema"]["video_fps_mean"] == 1.0  # 3 frames of 3 s
    assert result.configs["egoschema"]["max_frames"] == 64
    assert result.configs["egoschema"]["prompt_version"] == 1
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert [(r["uid"], r["pred"], r["answer"], r["frames"]) for r in rows] == [
        ("q-alpha", 0, 0, 3),
        ("q-beta", 1, 0, 3),
    ]
    assert not (out / "submission.csv").exists()


def test_full_set_writes_submission_files(egoschema_root, tmp_path):
    out = tmp_path / "out"
    benchmark = _benchmark(egoschema_root, subset_only=False, submission_dir=str(out))
    metrics = evaluate(_model(), benchmarks=[benchmark], record_json=None).results["egoschema"]
    assert metrics["n_scored"] == 3.0 and metrics["n_answered"] == 2.0
    with open(out / "submission.csv", newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["q_uid", "answer"],
            ["q-alpha", "0"],
            ["q-beta", "1"],
            ["q-gamma", "0"],
        ]
    assert json.loads((out / "submission.json").read_text()) == {
        "q-alpha": 0,
        "q-beta": 1,
        "q-gamma": 0,
    }


def test_benchmark_refuses_missing_videos_and_partial_submissions(egoschema_root):
    (egoschema_root / "videos" / "q-beta.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="1 of 2 EgoSchema videos"):
        _benchmark(egoschema_root).prepare()
    with pytest.raises(ValueError, match="need all 5,031"):
        _benchmark(egoschema_root, subset_only=False, max_questions=2, submission_dir="x")


def test_packaged_config_composes():
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="egoschema_eval", overrides=["vlm_url=http://host:8000/v1"])
    assert cfg.checkpoint is None
    vlm = cfg.model_config.model.generative.vlm
    assert vlm._target_ == "generative:openai_chat" and vlm.base_url == "http://host:8000/v1"
    assert cfg.benchmarks[0]._target_ == "benchmark:egoschema"


def test_egl_eval_runs_egoschema_against_a_served_model(egoschema_root, tmp_path, chat_server):
    pytest.importorskip("openai")

    def reply(body):
        text = body["messages"][-1]["content"][-1]["text"]
        assert body["messages"][-1]["content"][0]["type"] == "video_url"
        assert body["mm_processor_kwargs"] == {"fps": 1.0}
        return "(B)" if "q-beta" in text else "A."

    chat_server.reply = reply
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "egolearn.evaluate",
        "--config-name",
        "egoschema_eval",
        f"vlm_url={chat_server.url}",
        f"egoschema_root={egoschema_root}",
        f"reply_cache={tmp_path / 'replies'}",
        f"output_json={output}",
        "+benchmarks.0.verify=false",
        "benchmarks.0.workers=1",
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["results"]["egoschema"]["acc"] == 0.5
    assert result["checkpoint"] is None
    assert result["model_config"]["model"]["generative"]["vlm"]["video_input"] == "video"
    assert len(chat_server.requests) == 2
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert len(chat_server.requests) == 2  # the rerun reads the reply cache
