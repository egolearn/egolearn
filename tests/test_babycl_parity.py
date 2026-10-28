"""Driver and stream options that reproduce mcl's BabyCL training loop.

Covers the step-indexed LR schedule, the exact max_steps cap, session-end
events ending an advance quantum, opt-in mixed precision and the
transcript-first tie order.
"""

import json
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import egolearn
from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.sources import base as S
from egolearn.data.sources.saycam import SaycamStream
from egolearn.memory.admission.base import EventBuilder
from egolearn.pipeline import PipelineError
from egolearn.training.driver import LockstepDriver
from egolearn.training.solver import ProgressCosine
from egolearn.utils.snapshot import load_snapshot, save_snapshot


@pytest.mark.parametrize("total", [26568, 1000, 150])
def test_step_indexed_schedule_matches_mcl_lambda_lr(total):
    # mcl: warmup max(100, int(0.02 * steps_needed)), linear (step + 1) / warm,
    # then cosine to 0; LambdaLR evaluates lambda(k) for the (k + 1)-th update
    warm = max(100, int(0.02 * total))

    def mcl_lambda(step):
        if step < warm:
            return float(step + 1) / float(max(1, warm))
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=mcl_lambda)
    schedule = ProgressCosine(warmup_frac=0.02, warmup_min_steps=100, total_steps=total)
    ctx = SimpleNamespace(clock=SimpleNamespace(step=0, progress=0.5))
    for step in sorted({0, 1, warm - 1, warm, warm + 1, total // 2, total - 1}):
        ctx.clock.step = step
        expected = mcl_lambda(step)
        assert schedule.multiplier(ctx) == pytest.approx(expected, abs=1e-12)
    for step in range(min(total, 300)):
        ctx.clock.step = step
        assert schedule.multiplier(ctx) == pytest.approx(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()


def test_step_indexed_schedule_rejects_a_nonpositive_budget():
    with pytest.raises(ValueError, match="total_steps"):
        ProgressCosine(total_steps=0)


def test_max_steps_caps_inside_a_quantum(toy_config, tmp_path):
    cfg = toy_config(tmp_path)
    cfg["driver"]["rate"] = {"steps_per_admit": 8, "resample": True}
    cfg["driver"]["max_steps"] = 5
    pipe = egolearn.build(cfg)
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 5


class _Stream(S.Stream):
    def __init__(self, lengths):
        self.lengths = lengths

    def sessions(self):
        for s, n in enumerate(self.lengths):
            sid = f"s{s}"
            times = [0.2 * i for i in range(n)]
            yield S.Session(
                sid,
                {"video": S.DenseFrameChannel("video", sid, times, [(sid, i) for i in range(n)])},
            )


class _EverySecondFrameAndAtSessionEnd(EventBuilder):
    """Finalizes a 'Tick' on each session's frame 2, and one more when it ends."""

    def start_session(self, sid):
        self.sid, self.seen, self.out = sid, 0, []

    def ingest(self, obs):
        self.seen += 1
        if self.seen == 3:
            self.out.append({K.EVENT_SCHEMA: "Tick", "at": (self.sid, self.seen - 1)})

    def end_session(self):
        self.out.append({K.EVENT_SCHEMA: "Tick", "at": (self.sid, "flush")})

    def poll(self):
        out, self.out = self.out, []
        return out


def test_events_flushed_at_session_end_end_the_quantum():
    param = torch.nn.Parameter(torch.zeros(1))
    # no batches are composed: the driver only advances the stream here
    unused: Any = None
    composer: Any = SimpleNamespace(loader=None)
    driver = LockstepDriver(
        stream=_Stream([4, 4]),
        admission=[_EverySecondFrameAndAtSessionEnd()],
        memories={},
        routes={},
        recipe=unused,
        composer=composer,
        model=unused,
        losses=[],
        optimizer=torch.optim.SGD([param], lr=0.1),
        schedule=None,
        ctx=Context(seed=0, device="cpu"),
        advance={"kind": "events", "schema": "Tick", "n": 1},
    )
    admitted = []
    while driver._advance_once():
        admitted.append(driver.ctx.clock.admitted)
    # frame 2 of s0; then s0's flush, which surfaces with s1's first frame;
    # then frame 2 of s1; then s1's flush at the end of the stream
    assert admitted == [3, 5, 7, 8]
    assert [ev["at"] for ev in driver.stream_head["Tick"]] == [
        ("s0", 2),
        ("s0", "flush"),
        ("s1", 2),
        ("s1", "flush"),
    ]


def _amp_config(toy_config, tmp_path, **amp):
    cfg = toy_config(tmp_path)
    cfg["driver"]["amp"] = amp
    return cfg


def test_amp_runs_named_slots_and_their_losses_in_float32(toy_config, tmp_path):
    pipe = egolearn.build(_amp_config(toy_config, tmp_path, dtype="bfloat16", fp32_slots=["curr"]))
    assert pipe.driver.scaler is None  # bfloat16 needs no loss scaling
    calls = []
    precompute = pipe.model.precompute_feats

    def recording(batch, device, skip=frozenset()):
        calls.append((sorted(set(batch) - set(skip)), torch.is_autocast_enabled("cpu")))
        return precompute(batch, device, skip=skip)

    pipe.model.precompute_feats = recording
    pipe.driver.fit()
    assert pipe.ctx.clock.step == 2
    # per step: the autocast pass has nothing left to encode, the fp32 pass encodes curr
    assert calls[:2] == [([], True), (["curr"], False)]


def test_amp_autocasts_unnamed_slots(toy_config, tmp_path):
    pipe = egolearn.build(_amp_config(toy_config, tmp_path, dtype="bfloat16"))
    seen = []
    precompute = pipe.model.precompute_feats

    def recording(batch, device, skip=frozenset()):
        seen.append(torch.is_autocast_enabled("cpu"))
        return precompute(batch, device, skip=skip)

    pipe.model.precompute_feats = recording
    pipe.driver.fit()
    assert seen and all(seen)


def test_amp_rejects_unknown_slots_and_settings(toy_config, tmp_path):
    with pytest.raises(PipelineError, match="fp32_slots"):
        egolearn.build(_amp_config(toy_config, tmp_path, fp32_slots=["pairs"]))
    with pytest.raises(ValueError, match="dtype"):
        egolearn.build(_amp_config(toy_config, tmp_path, dtype="float8"))


def test_float16_loss_scaler_travels_in_snapshots(toy_config, tmp_path):
    pipe = egolearn.build(_amp_config(toy_config, tmp_path, dtype="float16"))
    assert isinstance(pipe.driver.scaler, torch.amp.GradScaler)
    state = pipe.driver.scaler.state_dict()
    state.update(scale=128.0, _growth_tracker=5)
    pipe.driver.scaler.load_state_dict(state)
    path = save_snapshot(str(tmp_path / "snap.pth"), pipe.driver)

    resumed = egolearn.build(_amp_config(toy_config, tmp_path / "resumed", dtype="float16"))
    load_snapshot(path, resumed.driver)
    assert resumed.driver.scaler.get_scale() == 128.0
    assert resumed.driver.scaler.state_dict()["_growth_tracker"] == 5


@pytest.mark.parametrize("transcript_first", [False, True])
def test_transcript_first_breaks_equal_time_ties(tmp_path, transcript_first):
    session = tmp_path / "frames" / "S_20130101_0000_01"
    session.mkdir(parents=True)
    for t in ("0.00", "0.20", "0.40"):
        (session / f"S_20130101_0000_01_{t}.jpg").touch()
    transcript = tmp_path / "expand.json"
    transcript.write_text(
        json.dumps(
            [
                {
                    "video_filename": "S_20130101_0000_01.avi",
                    "utterance": "ball",
                    "utterance_num": 1,
                    "timestamps": [0.2, 0.4],
                }
            ]
        )
    )
    stream = SaycamStream(
        str(tmp_path / "frames"), transcript_json=str(transcript), transcript_first=transcript_first
    )
    (only,) = stream.sessions()
    order = [(obs.channel, obs.t) for obs in only.observations()]
    tie = [("transcript", 0.4), ("video", 0.4)]
    expected_tail = tie if transcript_first else tie[::-1]
    assert order[:2] == [("video", 0.0), ("video", 0.2)]
    assert order[2:] == expected_tail
