"""Violation-of-expectation (VoE) probing: is a physically impossible video
more surprising than its matched possible twin?

A generic readout like four_afc: benchmark packages list the matched
(possible, impossible) video pairs (intphys2/), this module owns decoding
and scoring.  Protocol (defaults: 64 frames, contexts 2-16):

* each video is sampled to `num_frames` frames spread uniformly over its
  whole duration, each preprocessed with the eval `view`;
* for every length C in `context_frames`, the first C frames are the
  context and the rest the target.  Surprise is the V-JEPA objective
  itself: the L1 distance between the predictor's output at the target
  tokens and the frozen target encoder's layer-normalized tokens there
  (model/losses/vjepa.py); `distance: l2` takes the mean squared error.
  All lengths share one decode and one target-encoder pass;
* a pair is correct iff the impossible video is strictly more surprising;
  chance = 1/2.  `voe_acc_ctx<C>` scores each context length alone;
  `voe_acc` first averages each video's surprise over the context lengths.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from egolearn.config.registry import VIEWS
from egolearn.core import keys as K
from egolearn.data import views  # noqa: F401  (registers the eval views)
from egolearn.evaluation.probe import Probe


def read_video(path: str, num_frames: int, transform) -> torch.Tensor:
    """`num_frames` frames spread uniformly over the whole video -> [T, C, H, W]."""
    import av  # optional dependency (video / train extras)

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames
        if not total:
            raise ValueError(f"{path}: the container does not report a frame count")
        idx = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
        frames = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in idx:
                frames[i] = transform(frame.to_image())
            if i == idx[-1]:
                break
    missing = sorted(set(idx) - frames.keys())
    if missing:
        raise ValueError(f"{path}: frame {missing[0]} of {total} did not decode")
    return torch.stack([frames[i] for i in idx])


class VoEProbe(Probe):
    """Pairwise VoE accuracy; subclasses implement `load_pairs`."""

    requires_capabilities = (K.CAP_VIDEO_REPRESENTATION, K.CAP_PREDICTION)

    def __init__(
        self,
        name: str,
        data_root: str,
        num_frames: int = 64,
        context_frames: int | Sequence[int] = (2, 4, 6, 8, 10, 12, 14, 16),
        distance: str = "l1",
        context: str = "video",
        target: str = "video_target",
        predictor: str = "pred",
        view: str = "resize",
        size: int = 224,
    ):
        contexts = [context_frames] if isinstance(context_frames, int) else list(context_frames)
        if not contexts or not all(0 < c < num_frames for c in contexts):
            raise ValueError("every context_frames must leave at least one target frame")
        if len(set(contexts)) != len(contexts):
            raise ValueError(f"context_frames has duplicates: {contexts}")
        if distance not in ("l1", "l2"):
            raise ValueError(f"distance must be 'l1' or 'l2', got {distance!r}")
        self.name = name
        self.data_root = data_root
        self.num_frames = num_frames
        self.context_frames = contexts
        self.distance = distance
        self.context = context
        self.target = target
        self.predictor = predictor
        self.transform = VIEWS.get(view)(size=size)
        self.pairs: list[tuple[str, str]] | None = None

    def load_pairs(self) -> list[tuple[str, str]]:
        """Matched (possible, impossible) video paths under `data_root`."""
        raise NotImplementedError

    def prepare(self) -> None:
        if self.pairs is None:
            self.pairs = self.load_pairs()
            if not self.pairs:
                raise ValueError(f"{self.name}: no (possible, impossible) pairs found")

    def metadata(self) -> dict:
        return {} if self.pairs is None else {"n_pairs": len(self.pairs)}

    def surprise(self, model, clips: torch.Tensor) -> dict[int, torch.Tensor]:
        """[B, T, C, H, W] clips -> {context length: [B] surprise of the frames after it}.

        Scores the way `loss:vjepa` trains; override it for other model families.
        """
        out = model.encode(self.target, clips)
        if out.tokens is None or "grid" not in out.aux:
            raise ValueError(f"encoder {self.target!r} returns no token grid (use e.g. video_vit)")
        targets = F.layer_norm(out.tokens, (out.tokens.size(-1),))
        gt, gh, gw = (int(g) for g in out.aux["grid"].tolist())
        pos = torch.arange(gt * gh * gw, device=clips.device).expand(clips.size(0), -1)
        scores = {}
        for c in self.context_frames:
            if c * gt % self.num_frames:
                raise ValueError(
                    f"context_frames must be multiples of the tubelet ({self.num_frames // gt})"
                )
            n_ctx = c * gt // self.num_frames * gh * gw
            ctx_pos, tgt_pos = pos[:, :n_ctx], pos[:, n_ctx:]
            cout = model.encode(self.context, clips, visible=ctx_pos)
            pred = model.predict(
                self.predictor, cout.tokens, ctx_pos=ctx_pos, tgt_pos=tgt_pos, grid=(gt, gh, gw)
            )
            diff = pred - targets[:, n_ctx:]
            scores[c] = (diff.abs() if self.distance == "l1" else diff.pow(2)).mean(dim=(1, 2))
        return scores

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        self.prepare()
        assert self.pairs is not None
        hits: dict[str, int] = {}
        for pair in self.pairs:
            clips = torch.stack([read_video(p, self.num_frames, self.transform) for p in pair])
            surprise = self.surprise(model, clips.to(ctx.device))
            scores = {f"voe_acc_ctx{c}": s for c, s in surprise.items()}
            scores["voe_acc"] = torch.stack(list(surprise.values())).mean(dim=0)
            for key, s in scores.items():
                if not torch.isfinite(s).all():
                    raise ValueError(f"{self.name}: non-finite surprise for {pair}")
                possible, impossible = s.tolist()
                hits[key] = hits.get(key, 0) + int(impossible > possible)
        return {key: n / len(self.pairs) for key, n in hits.items()}
