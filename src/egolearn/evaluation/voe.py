"""Violation-of-expectation (VoE) probing: is a physically impossible video
more surprising than its matched possible twin?

A generic readout like four_afc: benchmark packages list the matched
(possible, impossible) video pairs (intphys2/, grasp/), this module owns
decoding and scoring.  Protocol (defaults: 64 frames, contexts 2-16):

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
  `voe_acc` first averages each video's surprise over the context lengths;
* `voe_auroc` ranks every video by that pooled surprise with no pairing:
  the probability that an impossible video outranks a possible one (ties
  count one half; chance = 1/2).  It is the single-video check that the
  pairwise score is not riding on pair-level artifacts;
* a benchmark may label each pair with groups (a test, its concepts);
  `voe_acc_<group>` and `voe_auroc_<group>` then score each group from the
  same pooled surprise.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from egolearn.config.registry import VIEWS
from egolearn.core import keys as K
from egolearn.data import views  # noqa: F401  (registers the eval views)
from egolearn.evaluation.probe import Probe


def read_video(path: str, num_frames: int, transform) -> torch.Tensor:
    """Sample `num_frames` distinct frames over the whole video -> [T, C, H, W].

    Raise ValueError if the video has fewer than `num_frames` frames.
    """
    import av  # optional dependency (video / train extras)

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames
        if not total:
            raise ValueError(f"{path}: the container does not report a frame count")
        if total < num_frames:
            raise ValueError(
                f"{path}: video has {total} frames, but num_frames={num_frames} "
                f"requires at least {num_frames}"
            )
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


def auroc(
    possible: Sequence[float] | np.ndarray, impossible: Sequence[float] | np.ndarray
) -> float:
    """P(an impossible video's surprise > a possible one's), ties counting one half."""
    possible = np.asarray(possible, dtype=np.float64).ravel()
    impossible = np.asarray(impossible, dtype=np.float64).ravel()
    if not len(possible) or not len(impossible):
        raise ValueError("AUROC needs at least one possible and one impossible video")
    diff = impossible[:, None] - possible[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / diff.size)


_GROUP_LABEL = re.compile(r"[a-z0-9_]+")


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
        self.groups: list[tuple[str, ...]] = []

    def load_pairs(self) -> Sequence[tuple[str, str] | tuple[str, str, Sequence[str]]]:
        """Matched (possible, impossible) video paths under `data_root`.

        An entry may carry a third element, the pair's group labels (each
        matching ``[a-z0-9_]+``); the scorer then also reports per-group metrics.
        """
        raise NotImplementedError

    def prepare(self) -> None:
        if self.pairs is None:
            pairs, groups = [], []
            for entry in self.load_pairs():
                possible, impossible, *rest = entry
                labels = tuple(rest[0]) if rest else ()
                for label in labels:
                    if not _GROUP_LABEL.fullmatch(label):
                        raise ValueError(f"group labels must match [a-z0-9_]+, got {label!r}")
                pairs.append((possible, impossible))
                groups.append(labels)
            if not pairs:
                raise ValueError(f"{self.name}: no (possible, impossible) pairs found")
            self.pairs, self.groups = pairs, groups

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
        pooled = []  # per pair: (possible, impossible) surprise averaged over the contexts
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
            pooled.append(scores["voe_acc"].tolist())
        metrics = {key: n / len(self.pairs) for key, n in hits.items()}
        table = np.asarray(pooled, dtype=np.float64)
        metrics["voe_auroc"] = auroc(table[:, 0], table[:, 1])
        for label in sorted({label for labels in self.groups for label in labels}):
            rows = table[[label in labels for labels in self.groups]]
            metrics[f"voe_acc_{label}"] = float((rows[:, 1] > rows[:, 0]).mean())
            metrics[f"voe_auroc_{label}"] = auroc(rows[:, 0], rows[:, 1])
        return metrics
