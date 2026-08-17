"""Memory Storyboard's method-specific components (1:1 ports).

This module owns the paper's contribution; everything else the config uses
comes from the core toolbox:

  * greedy_hierarchical_clustering / get_cluster_score -- VERBATIM port of
    memory-storyboard/utils.py, keeping the exact greedy behavior: it forces
    num_clusters-1 cuts even when the score decreases, scans candidates over
    range(1, N-1, subsample), compares with strictly-greater, and initializes
    best_change_point to 1;
  * admission:storyboard_recluster -- port of the recluster + prototype-merge
    logic embedded in train_storyboard_merging.py (cluster_shortmem +
    merge_labels);
  * embedder:learner_tap -- MS clusters with the learner's own backbone
    (cluster_shortmem runs self.model.backbone).

Framework adaptations (documented here; everything else is verbatim):
  * label writes go through the LabelStore instead of mutating two buffers.
    The original relabels the short window via set_labels AND the long buffer
    via searchsorted over frames newer than the window start -- but those are
    the same frames (the window IS the newest span of the stream), so a
    single store.assign covers both paths.  Class merges become store.remap
    redirects, which subsume the original's long-buffer range rewrites (every
    frame of class j resolves to the merge target).
  * the recluster trigger is checked once per ingested frame instead of once
    per training step; it fires at the same label-threshold crossing, at most
    step_frame_size frames earlier than the original's check cadence.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F

from egolearn.config.registry import ADMISSION, EMBEDDERS
from egolearn.core import keys as K
from egolearn.core.refs import ItemRef
from egolearn.data.sources.base import Observation
from egolearn.memory.admission.base import EmbeddingProvider, EventBuilder

# ---------------------------------------------------------------------------
# VERBATIM port of memory-storyboard/utils.py (np.sort -> sorted() on a list
# of python ints is the only textual change; behavior identical).
# ---------------------------------------------------------------------------


def greedy_hierarchical_clustering(similarity_matrix, num_clusters=20, subsample=1):
    seq_length = similarity_matrix.shape[0]
    change_points = []
    for _ in range(num_clusters - 1):
        best_cluster_score = -float("inf")
        best_change_point = 1
        temp_change_points = change_points.copy()
        for change_point in range(1, seq_length - 1, subsample):
            if change_point not in temp_change_points:
                temp_change_points.append(change_point)
                cluster_score = get_cluster_score(similarity_matrix, temp_change_points)
                if torch.isnan(cluster_score).item():
                    raise ValueError("Cluster score is NaN")
                temp_change_points.pop()
                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_change_point = change_point
        change_points.append(best_change_point)

    return sorted(change_points)


def get_cluster_score(similarity_matrix, change_points):
    change_points = sorted(change_points)
    sum_all_clusters = 0
    for i in range(len(change_points) + 1):
        if i == 0:
            start_idx = 0
        else:
            start_idx = change_points[i - 1]
        if i == len(change_points):
            end_idx = len(similarity_matrix)
        else:
            end_idx = change_points[i]

        len_cluster = end_idx - start_idx
        sum_cluster = similarity_matrix[start_idx:end_idx, start_idx:end_idx].sum()

        sum_all_clusters += sum_cluster / len_cluster
    return sum_all_clusters


# ---------------------------------------------------------------------------
@EMBEDDERS.register("learner_tap")
class LearnerTapEmbedder(EmbeddingProvider):
    """Embeds frames with the training model's own backbone, read at a named tap
    (an intermediate feature output).  This matches MS's cluster_shortmem, which
    runs self.model.backbone on the short window with eval transforms."""

    def __init__(self, tap: str = "vision.pool2048", view: str = "eval", batch_size: int = 64):
        self.tap = tap
        self.view = view
        self.batch_size = batch_size

    @torch.no_grad()
    def embed_refs(self, refs: list[ItemRef], normalize: bool = True) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("learner_tap embedder requires the pipeline model")
        was_training = self.model.training
        self.model.eval()
        outs = []
        for i in range(0, len(refs), self.batch_size):
            imgs = [
                self.loader.frame_views(r, 0, [self.view])[0] for r in refs[i : i + self.batch_size]
            ]
            x = torch.stack(imgs, dim=0).to(self.ctx.device)
            outs.append(self.model.vision_tap(x, self.tap).cpu())
        if was_training:
            self.model.train()
        feats = torch.cat(outs, dim=0)
        return F.normalize(feats, dim=1) if normalize else feats


# ---------------------------------------------------------------------------
@ADMISSION.register("storyboard_recluster")
class StoryboardRecluster(EventBuilder):
    """Port of Memory Storyboard's segmentation step, which works by relabeling:
    instead of emitting segment objects, it rewrites the class labels of frames
    already in memory.

    Original control flow (train_storyboard_merging.py):
      * every frame starts with the default label global_idx // class_length;
      * when the short window's OLDEST default label reaches shortmem_base_label
        (i.e. the window has fully turned over since the last recluster), the whole
        window is reclustered: embed every window frame with the learner's backbone,
        normalize, build the cosine similarity matrix, and run
        greedy_hierarchical_clustering with
        num_clusters = (newest default label - oldest default label + 1);
      * the new labels are searchsorted(change_points, frame_pos, right) +
        oldest_default_label, written retroactively (store.assign);
      * then shortmem_base_label = newest default label;
      * merge_labels: once past `warmup`, sample <= prototype_samples frames per
        class FROM THE LONG-TERM BUFFER, cache the samples per class, re-embed ALL
        cached prototypes with the CURRENT backbone (raw features, mean per class,
        then normalize), and compute the dynamic threshold
        quantile(sim, 1 - 1/N - merge_threshold); each new class j merges into its
        most similar older class when their similarity is above the threshold
        (store.remap); finally merged_classes = shortmem_base_label.

    This component only writes labels: it emits no events.
    Requires bindings: label_store, embedder, memory (for prototypes).
    """

    subscribes = ("video",)
    emits = ()

    def __init__(
        self,
        window_frames: int,
        class_length: int,
        window_stride: int = 1,
        warmup: int = 0,
        merge: bool = True,
        merge_threshold: float = 0.003,
        prototype_samples: int = 32,
        prototype_tier: str = "long",
        cp_subsample: int = 1,
    ):
        """`window_frames`/`window_stride` mirror the short FIFO buffer: the original
        reclusters the SHORT BUFFER's contents, and that buffer subsamples the stream
        (--subsample, every 8th frame).  Set window_stride equal to the short tier's
        admit_stride so the recluster window holds exactly the same frames as the
        buffer.  `window_frames` counts stored entries (the FIFO's capacity), not raw
        stream frames."""
        self.window_frames = int(window_frames)
        self.class_length = int(class_length)
        self.window_stride = max(1, int(window_stride))
        self.warmup = int(warmup)
        self.merge = bool(merge)
        self.merge_threshold = float(merge_threshold)
        self.prototype_samples = int(prototype_samples)
        self.prototype_tier = prototype_tier
        self.cp_subsample = max(1, int(cp_subsample))

        # (ref.key, ref, default_label), newest right -- the short window
        self._window = deque(maxlen=self.window_frames)
        self._global_idx = 0
        self._base_label = 0  # shortmem_base_label
        self._merged_classes = 0
        # label -> cached sampled refs; re-embedded with the current model
        # every merge round (the original caches sampled IMAGES per class)
        self._prototype_refs: dict[int, list[ItemRef]] = {}

    def ingest(self, obs: Observation) -> None:
        if obs.channel != "video":
            return
        frame_idx = self._global_idx
        self._global_idx += 1
        if frame_idx % self.window_stride != 0:
            return
        default_label = frame_idx // self.class_length
        self._window.append((obs.ref.key, obs.ref, default_label))
        # trigger: original checks buffer[0]['label'] >= shortmem_base_label
        if len(self._window) == self._window.maxlen and self._window[0][2] >= self._base_label:
            self._recluster()

    # ------------------------------------------------------------------
    def _recluster(self) -> None:
        keys = [k for k, _, _ in self._window]
        refs = [r for _, r, _ in self._window]
        oldest_default = self._window[0][2]
        newest_default = self._window[-1][2]
        num_clusters = newest_default - oldest_default + 1

        # cluster_shortmem: embed all window frames, normalize, cosine matrix
        all_embds = self.embedder.embed_refs(refs, normalize=True)
        similarity_matrix = torch.mm(all_embds, all_embds.t())
        change_points = greedy_hierarchical_clustering(
            similarity_matrix, num_clusters=num_clusters, subsample=self.cp_subsample
        )

        class_labels = (
            torch.searchsorted(torch.tensor(change_points), torch.arange(len(refs)), side="right")
            + oldest_default
        )
        self.label_store.assign(keys, class_labels.tolist())

        self._base_label = newest_default
        if self.merge:
            self._merge_labels()
        self._merged_classes = self._base_label  # unconditional, as original

    def _merge_labels(self) -> None:
        if not (self._base_label > self.warmup):
            return
        if self.memory is None:
            raise RuntimeError(
                "storyboard_recluster merge=true requires a bound memory "
                "(config: `memory: <name>`) -- prototypes are sampled from "
                "the long-term buffer, as in the original"
            )

        # label -> refs currently resident in the long-term buffer
        label_to_refs: dict[int, list[ItemRef]] = {}
        for entry in self.memory.tiers[self.prototype_tier].entries:
            ref = entry.get(K.VIDEO_REF)
            if ref is None:
                continue
            lab = self.label_store.resolve(ref.key)
            if lab is None:
                continue
            label_to_refs.setdefault(lab, []).append(ref)

        rng = self.ctx.rng("admission.storyboard_recluster.prototypes")
        for lab, refs in label_to_refs.items():
            if len(refs) > self.prototype_samples:
                refs = rng.sample(refs, self.prototype_samples)
            self._prototype_refs[lab] = refs  # overwrite, like the image cache

        proto_labels = sorted(self._prototype_refs)
        if len(proto_labels) < 2:
            return
        label2pos = {l: i for i, l in enumerate(proto_labels)}

        # re-embed every cached prototype set with the CURRENT backbone:
        # raw features -> mean per class -> normalize the mean (original order)
        sampled_embeddings = []
        for l in proto_labels:
            emb = self.embedder.embed_refs(self._prototype_refs[l], normalize=False)
            sampled_embeddings.append(torch.mean(emb, dim=0))
        sampled_embeddings = torch.stack(sampled_embeddings, dim=0)
        sampled_embeddings = F.normalize(sampled_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(sampled_embeddings, sampled_embeddings.t())

        threshold = torch.quantile(
            similarity_matrix, 1 - 1 / len(similarity_matrix) - self.merge_threshold
        ).item()

        for j in range(max(self._merged_classes, 1), self._base_label):
            if j not in label2pos:
                continue
            pos_j = label2pos[j]
            older_labels = [l for l in proto_labels if l < j]
            if len(older_labels) == 0:
                continue
            older_pos = torch.tensor([label2pos[l] for l in older_labels])
            sims = similarity_matrix[older_pos, pos_j]
            max_value, idx = torch.max(sims, dim=0)
            if max_value > threshold:
                merge_into = older_labels[idx.item()]
                merge_into = self.label_store.canonical(merge_into)
                self.label_store.remap(j, merge_into)
