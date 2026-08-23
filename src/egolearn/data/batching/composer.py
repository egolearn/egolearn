"""Composer: takes the refs a recipe sampled and turns them into a loaded,
collated batch of tensors.

Two label fields are computed here at batch time instead of being stored:
  * seg.id          -- looked up in the slot's LabelStore at THIS moment, so
                       Memory Storyboard's retroactive relabeling of old
                       segments is visible to the loss;
  * seg.batch_label -- the item's position within the slot (BabyCL's TCL
                       labels).

Batch layout: ``batch[slot_name]`` is a dict with
  views    FloatTensor [B, V, C, H, W]  (clip slots: [B, V, T, C, H, W])
  modality str, what the views are ("image", "video", ...); the model routes
           each slot to the encoder of this modality
  labels   LongTensor  [B]            (present iff labels_from/batch_label)
  text_ids LongTensor  [B, T], text_len LongTensor [B]   (iff slot.text)
  weights  FloatTensor [B] or None    (from weight.sample if any item has it)
  meta     list of source items (refs + payload; read-only)
where B = n_items * frames_per_item (segment items are flattened to frames
with repeated labels, matching BabyCL's _gather_segments).
"""

from __future__ import annotations

import torch

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.loading import Loader
from egolearn.data.text import VocabTokenizer
from egolearn.memory.labels import LabelStore
from egolearn.memory.table import TieredMemory

from .recipe import Recipe, SlotSpec


class SlotUnfilled(Exception):
    def __init__(self, slot: str, got: int, want: int):
        super().__init__(f"slot {slot!r} filled {got}/{want}")
        self.slot = slot


class Composer:
    def __init__(
        self,
        loader: Loader,
        tokenizer: VocabTokenizer | None = None,
        label_stores: dict[str, LabelStore] | None = None,
    ):
        self.loader = loader
        self.tokenizer = tokenizer
        self.label_stores = label_stores or {}

    # ------------------------------------------------------------------
    def compose(
        self,
        recipe: Recipe,
        memories: dict[str, TieredMemory],
        stream_head: dict[str, list[dict]],
        ctx: Context,
    ) -> dict | None:
        # Draw the items for EVERY slot first -- refs only, no image decoding.
        # If a strict slot comes up short we find out before a single pixel is
        # decoded, so the other slots' decode+augment work is never wasted.
        # mcl's feeder does the same: it checks use_annot before gathering.
        try:
            draws = {}
            for slot in recipe.slots:
                items = self._draw(slot, memories, stream_head, ctx)
                if (slot.strict and len(items) < slot.n) or not items:
                    raise SlotUnfilled(slot.name, len(items), slot.n)
                draws[slot.name] = items
        except SlotUnfilled:
            if recipe.fallback is not None:
                return self.compose(recipe.fallback, memories, stream_head, ctx)
            return None
        return {slot.name: self._fill_slot(slot, draws[slot.name], ctx) for slot in recipe.slots}

    # ------------------------------------------------------------------
    def _draw(self, slot: SlotSpec, memories, stream_head, ctx) -> list[dict]:
        src = slot.source
        kind = src["kind"]
        if kind == "tier":
            return memories[src["memory"]].sample_tier(src["tier"], slot.n, ctx)
        if kind == "mix":
            return memories[src["memory"]].sample_mix(slot.n, src["fractions"], ctx)
        if kind == "stream_head":
            recent = stream_head.get(src["schema"], [])
            return list(recent[-slot.n :])
        raise ValueError(kind)

    def _fill_slot(self, slot: SlotSpec, items: list[dict], ctx) -> dict:
        rng = ctx.rng(f"composer.{slot.name}")
        store = None
        if slot.labels_from is not None:
            store = self.label_stores.get(slot.labels_from)
            if store is None:
                raise KeyError(f"slot {slot.name!r}: unknown label store {slot.labels_from!r}")

        views, labels, texts, weights, metas = [], [], [], [], []
        work: list[tuple] = []  # (ref, frame_idx, view_names, seed) for batch loading
        any_weight = False
        for pos, item in enumerate(items):
            K.require_keys(item, [K.VIDEO_REF], f"composer slot {slot.name!r}")
            ref = item[K.VIDEO_REF]

            label: int | None
            if slot.batch_label:
                label = pos
            elif store is not None:
                label = store.resolve(ref.key)
                if label is None:
                    raise K.KeyFlowError(
                        f"slot {slot.name!r}: label store {slot.labels_from!r} "
                        f"has no label for {ref.key!r} -- is the labeler "
                        "component wired before the emitter in `admission`?"
                    )
            else:
                label = None

            w = item.get(K.WEIGHT_SAMPLE)
            if slot.unit == "clip":
                # one contiguous clip per item; all its frames share the same
                # random augmentation parameters (temporal consistency)
                views.append(
                    self.loader.clip_views(ref, slot.views, slot.clip_len, slot.clip_stride, rng)
                )
                if label is not None:
                    labels.append(label)
                metas.append(item)
                weights.append(float(w) if w is not None else 1.0)
                if w is not None:
                    any_weight = True
            else:
                k = slot.frames_per_item if slot.unit == "segment" else 1
                frame_idx = self.loader.pick_frame_indices(ref, k, rng)
                for fi in frame_idx:
                    # each frame's augmentation seed depends on what the item
                    # IS (step, slot, position), so worker-pool loading gives
                    # the same tensors as in-process loading
                    seed = ctx.derive_seed(f"aug:{ctx.clock.step}:{slot.name}:{len(work)}")
                    work.append((ref, fi, slot.views, seed))
                    if label is not None:
                        labels.append(label)
                    metas.append(item)
                    weights.append(float(w) if w is not None else 1.0)
                    if w is not None:
                        any_weight = True
            if slot.text:
                texts.append(item.get(K.TEXT_UTTERANCE, "") or " ")
        if work:
            # refs were picked above in deterministic order; only the
            # decode+augment work fans out (possibly to worker processes)
            views.extend(self.loader.frame_views_batch(work))

        out = {
            # [B, V, C, H, W]; clip slots: [B, V, T, C, H, W]
            K.SLOT_VIEWS: torch.stack(views, dim=0),
            K.SLOT_MODALITY: slot.modality,  # what the views are; encoder routing reads this
            K.SLOT_META: metas,
            K.SLOT_WEIGHTS: torch.tensor(weights) if any_weight else None,
        }
        if labels:
            out[K.SLOT_LABELS] = torch.tensor(labels, dtype=torch.long)
        if slot.text:
            if self.tokenizer is None:
                raise K.KeyFlowError(
                    f"slot {slot.name!r} sets text=true but no tokenizer is configured"
                )
            ids, lens = self.tokenizer.batch(texts)
            out[K.SLOT_TEXT_IDS], out[K.SLOT_TEXT_LEN] = ids, lens
        return out
