"""Composer: takes the refs a recipe sampled and turns them into a loaded,
collated batch of tensors.

Randomness: everything a batch draws -- which items are sampled from memory,
which frame inside a segment, where a clip starts, every augmentation
parameter -- derives from the batch's own sequence number (`compose_seq` on
the clock) and the run seed, never from a shared RNG stream.  A batch is
therefore a pure function of (seed, sequence number, memory contents): the
same batch composes identically whether it is built just-in-time, prefetched
while the previous step trains, or rebuilt by a resumed run whose crashed
predecessor had it in flight.

Loading is two-phase: refs for EVERY slot are drawn first (no pixels), then
the decode+augment work of the whole batch goes to the Loader in one call --
frames and clips together -- so a worker pool sees one well-packed submission
instead of one barrier per slot, and a strict slot coming up short is
discovered before a single pixel is decoded (mcl's feeder gates the same
way: it checks use_annot before gathering).

Two label fields are computed here at batch time instead of being stored:
  * seg.id          -- looked up in the slot's LabelStore at THIS moment, so
                       Memory Storyboard's retroactive relabeling of old
                       segments is visible to the loss;
  * seg.batch_label -- the item's position within the slot, for recipes that
                       request batch-local labels.

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

import random

import torch

from egolearn.core import keys as K
from egolearn.core.context import Context
from egolearn.data.loading import Loader, WorkItem
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
        seq: int | None = None,
    ) -> dict | None:
        """Build one batch.  Without `seq`, the sequence number is taken from
        the clock and committed immediately.  A driver that prefetches passes
        the seq it allocated and commits it only when the batch is handed to
        training, so a snapshot taken while a prefetch is in flight stays
        consistent (the resumed run rebuilds the identical batch)."""
        if seq is None:
            seq = ctx.clock.compose_seq
            ctx.clock.compose_seq = seq + 1
        rng = ctx.fresh_rng(f"compose.{seq}")
        return self._compose(recipe, memories, stream_head, ctx, seq, rng)

    def _compose(
        self,
        recipe: Recipe,
        memories: dict[str, TieredMemory],
        stream_head: dict[str, list[dict]],
        ctx: Context,
        seq: int,
        rng: random.Random,
    ) -> dict | None:
        # Draw the items for EVERY slot first -- refs only, no image decoding.
        try:
            draws = {}
            for slot in recipe.slots:
                items = self._draw(slot, memories, stream_head, rng)
                if (slot.strict and len(items) < slot.n) or not items:
                    raise SlotUnfilled(slot.name, len(items), slot.n)
                draws[slot.name] = items
        except SlotUnfilled:
            if recipe.fallback is not None:
                # same seq, same rng: the retry is part of this one batch
                return self._compose(recipe.fallback, memories, stream_head, ctx, seq, rng)
            return None

        # Fill: collect the decode+augment work of every slot, load it in ONE
        # Loader call (one well-packed pool submission), then scatter the
        # tensors back to their slots.
        filled = [self._fill_slot(slot, draws[slot.name], ctx, seq, rng) for slot in recipe.slots]
        if len(filled) == 1:
            out, work = filled[0]
            out[K.SLOT_VIEWS] = self.loader.tensor_batch(work)
            return {recipe.slots[0].name: out}
        tensors = self.loader.views_batch([w for _, work in filled for w in work])
        batch, pos = {}, 0
        for slot, (out, work) in zip(recipe.slots, filled, strict=True):
            out[K.SLOT_VIEWS] = torch.stack(tensors[pos : pos + len(work)], dim=0)
            pos += len(work)
            batch[slot.name] = out
        return batch

    # ------------------------------------------------------------------
    def _draw(self, slot: SlotSpec, memories, stream_head, rng: random.Random) -> list[dict]:
        src = slot.source
        kind = src["kind"]
        if kind == "tier":
            return memories[src["memory"]].sample_tier(src["tier"], slot.n, rng)
        if kind == "mix":
            return memories[src["memory"]].sample_mix(slot.n, src["fractions"], rng)
        if kind == "stream_head":
            recent = stream_head.get(src["schema"], [])
            return list(recent[-slot.n :])
        raise ValueError(kind)

    def _fill_slot(
        self, slot: SlotSpec, items: list[dict], ctx, seq: int, rng: random.Random
    ) -> tuple[dict, list[WorkItem]]:
        """Everything of one slot except the pixels: labels, text, weights,
        meta -- plus the WorkItems whose loaded tensors become slot.views."""
        store = None
        if slot.labels_from is not None:
            store = self.label_stores.get(slot.labels_from)
            if store is None:
                raise KeyError(f"slot {slot.name!r}: unknown label store {slot.labels_from!r}")

        work: list[WorkItem] = []
        labels, texts, weights, metas = [], [], [], []
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
                # augmentation parameters (temporal consistency)
                start = self.loader.pick_clip_start(ref, slot.clip_len, slot.clip_stride, rng)
                seed = ctx.derive_seed(f"aug:{seq}:{slot.name}:{len(work)}")
                work.append(("clip", ref, start, slot.views, slot.clip_len, slot.clip_stride, seed))
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
                    seed = ctx.derive_seed(f"aug:{seq}:{slot.name}:{len(work)}")
                    task: WorkItem = ("frame", ref, fi, slot.views, seed)
                    if K.VIEW_CONTEXT in item:
                        if slot.unit != "frame":
                            raise ValueError("contextual views currently require point frame slots")
                        task = (*task, item[K.VIEW_CONTEXT])
                    work.append(task)
                    if label is not None:
                        labels.append(label)
                    metas.append(item)
                    weights.append(float(w) if w is not None else 1.0)
                    if w is not None:
                        any_weight = True
            if slot.text:
                texts.append(item.get(K.TEXT_UTTERANCE, "") or " ")

        out = {
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
        return out, work
