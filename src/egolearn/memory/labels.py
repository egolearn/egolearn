"""LabelStore: labels that can still be corrected after items are in memory.

Items in memory store a key instead of a label; the label is looked up here
when a batch is built, so it can still be corrected later.  The rules this
encodes (code-verified against both published systems):

  * ItemRefs stored in memory are IMMUTABLE.  An admission component (the
    part that decides what enters memory) may only emit an event once it can
    never change.  BabyCL enforces this with a watermark: a segment is only
    emitted once the stream is far enough past its end that nothing can
    change it.
  * Segment labels are NOT necessarily immutable.  The original Memory
    Storyboard reclusters its short window and retroactively relabels items
    already in long-term memory, then merges classes even later.  That is why
    labels live in this separate store: the admission component that created
    them writes here, and the batch composer reads the current value at batch
    time.  A revision is an explicit, logged operation (`assign` / `remap`),
    never a mutation of the stored items themselves.
"""

from __future__ import annotations


class LabelStore:
    """Maps an item's ref.key to its class id.  Merged classes are handled
    with redirect pointers (union-find style): old id -> surviving id."""

    def __init__(self, name: str = "labels"):
        self.name = name
        self._labels: dict[tuple, int] = {}
        self._redirect: dict[int, int] = {}
        self.next_label = 0
        self.version = 0  # +1 on every label change; snapshots record it

    # ---- writes (only admission components call these) ----------------------
    def new_label(self) -> int:
        lab = self.next_label
        self.next_label += 1
        return lab

    def assign(self, ref_keys, labels) -> None:
        """Assign labels to many keys at once.  Assigning a key that already
        has a label overwrites it -- Memory Storyboard's recluster step
        relies on this; BabyCL simply never reassigns."""
        for k, lab in zip(ref_keys, labels, strict=False):
            self._labels[k] = int(lab)
        self.next_label = max(self.next_label, max((int(lab) for lab in labels), default=-1) + 1)
        self.version += 1

    def remap(self, old: int, new: int) -> None:
        """Merge class `old` into class `new`: from now on `old` resolves to
        `new` (Memory Storyboard's prototype-merging step)."""
        new = self.canonical(new)
        if old == new:
            return
        self._redirect[old] = new
        self.version += 1

    # ---- reads (composer / losses) ------------------------------------------
    def canonical(self, label: int) -> int:
        seen = []
        while label in self._redirect:
            seen.append(label)
            label = self._redirect[label]
        for s in seen:  # path compression: point every hop straight at the final label
            self._redirect[s] = label
        return label

    def resolve(self, ref_key) -> int | None:
        lab = self._labels.get(ref_key)
        return None if lab is None else self.canonical(lab)

    # ---- snapshot -----------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "labels": dict(self._labels),
            "redirect": dict(self._redirect),
            "next_label": self.next_label,
            "version": self.version,
        }

    def load_state_dict(self, st: dict) -> None:
        self._labels = dict(st["labels"])
        self._redirect = dict(st["redirect"])
        self.next_label = st["next_label"]
        self.version = st["version"]
