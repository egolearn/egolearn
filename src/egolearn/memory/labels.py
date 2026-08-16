"""LabelStore: revisable labels behind an indirection layer.

The finalization contract (code-verified against both published systems):

  * ItemRefs stored in memory are IMMUTABLE.  An admission component may only
    emit an event once it can never change (BabyCL's watermark: finalize
    behind a safety margin).
  * Segment labels are NOT necessarily immutable.  Original Memory Storyboard
    reclusters its short window and retroactively relabels items already in
    long-term memory, then merges classes even later.  Therefore labels are
    resolved through this indirection layer -- owned by the admission
    component that writes them and dereferenced at batch-composition time.
    Revision is an explicit, loggable control-plane operation
    (`assign` / `remap`), never a mutation of stored items.
"""

from __future__ import annotations


class LabelStore:
    """ref.key -> class id, with merge redirects (union-find style)."""

    def __init__(self, name: str = "labels"):
        self.name = name
        self._labels: dict[tuple, int] = {}
        self._redirect: dict[int, int] = {}
        self.next_label = 0
        self.version = 0  # bumped on every revision; snapshots record it

    # ---- writes (admission-owned) ------------------------------------------
    def new_label(self) -> int:
        lab = self.next_label
        self.next_label += 1
        return lab

    def assign(self, ref_keys, labels) -> None:
        """Bulk (re)assignment. Reassignment IS allowed -- this is the MS
        recluster path; BabyCL simply never reassigns."""
        for k, lab in zip(ref_keys, labels, strict=False):
            self._labels[k] = int(lab)
        self.next_label = max(self.next_label, max((int(lab) for lab in labels), default=-1) + 1)
        self.version += 1

    def remap(self, old: int, new: int) -> None:
        """Merge class `old` into `new` (MS prototype merging)."""
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
        for s in seen:  # path compression
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
