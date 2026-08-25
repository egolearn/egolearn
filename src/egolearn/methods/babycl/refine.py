"""VERBATIM port of BabyCL's segmentation math (mcl utils/online_ms_refine.py).

These six functions are copied 1:1 from the original; the only change is that
device-placement arguments were dropped (all tensors live on CPU here).  Do not
"clean up" this file -- matching the published code exactly is the point.
"""

from __future__ import annotations

import torch


def build_prefix_from_sim(S: torch.Tensor) -> torch.Tensor:
    prefix = torch.zeros((S.size(0) + 1, S.size(1) + 1), device=S.device, dtype=S.dtype)
    prefix[1:, 1:] = torch.cumsum(torch.cumsum(S, dim=0), dim=1)
    return prefix


def rect_sum(prefix: torch.Tensor, s: int, e: int) -> torch.Tensor:
    return prefix[e, e] - prefix[s, e] - prefix[e, s] + prefix[s, s]


def seg_score(prefix: torch.Tensor, s: int, e: int) -> torch.Tensor:
    L = max(1, e - s)
    return rect_sum(prefix, s, e) / L


def expand_chunk_by_similarity(prefix: torch.Tensor,
                               a: int, b: int,
                               left_bound: int, right_bound: int,
                               min_len: int,
                               max_len: int | None = None) -> tuple[int, int]:
    s, e = int(a), int(b)
    while True:
        cur = seg_score(prefix, s, e)
        best_gain = torch.tensor(0.0, device=prefix.device)
        best_move = None
        if s > left_bound and (max_len is None or (e - (s - 1)) <= max_len):
            sc = seg_score(prefix, s - 1, e); gain = sc - cur
            if gain > best_gain: best_gain = gain; best_move = ("L", s - 1, e)
        if e < right_bound and (max_len is None or ((e + 1) - s) <= max_len):
            sc = seg_score(prefix, s, e + 1); gain = sc - cur
            if gain > best_gain: best_gain = gain; best_move = ("R", s, e + 1)
        if best_move is None or float(best_gain.item()) <= 0.0:
            break
        _, ns, ne = best_move; s, e = ns, ne

    while (e - s) < max(1, min_len):
        cand = []
        if s > left_bound and (max_len is None or (e - (s - 1)) <= max_len): cand.append((s - 1, e))
        if e < right_bound and (max_len is None or ((e + 1) - s) <= max_len): cand.append((s, e + 1))
        if not cand: break
        best = None; best_sc = None
        for ns, ne in cand:
            sc = seg_score(prefix, ns, ne)
            if (best_sc is None) or (float(sc.item()) > float(best_sc.item())):
                best_sc = sc; best = (ns, ne)
        s, e = best
    return int(s), int(e)


def cut_gap_region_by_similarity(prefix: torch.Tensor,
                                 gs: int, ge: int,
                                 T_sub: int,
                                 min_seg_sub: int,
                                 subsample: int = 1,
                                 allow_last_short: bool = True) -> list[tuple[int, int]]:
    M = ge - gs
    if M <= 0: return []
    K = max(1, round(M / max(1, T_sub)))
    cps = []
    for _ in range(K - 1):
        best_cp, best_delta = None, -1e30
        parts = []; last = gs
        for cp in [*sorted(cps), ge]:
            parts.append((last, cp)); last = cp
        base = sum(float(seg_score(prefix, s, e).item()) for (s, e) in parts)
        for (s, e) in parts:
            if e - s < 2 * min_seg_sub: continue
            for cp in range(s + min_seg_sub, e - min_seg_sub + 1, subsample):
                sc = float(seg_score(prefix, s, cp).item() + seg_score(prefix, cp, e).item())
                delta = sc - base
                if delta > best_delta: best_delta = delta; best_cp = cp
        if best_cp is None: break
        cps.append(int(best_cp))
    cps = sorted(set(cps))
    segs = []
    last = gs
    for cp in [*cps, ge]:
        s, e = last, cp
        if e - s >= min_seg_sub: segs.append((s, e))
        last = cp
    if segs:
        tail = ge - segs[-1][1]
        if tail > 0 and (tail < min_seg_sub) and not allow_last_short:
            segs[-1] = (segs[-1][0], ge)
        elif tail > 0:
            segs.append((segs[-1][1], ge))
    else:
        segs = [(gs, ge)]
    return segs


def pick_representative_from_feats(feats: torch.Tensor, s: int, e: int) -> int:
    seg = feats[s:e]
    centroid = torch.nn.functional.normalize(seg.mean(dim=0, keepdim=True), dim=1)
    sims = (seg @ centroid.T).squeeze(1)
    return s + int(torch.argmax(sims).item())
