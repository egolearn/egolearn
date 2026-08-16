# Memory Storyboard (reproduction project)

Reproduces [Memory Storyboard](https://arxiv.org/abs/2501.12254) (streaming
visual SSL with temporal segmentation and a two-tier memory).

**This project owns the paper's contribution, as 1:1 ports** (`components.py`,
registered on import via the config's `_imports_`):

* `greedy_hierarchical_clustering` / `get_cluster_score` — **verbatim** from
  the public repo's `utils.py` (forces num_clusters−1 cuts even when the
  score decreases; candidate scan `range(1, N−1, subsample)`; first-best
  tie-breaking)
* `admission:storyboard_recluster` — the recluster + prototype-merge logic from
  `train_storyboard_merging.py`: trigger on short-window turnover,
  retroactive relabeling through the LabelStore, prototypes sampled
  ≤32/class **from the long-term buffer** (the component binds `memory: frames`),
  raw-mean-then-normalize embeddings, dynamic quantile threshold
* `embedder:learner_tap` — MS segments with the learner's own backbone

Everything else in the config is the core toolbox: saycam adapter,
ntxent/tcl, simclr views, resnext50, fifo/reservoir_weighted tables with
per-tier `admit_stride` (short FIFO takes every 8th frame, long reservoir
takes all — as in the original samplers).

Run:

```bash
python tools/train.py --config projects/storyboard/configs/saycam.yaml --dry-run
python tools/train.py --config projects/storyboard/configs/saycam.yaml
```

Documented framework adaptations (algorithms untouched): label writes go
through the LabelStore (the original's separate short-window and long-buffer
relabel paths write the same mapping — see `components.py` docstring); the
recluster trigger is checked per frame instead of per training step (fires
at the same threshold crossing).

KrishnaCam: swap `stream:` for a frames-directory adapter at 10 fps.
 