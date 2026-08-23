# Roadmap

Consolidates the lab's egocentric-learning line into one ecosystem. Source repos —
[memory-storyboard](https://github.com/agentic-learning-ai-lab/memory-storyboard),
[lifelong-memory](https://github.com/agentic-learning-ai-lab/lifelong-memory),
[poodle](https://github.com/agentic-learning-ai-lab/poodle),
[midway-network](https://github.com/agentic-learning-ai-lab/midway-network) —
remain the frozen reproduction artifacts for their papers.

**Porting policy: nothing below is a commitment.** Components enter via triage: a
concrete need, agreed scope, and a code-owner-reviewed PR. Items are candidates
ordered by expected need; some may never be needed. Ports arrive de-duplicated,
seeded, tested, and free of hardcoded paths ([design/streaming.md](design/streaming.md)).

## Phase 0 — Scaffold ✅

Package skeleton, CI gates, pre-commit, branch protection, governance docs.

## Phase 1 — Streaming & video base

- [x] StreamIndex: deterministic, temporally ordered index of typed records;
      captions, transcripts, and in-stream QA interleave by timestamp
- [ ] Offline frame-extraction CLI + frame-dir reads; on-the-fly mp4 reader
      deferred
- [ ] Video metadata cache, one schema
- [x] Streaming dataset with mutable shared-memory labels and deterministic
      ordering
- [ ] Pair-consistent transforms; guaranteed-overlap two-crop
      (clip-consistent transforms shipped; guaranteed-overlap two-crop pending)
- [ ] Dataset registry: SAYCam, KrishnaCam, BabyView, Ego4D, Walking Tours —
      roots from config only

## Phase 2 — Replay, segmentation, losses

- [x] Index replay as retention policy × sampling policy, seeded, behind the
      `Sampler` surface; beyond-index memory forms arrive separately
- [x] Temporal segmentation core
- [x] SSL losses as composable modules

## Phase 3 — Models & one streaming trainer

- [ ] Embedding and text interface boundaries per design/streaming.md §5
- [ ] Backbones with attribution headers; ViT sibling
- [x] Composable `SSLModel(backbone, heads, losses)` replacing per-ablation
      classes
- [x] One streaming trainer — seeded, resumable, state-dict checkpoints —
      replacing the seven trainer scripts
- [x] Memory Storyboard lands as `src/egolearn/methods/storyboard` + `projects/storyboard`

## Phase 4 — Readout evals

- [ ] Linear probe: one implementation, no apex dependency
- [ ] kNN/SVM readout with explicit split files and validation-based selection

## Phase 5 — Memory & QA

- [ ] LLM client registry: providers, retries, disk cache, cost accounting,
      env-var keys
- [ ] Prompt templates as data files; Ego4D NLQ and EgoSchema adapters with
      metrics and submission formatting
- [ ] Caption store and digest with pluggable text encoder
- [ ] LifelongMemory reference as `projects/lifelong-memory`

## Phase 6 — Cluster launch & artifacts

- [ ] Hydra+submitit compute layer for NYU Torch
- [ ] Per-run code snapshotting
- [ ] Hub-hosted checkpoints and split files; retire Drive links

## Performance ledger (from PR review, 2026-08)

Deliberately NOT addressed yet: the determinism tests (same seed = same
weights; resumed = uninterrupted; stream tail trains) are the safety net,
and pipelining is exactly where determinism tends to break.  With those
tests green, run the bench scripts on real data first and let the numbers
decide the order.

- [x] Decode+augment now fans out to worker processes (`loader.workers`),
      with per-item seeds so the worker count never changes what a run
      computes.  Still open: double-buffering the (now parallel) batch
      composer against the GPU step, so data work and the GPU overlap too.
- [ ] Replay re-decodes the same frames every time it samples them (memory
      stores refs) — a small LRU cache of decoded tensors would help a lot.
- [ ] bf16 and channels_last on the backbone.
- [ ] poodle's seek-per-frame issue waits for the poodle integration.

## Ongoing

- BabyCL — continual visual and verbal learning, successor of Memory
  Storyboard — releases its code here when published.
- Further egocentric QA and streaming-memory work arrives via `-dev` branches.
