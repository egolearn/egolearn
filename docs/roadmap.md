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
- [x] Video model path: V-JEPA (video ViT + predictor + EMA loss, V-JEPA 2
      as the reference) trains on clip slots in both loops; VideoMAE-class
      masked modeling reuses the same sockets (mask samplers, visible-token
      forward, token outputs) and needs only its decoder + pixel loss
- [x] Composable `SSLModel(backbone, heads, losses)` replacing per-ablation
      classes
- [x] One streaming trainer — seeded, resumable, state-dict checkpoints —
      replacing the seven trainer scripts
- [x] Memory Storyboard lands as `src/egolearn/methods/storyboard` + `projects/storyboard`

## Phase 4 — Readout evals

- [x] Native checkpoint evaluation API and CLI, with model-only reconstruction,
      structured results, and standalone ImageNet gradient decoding

- [x] Novel-object shape-vs-texture bias probe with deterministic stimuli
- [x] Linear probes: `probe:linear` caches features from deterministic views
      and fits zero-init logistic regression by full-batch L-BFGS;
      `probe:linear_sgd` provides a stochastic-view, L2-normalized SGD/LARC/cosine
      decoder while isolating its RNG. Frozen adapters cover
      BabyView and Labeled-S; ImageNet supports stochastic-view SGD decoding.
- [x] Violation-of-expectation probe: IntPhys 2 pairwise accuracy from V-JEPA
      surprise (`probe:intphys2`)
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

- [x] Decode+augment fans out to worker processes (`loader.workers`), with
      per-item seeds so the worker count never changes what a run computes;
      frames and clips travel in one pool submission per batch.
- [x] The composer double-buffers against the GPU step (`driver.prefetch`):
      every batch's randomness derives from its own sequence number, so
      prefetch on/off is provably identical (equality + resume tests), and
      components that mutate sampling state between steps can declare it to
      opt out.
- [ ] Worker→trainer transport still pays pickle+pipe for every augmented
      tensor (~40MB/batch on BabyCL): move to shared-memory buffers, or ship
      uint8 and normalize on the GPU (needs views to declare a device
      stage — an interface change). Pinned staging for truly async H2D
      belongs to the same pass.
- [ ] Replay re-decodes the same frames every time it samples them (memory
      stores refs) — a small LRU cache of decoded tensors would help a lot.
- [ ] bf16 and channels_last on the backbone.
- [ ] poodle's seek-per-frame issue waits for the poodle integration.

## Ongoing

- BabyCL — continual visual and verbal learning — releases its code here when published.
- Further egocentric QA and streaming-memory work arrives via `-dev` branches.
