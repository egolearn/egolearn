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

- [x] Ordered stream sessions with time-aligned video and transcript channels
- [ ] Offline frame-extraction CLI + frame-dir reads; on-the-fly mp4 reader
      deferred
- [ ] Video metadata cache, one schema
- [x] Deterministic stream ordering and labels resolved at batch time through
      a revisable `LabelStore`
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
- [x] Composable models with named encoders, heads, predictors, and separate losses
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
- [x] Violation-of-expectation probes: IntPhys 2 and GRASP Level 2 pairwise
      accuracy and AUROC from V-JEPA surprise (`probe:intphys2`, `probe:grasp`)
- [ ] kNN/SVM readout with explicit split files and validation-based selection

## Phase 5 — Memory & QA

- [ ] LLM client registry: providers, retries, disk cache, cost accounting,
      env-var keys
- [ ] Prompt templates as data files; Ego4D NLQ and EgoSchema adapters with
      metrics and submission formatting
- [ ] Caption store and digest with pluggable text encoder
- [ ] LifelongMemory reference as `projects/lifelong-memory`

## Phase 6 — Cluster launch & artifacts

- [x] Site-neutral Slurm/Apptainer submission and Hydra sweep arrays
- [x] Per-run clean-commit code archives and submission records
- [x] Hub artifact mechanism: revisioned paths, verified fetch and private-by-default push
- [ ] Publish approved checkpoint/split collections and retire Drive links

## Performance ledger (from PR review, 2026-08)

Worker loading and prefetch are implemented, with equality and resume tests covering
their sampling behavior. The remaining candidates below require measurements on
real workloads before choosing an implementation.

- [x] Decode and augmentation use worker processes (`loader.workers`), with
      per-item seeds preserving sample choices and augmentations across supported
      worker counts. Frames and clips travel in one pool submission per batch.
- [x] Batch prefetch overlaps loading with training (`driver.prefetch`), using
      batch-specific seeds. Equality and resume tests cover supported configurations.
      Components that mutate sampling state between updates declare that constraint,
      preventing incompatible prefetch configurations.
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

Unpublished research stays on branches of the private development repository.
Project branches enter the public repository after the publication review
described in [RELEASING.md](../RELEASING.md).
