# Changelog

All notable changes are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [SemVer](https://semver.org) (0.x: minor = features, patch = fixes).

## [Unreleased]

### Added

- Gradient-based linear decoding (`probe:linear_sgd`):
  stochastic train crops, frozen encoder, optional feature L2 normalization,
  SGD, LARC, cosine decay, and deterministic RNG isolation. Direct adapters are available
  for full ImageNet (`probe:imagenet_linear_sgd`, caller-supplied ImageFolder
  root) and Labeled-S (`probe:labeled_s_linear_sgd`).
- Direct Labeled-S linear probing (`benchmarks/labeled_s_linear/`,
  `probe:labeled_s_linear`): the canonical
  metadata is pinned and resolved in memory against a
  user-supplied dataset root using its frozen 2,886-row train boundary, with no
  generated machine-specific manifest. The generic linear scorer now accepts
  an in-memory classification split while retaining JSON-manifest compatibility.
- Shape-bias evaluation from Tartaglini et al. Experiment 3: pinned,
  deterministic 80%-size unaligned stimuli; a registered image-representation
  probe over all 57,600 novel shape/texture triplets.
- BabyView benchmarks (`benchmarks/babyview_4afc/`): zero-shot 4AFC (the
  Labeled-S 4150 protocol on the public BabyView object crops, scored by
  the unchanged probe:four_afc; the trial generator is an RNG-faithful
  port of the source repo's, and center-crop preprocessing is frozen by a
  3-models x 3-transforms comparison) plus a 72-class linear-probe split.
- Fixed-feature linear probing (`probe:linear`): held-out top-1 of a zero-initialized
  logistic regression fitted with full-batch L-BFGS on frozen,
  train-standardized features. Deterministic and RNG-free, so probing
  mid-training cannot perturb the run (tested); splits are frozen in the
  manifest.

### Changed

- Full ImageNet gradient decoding (`probe:imagenet_linear_sgd`) now defaults to
  224-pixel crops, matching BabyCL's training resolution. Set `size: 112` to
  reproduce Memory Storyboard's released evaluation command.
- Gradient linear-probe augmentations now use per-item derived seeds, so changing
  the data-loader worker count does not change the probe score.
- probe:four_afc moved from the labeled_s_4afc benchmark package to
  `evaluation/four_afc.py`: with two datasets using it, the scoring is
  generic and the benchmark packages own only their manifests (the
  promotion rule). Config `_imports_` paths are unchanged.

- Prefetch (`driver.prefetch`): the next batch is composed -- sampled,
  decoded, augmented -- while the current one trains on the GPU, in both
  loops. Valid because sampling state only changes when the stream advances;
  a component that breaks that rule declares `mutates_sampling_state` and
  the build then refuses to prefetch. Underneath, every batch's randomness
  now derives from its own sequence number, making a batch a pure function
  of (seed, sequence, memory contents): prefetch on/off provably changes
  nothing, a snapshot taken while a prefetch is in flight resumes exactly,
  and clips ride the worker pool alongside frames in one submission per
  batch instead of one per slot. Covered by equality and resume tests.
- V-JEPA video pretraining, with V-JEPA 2 as the reference design:
  `encoder:video_vit` (tubelet embedding, 3D rotary positions, and a
  visible-token forward so the context pass only pays for unmasked tokens),
  `predictor:vjepa`, `loss:vjepa` (multiblock masks on the token grid, a
  frozen EMA-target twin updated after each step, L1 in representation
  space), the `clip_rrc` view, and `unit: clip` offline batches
  (non-overlapping windows, the conventional video pretraining epoch).
  Mask randomness derives from (seed, step), so the determinism guarantees
  extend to it: same seed = same weights and resumed = uninterrupted, EMA
  target included, both covered by tests. Losses that run their own encoder
  forwards declare `own_forward`, and the drivers skip the shared pooled
  precompute for slots only such losses read.
- Parallel decode+augment: `loader.workers: N` fans stage two of batch
  loading (refs -> tensors) out to a spawn process pool. Every frame's
  augmentation randomness comes from its own derived seed, so the worker
  count never changes what a run computes (covered by a serial-vs-pool
  equality test); stream order and ref picking stay sequential.

### Fixed

- Determinism (from PR review): the configured seed now also seeds the
  global torch/python RNGs and snapshots carry their state, so the same
  seed gives bit-identical weights and a resumed run matches an
  uninterrupted one; resume restores RNG states after the stream replay
  instead of before it; the stream's tail (a partial advance quantum, and
  events flushed at the final session end) still trains; the channel
  tie-break in the session merge binds eagerly, so an utterance ending
  exactly on a frame time surfaces after that frame and its chunk keeps
  its final step; the LR schedule applies before the optimizer step, so
  warmup covers step 1; an unfinished gradient-accumulation window is
  applied when the stream ends, and on a max_steps stop its pending
  gradients travel inside the snapshot so a continued run folds them into
  one full window; snapshots taken after the stream ran dry resume without
  re-admitting the final session flush. All covered by weight-level tests.
- TCL no longer materializes the raw pixel tensor when precomputed
  features are used; W&B runs start on first use, so dry runs never open
  one; text slots with several frames per item are rejected at build; the
  mypy CI job type-checks against real torch.

### Added

- Composable model layer: named encoders (image/video/text) + heads +
  predictors behind one `ComposedModel` surface; encoder outputs wrapped in
  `EncoderOutput` (pooled always; tokens/spatial/aux fields are additive);
  batch slots declare their modality and route to the matching encoder
  (routing is fixed at construction: one trainable encoder per modality,
  frozen copies stay in eval mode and out of the optimizer); capability
  set derived from components and checked against probe requirements at
  build time; reserved generative interface.
- Streaming learning framework: stream adapters (SAYCam/BabyView/video/toy),
  tiered memories with revisable labels, admission event builders, slot-based
  batch recipes, SSL + multimodal losses, lockstep streaming driver and
  offline-IID driver, progress-clock LR schedule, world-state snapshots with
  deterministic resume, stage-latency profiling, wandb hook.
- Evaluation probes: Labeled-S 4AFC, visual two-word test, Baby Winoground.
- Hydra train entry point (`python -m egolearn.train`); registry-based
  component instantiation under the hydra shell.
- `video`/`train` extras; CI test job installs the train extra.
- Generic Slurm launchers and data-plane/full-pipeline benchmarks under
  `scripts/`.
- Repository scaffold: src-layout package with packaged hydra configs, tiered test
  setup, CI gates (lint/types/test/lock/gitleaks), pre-commit hooks, branch
  protection script, and governance docs.
