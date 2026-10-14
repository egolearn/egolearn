# Changelog

All notable changes are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [SemVer](https://semver.org) (0.x: minor = features, patch = fixes).

## [Unreleased]

### Added

- Landing page at the site root (`landing/`), with the docs site under `/docs/`;
  `scripts/docs/build.sh` builds both for Cloudflare Pages.
- `egl export` writes safetensors model weights and JSON construction metadata,
  contract, and portable source provenance. Evaluation and initialization share
  a reader for exported directories and legacy checkpoints; full snapshots need
  explicit trust. Model Hub uploads accept the exported pair only, and model
  references without a filename fail before downloading. Benchmark suites pin
  both exported files, including paired Hub downloads; environment diagnostics
  check the safetensors dependency.
- Complete `egl` lifecycle commands: a shared dataset catalog with access tiers,
  Hub fetch and checksum verification, benchmark preparation under one data root,
  read-only environment diagnostics, and dispatch to the cluster launcher.
  Dataset preparation help works without optional training dependencies.
- Pinned benchmark suites with frozen per-seed configs, local or Slurm-array
  execution, atomic run records, and complete-seed mean/SD tables with baseline
  comparisons. Training and evaluation can also opt into `record_json` directly.
- Add one-command Slurm/Apptainer submission with CPU environment preparation,
  a pinned lab image, scratch caches, arrays and immutable per-run code snapshots.
- Memory Storyboard and BabyCL as 1:1 method ports (`src/egolearn/methods/`)
  with reproduction configs under `projects/`.
- BabyCL offline training with complete storyboard/utterance indexing, mixed or
  pairs-only batches, utterance epochs, and deterministic prefetch/resume.
- Docs site skeleton: MkDocs Material with the lab theme, built from `docs/` plus
  the root and project READMEs, in staging mode (`noindex`) by default; a
  non-required `docs` CI job builds it with `--strict`.
- Torch-free Hub path resolution and checksummed, private-by-default dataset/checkpoint
  fetch and push commands; documented dataset access tiers. `init_from.checkpoint`
  accepts model Hub references and retains their resolved commit across snapshot resume.
- PR message guidance and template sections for Summary, optional Decisions,
  experiment Results, Test plan, and Checklist. The stdlib-only
  `scripts/check_pr_message.py` checks drafts locally and warns when project or
  YAML config changes have no Results section. Checklist checks use the template's
  top-level boxes and ignore nested task examples. Summary checks treat numbered
  items as bullets and require a separate prose paragraph; it is not a CI gate.
- Installed `egl train` and `egl eval` commands accept packaged config names or
  YAML paths, with `egolearn` as an equivalent command alias. Hydra handles later
  config-selection flags, including abbreviations, using its normal precedence.
  Explicit files retain access to packaged defaults and user-supplied search paths.
- Evaluation data guide with centralized SAYCam access and frame preparation,
  Labeled-S 4AFC manifest/vocabulary instructions, and BabyVLM task attribution.
  A tested `labeled_s_4afc.prepare` command converts frozen trials without resampling.
- `init_from` initializes selected model components from a checkpoint or snapshot
  before optimizer construction, with strict tensor checks and source provenance.
  Training rejects simultaneous `init_from` and `resume` before pipeline construction;
  snapshot resume preserves initialization provenance in subsequent checkpoints.

- Standalone evaluation API (`evaluate`, `evaluate_checkpoint`, `load_model`) and
  `python -m egolearn.evaluate`, with benchmark configs, structured JSON results,
  checkpoint provenance, and isolated evaluation RNGs. Existing probes can also
  run through this API. Pipeline checkpoints and snapshots now save resolved
  model construction metadata; older files accept an explicit model config.

- Stochastic-view SGD linear decoding (`probe:linear_sgd`): stochastic train crops,
  frozen encoder, optional feature L2 normalization, SGD, LARC, cosine decay,
  and deterministic RNG isolation. The probe seed controls decoder initialization,
  shuffling, augmentations, and worker seeding independently of the run seed.
  Per-item augmentation seeds keep scores independent of the data-loader worker
  count. Worker pools are reused across epochs within each probe run. Evaluation
  uses a registered `view` (default: `imagenet_center_crop`) with shared ImageNet
  normalization.
  Match `view` and `size` across linear probes to align evaluation inputs. The
  full ImageNet adapter (`benchmark:imagenet_linear_sgd`, caller-supplied
  ImageFolder root) defaults to 224-pixel crops. The Apex-derived LARC
  implementation retains its full upstream BSD-3-Clause license.
- Direct Labeled-S linear probing (`benchmarks/labeled_s_linear/`,
  `probe:labeled_s_linear`): the canonical metadata is pinned and resolved in
  memory against a user-supplied dataset root using its frozen 2,886-row train
  boundary, with no generated machine-specific manifest. The generic linear
  scorer now accepts an in-memory classification split while retaining
  JSON-manifest compatibility; both construction paths validate non-empty
  classes and splits and label ranges.
- Registered spatial relation categorization probe over 2D abstract scenes and
  3D rendered scenes: equal component/cell accuracy averaging, descriptive
  metrics, condition diagnostics, and separate monitoring/full profiles.

- Standalone full-versus-monitoring spatial comparison for external frozen DINO
  checkpoints, with per-trial scores, condition summaries, provenance hashes,
  and balanced-subset sampling/ranking checks.

- Fixed spatial monitoring subsets: 768 Exp. 1 trials and 512 Exp. 3 trials,
  balanced by condition and candidate order, selected reproducibly from the
  full datasets with parent hashes and original trial IDs preserved.

- Spatial relation categorization data preparation for Experiments 1 and 3:
  portable, checksummed images and labeled two-choice trials; balanced
  relation/target conditions, seeded candidate order, and provenance records.

- Shape-bias evaluation from Tartaglini et al. Experiment 3: pinned,
  deterministic 80%-size unaligned stimuli; a registered image-representation
  probe over all 57,600 novel shape/texture triplets.
- BabyView benchmarks (`benchmarks/babyview_4afc/`): zero-shot 4AFC (the
  Labeled-S 4150 protocol on the public BabyView object crops, scored by
  the unchanged probe:four_afc; the trial generator is an RNG-faithful
  port of the source repo's, and center-crop preprocessing is frozen by a
  3-models x 3-transforms comparison) plus a 72-class linear-probe split.
- Fixed-feature L-BFGS linear probing (`probe:linear`): held-out top-1 of a
  zero-initialized logistic regression fitted with full-batch L-BFGS on frozen,
  train-standardized features. Deterministic and RNG-free, so probing
  mid-training cannot perturb the run (tested); splits are frozen in the
  manifest.
- Packed frame store (`egolearn.data.packed`, `stream:packed_frames`): frames
  packed into large shard files with a numpy offset index, built and verified
  with `python -m egolearn.data.packed`; the same sessions, times, refs and
  pixels as `stream:saycam`, one read per frame and per contiguous clip, and a
  memory-mapped index instead of per-session listings, so Loader workers start
  in seconds instead of minutes. Cold random 256-frame batches drop from 1.2 s
  to 36 ms on network scratch. `Stream.open_frames` lets the Loader fetch a
  clip's frames in one call. See `docs/data_loading.md`.

### Changed

- Trim redundant training smoke tests and validation matrices; share resume and
  synthetic-config setup, and document how to keep new tests focused.

- `egl doctor --submit` checks submission tools and cluster storage on planning
  hosts, including CPU allocations. The documented benchmark workflow stops
  before creating a plan if diagnostics fail.
- Cluster instructions use the shell launcher on login nodes and run Python
  diagnostics and benchmark planning inside compute allocations.
- Benchmark baseline comparisons require matching seed sets and input keys/checksums;
  relocated identical inputs remain comparable, and mismatches produce no deltas.
- Run `scripts/slurm/train.sbatch` and `bench.sbatch` directly on the login node;
  passing either wrapper to `sbatch` now fails with migration instructions.
  Benchmark JSON is written to the isolated run's `work/bench.json`.

- Full ImageNet evaluation moves from `probe:imagenet_linear_sgd` to standalone
  `benchmark:imagenet_linear_sgd`. Dataset enumeration is deferred to preparation
  and cached across calls on the same benchmark instance. Remove ImageNet from
  training `probes` and use the packaged `imagenet_linear_eval` config; the former
  `ImageNetLinearSGDProbe` Python class is replaced by `ImageNetLinearBenchmark`.

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
- The SimCLR-style image views (`simclr_strong`, `babycl_ms_aug`) default to a
  separable Gaussian blur (`blur_impl=separable`: about 4x cheaper, within one
  intensity level of the reference, same RNG draws); `blur_impl=torchvision`
  keeps the bit-exact reference and `none` is the ablation. Loader workers
  flush denormal floats (`loader.flush_denormal`, default on): the blur's
  small-sigma kernels ran ~100x slower on subnormal tail weights that cannot
  change a rounded pixel; loaded tensors are identical with or without it
  (tested) and the trainer process is untouched. Frame-directory listings
  sort by (time, name), independent of `os.listdir` order.

### Fixed

- Keep a root 404 page in the combined site build and apply the staging flag
  consistently to the landing page and docs, including an explicitly empty value.
- All benchmark configs recorded by standalone evaluation reduce absolute local
  path settings to file or directory names, while preserving original top-level
  `hf://` references, structured metadata, and paths used to load benchmark data.

- Evaluation rejects non-finite metrics before recording a benchmark, preserving
  serializable partial results. Linear-probe configs record dataset and manifest
  names without machine-specific directories, alongside existing split hashes.

- Share restricted checkpoint reading between evaluation and initialization;
  full snapshots now require explicit trust, and source provenance stores only
  checkpoint file names and hashes.
- Record evaluation code revisions and linear-probe settings and split hashes,
  preserve completed benchmark results on failure, and restore MPS RNG state.
- Snapshot resume restores only saved `init_from` lineage, preserving the live
  model's construction metadata in subsequent checkpoints and snapshots.
- Slurm submission creates its code snapshot and run record with Bash and standard
  Unix tools, without invoking Python on the login node. Command arguments must be
  valid UTF-8.
- Benchmark entry points compose Hydra defaults in project configs; offline BabyCL
  retains its 5,000-step checkpoint cadence when inheriting shared settings.
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
