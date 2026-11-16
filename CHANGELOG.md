# Changelog

User-visible changes are grouped by kind, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [SemVer](https://semver.org), with minor releases for features and
patch releases for fixes during 0.x development.

## [Unreleased]

### Added

- MAE, ablation, and WalkingTours guides appear in site navigation and search, with dataset and hardware prerequisites.
- MAE ablations match streaming and shuffled update budgets, with separate consumption, retention, window, and throughput metrics.
- MAE provides ViT-S/16 and ViT-B/16 reconstruction recipes, a synthetic CPU example, and resumable full-pass training.
- StreamMAE adds stochastic depth, color jitter, DataDrop, and motion-biased crops with cached admission scores.
- WalkingTours preparation and training recipes support checksum-verified original videos and user-supplied WT++ frames.
- The Slurm launcher can requeue runners that save state and return pause exit code 75.

- `benchmark:imagenet_full_finetune` supports full-network ImageNet evaluation through `egl eval`, resumable adaptation, and optional execution optimizations.
- EgoSchema and Ego4D NLQ score generative models zero-shot through generic multiple-choice and temporal-grounding scorers with configurable prompts and reply parsers.
- `generative:openai_chat` queries OpenAI-compatible servers such as vLLM, caching replies on disk by full request.
- Composed models carry generative components, and `egl eval` builds a model from `model_config` when `checkpoint` is null.
- The docs site generates component, Python API, CLI, config, and benchmark references from source on every build.
- KrishnaCam IID configurations support instance-only SimCLR and temporal-only classification with explicit update budgets.
- Linear SGD evaluation exposes batch prefetching and worker persistence without changing sampled inputs.

- BabyCL supports verified lossless video caches after fixed spatial preprocessing,
  retaining original replay references and decoding requested frames on demand.
- PyAV supports optional NVDEC decoding, keeping intermediate frames on the GPU
  while preserving sampled frame indices.
- BabyCL supports visual KrishnaCam training, with GPU pilots and matched-request
  video-versus-JPEG loading benchmarks.
- `decoder:pyav` supports direct video batches, with streaming and offline
  KrishnaCam pilot configurations and deterministic worker loading.
- V-JEPA 2 provides pretraining components, matched IID and chronological streaming
  recipes, resumable runs, collapse diagnostics, throughput logs, and IntPhys evaluation.
- V-JEPA 2 evaluates static clips with existing Labeled-S, BabyView, and shape-bias
  scorers, recording the encoder and temporal input protocol.
- V-JEPA 2 runs support verified checkpoint migration to a new runtime, opt-in
  Slurm continuation, and preflight reporting of complete static source clips.
- The group supervisor coordinates checkpoint pauses and failures for independent
  V-JEPA 2 runs sharing an allocation.
- The loader supports optional packed and pinned batches, bounded decoded-frame
  caching, and stable worker assignment for replayed clips.
- Frame-directory streams support opt-in strict filenames, nested recording
  discovery, and gap-aware offline clip indexing.
- EgoLearn provides configurable streaming and offline training with replay memory,
  composable models and losses, evaluation hooks, snapshots, and stage timing.
- BabyCL and Memory Storyboard provide method implementations and reproduction
  configurations under `projects/`.
- BabyCL supports offline mixed-objective and pairs-only training with deterministic
  utterance epochs and snapshot continuation.
- V-JEPA supports masked video representation learning in streaming and offline
  training, including its EMA target in saved snapshots.
- `egl` and `egolearn` provide installed commands for training, evaluation, data,
  export, diagnostics, submission, and benchmark suites.
- `egl train` and `egl eval` accept packaged configuration names, YAML paths, and
  Hydra overrides.
- `init_from` copies selected model weights into a fresh run and records their
  source and selection in subsequent checkpoints.
- `evaluate`, `evaluate_checkpoint`, and `load_model` support standalone evaluation
  with model reconstruction, structured results, and checkpoint provenance.
- `egl export` writes safetensors weights and JSON model metadata for evaluation,
  initialization, and model publication.
- `egl bench` runs pinned benchmark suites locally or through Slurm arrays and
  generates complete-seed tables with baseline comparisons.
- Training and evaluation write atomic run records with input identities,
  checkpoint lineage, metrics, and failed-attempt status.
- The BabyCL SAYCam suite defines pinned-input, three-seed runs for Labeled-S 4AFC
  and fixed-feature linear decoding, without shipped reference scores.
- `probe:four_afc`, `probe:vtwt`, and `probe:baby_winoground` provide image–text
  evaluations for trained models.
- `probe:linear` fits RNG-free fixed-feature L-BFGS classifiers, while
  `probe:linear_sgd` isolates its RNG during stochastic-view SGD training.
- Labeled-S linear decoding uses a pinned canonical split, and BabyView provides
  frozen 4AFC trials and a 72-class linear split.
- Shape-bias evaluation scores 57,600 novel shape/texture triplets using pinned,
  deterministic stimuli.
- Spatial relation evaluation provides balanced 2D and 3D scores, fixed monitoring
  subsets, data preparation, and full-versus-monitoring comparisons.
- `probe:intphys2` scores 506 possible/impossible video pairs with configurable
  context lengths and L1 or L2 prediction error.
- `probe:grasp` scores GRASP Level 2 video pairs by test and concept using a
  local manifest from `egl data prepare grasp`.
- The `video` and `train` extras include PyAV (`av`) for benchmark video decoding.
- Shape-bias and spatial monitoring collections have revision-pinned hosted data
  and packaged checkpoint-evaluation configurations.
- `egl data` provides a dataset catalog, checksummed fetching, verification, and
  benchmark preparation under a shared data root.
- Hub references resolve revisioned data and model paths and retain the resolved
  commit in evaluation and initialization metadata.
- `stream:packed_frames` reads encoded frames from indexed shards, with commands
  to build, inspect, and verify a store.
- Slurm launchers prepare container environments, archive clean commits, and
  submit individual jobs or Hydra sweep arrays.
- `egl doctor` checks environment dependencies, CUDA, data roots, and cluster
  storage without modifying settings.
- The documentation site uses MkDocs under `/docs/`, with a landing page at the
  site root and `noindex` tags enabled for staging.
- Contribution guidance includes a PR template and a local PR-message checker.
- `saycam_default` reproduces the BabyCL paper's run, including its 26568-step
  budget, mixed precision, and 2-D Gaussian blur.
- `solver.schedule.total_steps` indexes the cosine schedule by gradient steps.
- `driver.amp` runs encoder forwards and losses under float16 or bfloat16
  autocast, with named slots kept in float32.
- `stream.transcript_first` surfaces an utterance before a frame with the same
  timestamp, on frame directories and packed frame stores alike.
- `paper_evals.yaml` runs the BabyCL paper's seven end-of-training evaluations
  with the paper's protocols.
- `probe:labeled_s_stratified_linear` and `probe:mini_imagenet_linear` follow the
  BabyCL paper's linear decoding protocols.
- `probe:linear` accepts `feature_norm: l2` and a scikit-learn-style penalty `c`.
- The BabyCL README reports per-seed SAYCam results next to the paper's numbers,
  with each protocol difference stated.

### Changed

- Full-pass MAE schedules use optimizer updates so streaming and IID runs receive identical learning rates.
- Serial augmentation preserves CUDA randomness by seeding only the CPU generator.

- The `generation` capability becomes `text_generation`, and generative components must produce text.
- Site builds use pinned uv in a separate Python 3.12 environment and keep Cloudflare previews unindexed.
- The test suite trims repetitive validation matrices and consolidates training, CLI, recording, and import checks.
- VoE probes report pooled-surprise AUROC alongside pairwise accuracy, with
  optional group metrics for labeled pairs.
- The documentation separates task guides from reference material and corrects
  stale descriptions of training concurrency, stopping conditions, and job snapshots.
- `driver.prefetch` overlaps batch composition with training while preserving
  supported sampling and snapshot-continuation behavior, and requires `loader.workers >= 1`.
- `loader.workers` parallelizes decoding and augmentation with per-item seeds,
  preserving inputs across worker counts.
- SimCLR image views default to separable blur, within one intensity level of the
  reference, with `blur_impl=torchvision` restoring reference behavior and `blur_impl=none`
  disabling blur; see [data-loading controls](docs/data_loading.md).
- `loader.flush_denormal` defaults to true in loader workers, accelerating blur without
  changing tested pixel outputs or the trainer's arithmetic mode.
- Benchmark baseline comparisons require matching seeds and input pins, while
  accepting identical artifacts stored at different local paths.
- `egl doctor --submit` checks submission tools and cluster storage in the CPU
  allocation used for benchmark planning.
- `scripts/slurm/train.sbatch` and `bench.sbatch` run directly on login nodes and
  reject invocation through `sbatch`.
- `scripts/slurm/bench.sbatch` writes data-loading benchmark JSON to the isolated run's `work/bench.json`.

### Removed

- `probe:imagenet_linear_sgd` is replaced by standalone `benchmark:imagenet_linear_sgd`
  and the `imagenet_linear_eval` config; see [ImageNet migration](docs/evaluation.md#imagenet-migration).
- `ImageNetLinearSGDProbe` is replaced by `ImageNetLinearBenchmark`.
- `scripts/prepare_shape_bias.py`, `scripts/prepare_spatial_relations.py`, and
  `scripts/prepare_spatial_monitoring.py` are replaced by `egl data prepare` and the
  [installed module commands](docs/cli.md#data).

### Fixed

- Full-pass MAE rejects explicit resume paths and checkpoint directories without a manifest before loading training state.
- Full-pass MAE permits moving run directories and changing logging or checkpoint frequency between resume attempts.
- MAE resume archives milestones ahead of the restored checkpoint and recreates them at their scheduled updates.
- MAE records intentional stops as paused and counts preparation and checkpoint replay toward its wall-clock budget.

- Staging sites allow crawlers to read `noindex` tags without `robots.txt` blocking access to the pages.
- VoE probes reject videos shorter than `num_frames`, preventing repeated frames
  from affecting IntPhys 2 and GRASP scores.
- Offline training raises on non-finite or empty losses, preserving batch order
  and closing loader workers without writing a final checkpoint.
- Temporal IID training validates empty recordings and rejects snapshot resumes
  when the temporal-label manifest changes or its saved identity is unavailable.
- Video benchmarks fall back to the CPU count on platforms without CPU affinity
  support, including macOS.
- Prepared-video loading checks cache coverage before training, and video benchmarks
  accept explicit session lists for partial caches.
- PyAV frame reads isolate returned images, preserving repeated requests and cached
  pixels when views mutate their inputs.
- Evaluation preserves completed benchmark results and provenance when a later
  benchmark fails, and rejects nonfinite metrics.
- Evaluation restores model modes, gradient flags, and RNG states, including MPS,
  after scoring or failure.
- Evaluation metadata replaces absolute local path settings with file or
  directory names while retaining Hub references and dataset identity.
- Automatic evaluation records serialize dataclass settings without replacing
  the objects passed to benchmark constructors.
- Spatial evaluation records capture both manifest hashes and pinned Hub commits
  before benchmark construction.
- Snapshot resume retains initialization lineage and producing-run provenance
  without replacing the live model's construction metadata.
- Training snapshots retain nonfinite-step counters and durable links to saved
  checkpoints across resumed attempts.
- Seeded training restores global RNG state after stream replay so resumed
  runs match uninterrupted runs in reproducibility tests.
- Streaming training processes the final partial advance and session flush
  without admitting the same final events again after resume.
- Utterances ending on a frame timestamp appear after that frame, retaining the
  final frame in the utterance chunk.
- The scheduled learning rate is set before each optimizer update, so warmup
  covers the first update.
- Unfinished gradient accumulation is applied at stream exhaustion or retained
  for resume.
- Frame-directory listings use stable time-and-name ordering regardless of
  filesystem enumeration order.
- TCL avoids loading raw pixels when precomputed features are supplied, and
  invalid multi-frame text slots fail during pipeline construction.
- W&B starts on first use so dry-run validation does not create a tracking run.
- Benchmark entry points compose Hydra defaults, and offline BabyCL retains
  its 5,000-step checkpoint cadence when inheriting shared settings.
- Slurm submission validates UTF-8 arguments and creates snapshots without
  executing Python on login nodes.
- The combined site build retains a root 404 page and applies the staging
  setting consistently to the landing page and documentation.
- A session's last segment gets its own training steps instead of sharing the
  next session's quantum.
- `driver.max_steps` stops at exactly that step, also inside a quantum.
- A run resumed from a snapshot taken inside a quantum finishes that quantum
  before reading more of the stream.
- A float16 run resumed with a partial gradient-accumulation window applies the
  pending gradients once, under the restored loss scale.
- `driver.max_steps` preserves pending accumulated gradients when it interrupts
  the final quantum after stream exhaustion.

### Security

- Legacy checkpoints use restricted loading by default, and full training
  snapshots require explicit trust during evaluation, initialization, and export.
- Model Hub uploads accept only validated exported weights and metadata,
  rejecting training snapshots and other payloads.
- Hub publishing creates private repositories by default and requires an
  explicit public-upload option for public repositories.
