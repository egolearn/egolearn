# Changelog

All notable changes are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [SemVer](https://semver.org) (0.x: minor = features, patch = fixes).

## [Unreleased]

### Added

- Streaming learning framework: stream adapters (SAYCam/BabyView/video/toy),
  tiered memories with revisable labels, admission event builders, slot-based
  batch recipes, SSL + multimodal losses, lockstep streaming driver and
  offline-IID driver, progress-clock LR schedule, world-state snapshots with
  deterministic resume, stage-latency profiling, wandb hook.
- Memory Storyboard and BabyCL as 1:1 method ports (`src/egolearn/methods/`)
  with reproduction configs under `projects/`.
- Evaluation probes: Labeled-S 4AFC, visual two-word test, Baby Winoground.
- Hydra train entry point (`python -m egolearn.train`); registry-based
  component instantiation under the hydra shell.
- `video`/`train` extras; CI test job installs the train extra.
- Generic Slurm launchers and data-plane/full-pipeline benchmarks under
  `scripts/`.

### Added

- Repository scaffold: src-layout package with packaged hydra configs, tiered test
  setup, CI gates (lint/types/test/lock/gitleaks), pre-commit hooks, branch
  protection script, and governance docs.
