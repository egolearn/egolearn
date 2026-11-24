# Changelog

User-visible changes are grouped by kind, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [SemVer](https://semver.org), with minor releases for features and
patch releases for fixes during 0.x development.

## [Unreleased]

## [0.1.0] - 2026-11-24

First public release.

### Added

- Streaming and offline training run self-supervised and image–text objectives with bounded replay memory, snapshots and deterministic resume.
- Streams read SAYCam, BabyView and WalkingTours frames, MP4 recordings through PyAV, and packed frame stores.
- BabyCL learns visual and verbal representations from SAYCam in one chronological pass, with three-seed SAYCam results next to the paper's numbers.
- Memory Storyboard combines SimCLR with supervised contrastive learning over temporal segments in replay.
- MAE and StreamMAE train ViT-S/16 and ViT-B/16 encoders by masked reconstruction, streaming or offline, with matched update budgets.
- V-JEPA 2 pretrains video encoders on SAYCam or BabyView, with matched IID and chronological streaming recipes.
- KrishnaCam configurations train visual baselines directly from MP4 recordings.
- The `toy_streaming` and `toy_mae` configurations train on synthetic data on a CPU, without datasets or pretrained weights.
- `egl eval` scores a model on Labeled-S, BabyView, Konkle, ImageNet, VTWT, Baby Winoground, shape-bias, spatial-relation, IntPhys 2 and GRASP benchmarks.
- EgoSchema and Ego4D NLQ score generative models behind an OpenAI-compatible server, such as vLLM.
- `evaluate`, `evaluate_checkpoint` and `load_model` run evaluations from Python and return results with checkpoint provenance.
- `egl bench` runs pinned benchmark suites locally or as Slurm arrays and tables complete-seed results against a baseline.
- Training and evaluation write run records with input hashes, checkpoint lineage and metrics.
- `egl data` lists the dataset catalog, fetches checksummed Hugging Face collections and prepares benchmark inputs.
- `egl export` writes safetensors weights with JSON metadata, and `egl publish` uploads models and results to Hugging Face.
- The model catalog lists BabyCL and V-JEPA 2 checkpoints under `egolearn/`, with SAYCam and BabyView models gated on Databrary access.
- `egl submit` sends commands and Hydra sweeps to Slurm, and `egl doctor` checks dependencies, CUDA and data roots.
- `CITATION.cff` gives the software citation, and distributions include the NVIDIA Apex, V-JEPA 2 and Ego4D license notices.

### Security

- Legacy checkpoints use restricted loading by default, and full training
  snapshots require explicit trust during evaluation, initialization, and export.
- Model Hub uploads accept only validated exported weights and metadata,
  rejecting training snapshots and other payloads.
- Hub publishing creates private repositories by default and requires an
  explicit public-upload option for public repositories.

[Unreleased]: https://github.com/egolearn/egolearn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/egolearn/egolearn/releases/tag/v0.1.0
