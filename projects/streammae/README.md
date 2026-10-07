# MAE and StreamMAE

Starting point for **I Have a Stream: Making Self-Supervised Learning Work on
Continuous Video**, Ivan Martinović, Lukas Knobel, Yuki M. Asano, 2026
([paper](https://arxiv.org/abs/2609.40333)). Work is tracked in
[issue #48](https://github.com/agentic-learning-ai-lab/egolearn/issues/48).

This project currently implements standard image
[MAE (He et al., CVPR 2022)](https://arxiv.org/abs/2111.06377).
StreamMAE regularization, DataDrop, and motion-biased cropping remain follow-up
work. No downstream performance or paper reproduction is claimed.

## Run

Install with `uv sync --extra train`. The synthetic CPU example validates the
whole training path without downloading weights or accessing data:

```bash
uv run python -m egolearn.train --config-name toy_mae dry_run=true
uv run python -m egolearn.train --config-name toy_mae
```

The project configs train ViT-S/16 or ViT-B/16 from scratch on pre-extracted
SAYCam frames. Supply the dataset root through `SAYCAM_ROOT` or an override:

```bash
uv run python -m egolearn.train --config-dir projects/streammae/configs \
    --config-name mae_small stream.root_dir="$SAYCAM_ROOT"
uv run python -m egolearn.train --config-dir projects/streammae/configs \
    --config-name mae_base stream.root_dir="$SAYCAM_ROOT"
```

`dry_run=true` builds the model and validates its wiring; the SAYCam adapter
still enumerates the frame directory, so it needs an existing directory.
The synthetic config needs no data path. Dataset orientation and frame rate
are preprocessing choices: `stream.fps` describes the prepared frames and
does not resample them. Use `views.reconstruction.vertical_flip=true` only
when the prepared images require it (add the key with `+` on the Hydra CLI).

Both project configs consume every prepared frame once into a 512-frame FIFO
window and train on the entire window after each new frame. Overlapping windows
reuse frames locally; there is no long-term replay. The initial fill consumes
511 frames, so the first update sees frames 1–512. The window continues across
session boundaries. These SAYCam recipes are a starting point, not WT++
reproduction configs. No augmentation depends on the previous frame yet.

To change batch size, update the window capacity, slot `n`, and warmup
`admitted` (batch size minus one) together. Learning rates are explicit:
rescale them by the new effective batch size divided by 512. `floor: 1.0`
preserves linear warmup followed by a constant learning rate; the solver's
`constant: true` switch would also disable warmup.

## Components

Set `_imports_: [egolearn.methods.mae]` in any streaming or offline config.

| Target | Role |
| --- | --- |
| `encoder:mae_vit` | RGB patch projection, fixed 2D sine/cosine positions, CLS token, pre-norm ViT; only visible patches enter attention during training |
| `predictor:mae` | Projects visible tokens, restores their spatial positions among learned mask tokens, and reconstructs the full patch grid |
| `loss:mae` | Independent uniform masking per image; MSE only on masked patches; optional per-patch target normalization |
| `view:mae` | One bicubic random resized crop, horizontal flip, ImageNet normalization |

The default mask ratio is 0.75. Normalized targets use sample variance plus
`1e-6`; `norm_pix_loss=false` reconstructs the supplied ImageNet-normalized
pixels directly. The encoder and decoder must agree on image and patch sizes.
Mask seeds depend on run seed, loss name, training step, and view index, so
checkpoint resume regenerates masks without consuming global RNG state.

`model.encode("vision", images)` encodes the full image for evaluation, returning
CLS in `pooled`, patch-only `tokens`, and a `spatial` feature grid. Optional
`pooling: mean` averages the normalized patch tokens for frozen probes. These
features and the existing frozen probes do not implement the paper's full
fine-tuning or attentive-probing protocols. Checkpoints record the method import
and model configuration and load through `egolearn.evaluation.load_model`.

The implementation uses PyTorch transformer layers and adds no dependencies.
It is independently written from the algorithm; no CC-BY-NC upstream MAE code
is vendored. External MAE checkpoint key conversion is not implemented.

## Full streaming pass and matched IID

The `full_saycam_streaming`, `full_saycam_iid`, `full_babyview_streaming`, and
`full_babyview_iid` configs use ViT-S/16, batch 512, BF16 forward passes,
float32 residual streams/loss/parameters/Adam state, and eight decode workers. They deliberately
share the same LR (1.6e-4, 5% warmup then constant) to isolate data ordering;
they do not reproduce the paper's separate IID LR sweep/cosine recipe.
SAYCam selects child S. BabyView uses all children under
`root/child/recording/frame_00000001.jpg`, ordered by child, date and numeric
part (recording ID breaks ties). These prepared archives already have their
orientation/crop applied; there is no additional vertical flip.

Use the full-pass runner, which derives the update budget from frame counts:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_saycam_streaming \
    stream.root_dir="$SAYCAM_ROOT"
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_babyview_iid \
    stream.root_dir="$BABYVIEW_ROOT"
```

For N frames and batch B, each mode performs **N − B + 1 updates**. Streaming
consumes each source frame once, with one update per frame after the FIFO fills;
IID repeatedly shuffles the entire dataset until that same update count. IID
drops each incomplete final batch, as the standard offline driver does.
Consequently a full pass here is much more compute than one IID epoch.

Each run records its recipe, budget, installed-source digest, and numerical
dependency versions in an immutable manifest, plus JSONL loss/LR metrics,
a status file, one rolling resumable snapshot every 5,000 updates, and archived
snapshots at 5/25/50/100% of the update budget, matched across modes. `final.pth` is written only on completion.
The full-pass runner automatically resumes `snapshot_latest.pth` and refuses
to reuse a directory with changed scientific settings. CPU tests check exact
interrupted-versus-uninterrupted model equality in both modes; CUDA kernels
are not guaranteed bitwise deterministic.

`run_control.max_seconds` (default 46 hours) or SIGUSR1 requests a pause after
a completed update. The runner saves a snapshot and exits **75** for scheduler requeue. `run_control.stop_after_steps=N` supports a
short preflight/resume check without changing the full LR schedule or budget,
and exits **76** so preflights cannot accidentally enter a requeue loop.
Set `run_control.expected_frames` from an independent archive inventory to
reject an incomplete mount. An initial snapshot protects startup failures.
Prefetch and gradient accumulation are disabled in this runner.

`scripts/slurm/mae.sbatch` requests one GPU, 16 CPUs, 128 GB and 48 hours,
and requeues only on exit 75. Supply account/partition/output at submission,
plus `MAE_SOURCE`, `MAE_PYTHON`, `MAE_IMAGE`, `MAE_BINDPATH`, `MAE_OVERLAY_LIST`
(one archive per line), and `MAE_RUN_ROOT`. Pass the config name followed by
Hydra overrides, including the mounted `stream.root_dir`. Set `MAE_REQUEUE=0`
for preflights. Use a frozen source tree and a non-editable installation for
long runs so later development cannot alter a queued or resumed job.
