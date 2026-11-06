# MAE and StreamMAE

Starting point for **I Have a Stream: Making Self-Supervised Learning Work on
Continuous Video**, Ivan Martinović, Lukas Knobel, Yuki M. Asano, 2026
([paper](https://arxiv.org/abs/2609.40333)). Work is tracked in
[issue #48](https://github.com/agentic-learning-ai-lab/egolearn/issues/48).

This project implements standard image
[MAE (He et al., CVPR 2022)](https://arxiv.org/abs/2111.06377) and the StreamMAE
training method. No downstream performance or paper reproduction is claimed.

Use the [matched ablation suite](ablations.md) to compare vanilla MAE, shuffled
MAE, DataDrop, regularization, and motion crops with equal optimizer budgets.

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
share the same update-indexed LR (1.6e-4, 5% warmup then constant) to isolate data ordering;
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

The full-pass configs use **stride 32**: fill 480 frames, advance 32, then train
on the entire 512-frame FIFO window. Subsequent updates each admit 32 new frames;
the last advance may be shorter so every source frame is consumed. For N frames,
batch B and stride S, both modes perform **ceil((N − B + S) / S) updates**.
IID reshuffles each epoch and drops incomplete batches. This is approximately
B/S = **16 IID epochs**, with exact update matching: for the current inventories,
114,187 updates on SAYCam and 230,115 on BabyView. The last IID epoch is 5 or 13
batches short, respectively, due to the initial buffer fill and rounding.

To change stride, set `run_control.stride`, `driver.advance.n`, and
`driver.warmup.admitted` (B − S) consistently. The runner rejects mismatches.
The shorter `mae_small`/`mae_base` configs still default to stride 1.

The full-pass loader packs worker results into chunk tensors, writes the final
batch into pinned memory, and sends uint8 crops to the GPU for the usual float32
ImageNet normalization. PIL crop/flip interpolation and per-item augmentation
seeds are unchanged. Streaming workers cache at most 1,024 decoded images each;
augmented crops are never cached. IID disables that cache and prefetches one
batch while training. Prefetch does not commit the next batch's sequence until
it is consumed, and its worker thread is joined before pausing.

Each run records its recipe, budget, installed-source digest, and numerical
dependency versions in an immutable manifest. It also writes JSONL loss/LR metrics,
a status file, and one rolling resumable snapshot every 5,000 updates.
Archived snapshots follow every IID epoch, including the short final epoch,
at the same updates in streaming. Files are named `snapshot_epoch_001.pth` through
`snapshot_epoch_016.pth`. `final.pth` is written only on completion.

Full-pass `final.pth` and `snapshot_*.pth` files contain resumable training state.
When evaluating snapshots you created or otherwise trust, pass
`trusted_snapshot=true`, or [export safetensors](../../docs/cli.md#exporting-weights) first.

The full-pass runner automatically resumes `snapshot_latest.pth` and refuses
to reuse a directory with changed scientific settings. Milestones ahead of the
restored update move into `abandoned_snapshots/<attempt>/` before training resumes.
They are recreated as the resumed run reaches each threshold. Existing milestones
at or before the restored update remain unchanged, and missing past milestones
are not reconstructed. CPU tests check exact
interrupted-versus-uninterrupted model equality in both modes; CUDA kernels
are not guaranteed bitwise deterministic.

Worker count, packed transfers, pinning, decoded-cache capacity and IID prefetch
can change between attempts; these throughput settings are recorded separately
under `attempts/`. The source/runtime fingerprint and scientific recipe still
must match. Pinning is disabled automatically when CUDA is unavailable.

`run_control.max_seconds` (default 46 hours) counts preparation and checkpoint
replay as well as training. SIGUSR1 is handled throughout those stages. A pending
pause takes effect once initial or restored state is ready, or after a completed
update during training. Preparation and replay finish before the runner can save
a resumable snapshot, so the limit is not a hard deadline.
The runner saves a snapshot and exits **75** for scheduler requeue.

`run_control.stop_after_steps=N` supports a
short preflight/resume check without changing the full LR schedule or budget,
and exits **76** so preflights cannot accidentally enter a requeue loop.
Both stops produce a `paused` run record with `pause_exit_code` and the completed
step count. Exceptions still produce failed records, and nonfinite metrics retain
their `nonfinite` status.

Set `run_control.expected_frames` from an independent archive inventory to
reject an incomplete mount. An initial snapshot permits recovery before the first update.
Streaming prefetch and gradient accumulation are disabled in this runner.
IID prefetch is supported with process workers.

`scripts/slurm/mae.sbatch` requests one GPU, 16 CPUs, 128 GB and 48 hours,
and requeues only on exit 75. Supply account/partition/output at submission,
plus `MAE_SOURCE`, `MAE_PYTHON`, `MAE_IMAGE`, `MAE_BINDPATH`, `MAE_OVERLAY_LIST`
(one archive per line), and `MAE_RUN_ROOT`. Pass the config name followed by
Hydra overrides, including the mounted `stream.root_dir`. Set `MAE_REQUEUE=0`
for preflights. Use a frozen source tree and a non-editable installation for
long runs so later development cannot alter a queued or resumed job.


## StreamMAE on SAYCam and BabyView

The `full_saycam_streammae` and `full_babyview_streammae` recipes use the same
full-pass runner and Slurm launcher as MAE:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_saycam_streammae \
    stream.root_dir="$SAYCAM_ROOT"
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_babyview_streammae \
    stream.root_dir="$BABYVIEW_ROOT"
```

They start ViT-S/16 from scratch with an unchanged MAE decoder and reconstruction
objective. The encoder has per-example stochastic depth on both residual
branches, increasing linearly across blocks to 0.25. Color jitter uses brightness,
contrast, saturation and hue strengths `(0.4, 0.4, 0.2, 0.1)` with probability 0.5.
The zero-drop-path encoder retains existing MAE parameter names and behavior.

DataDrop uses a **fixed-cardinality binary mask**: sample 512 distinct frames
uniformly from the current 2,048-frame FIFO. Dropped examples receive no training
view or model forward. This retains exactly 25% each step, rather than drawing
independent Bernoulli masks with variable cardinality. The learning rate remains
1.6e-4, scaled for the retained batch of 512. `run_control.window_size` controls
the window and `batch_size` controls retained examples. The runner validates
capacity, slot size, stride and initial fill. Update count is
`ceil((N - window_size + stride) / stride)`; the tail consumes all source frames.
With stride 32, this is 48 fewer updates than the 512-window MAE baseline because
of the longer initial fill. For stride 1, set both `run_control.stride=1` and
`driver.warmup.admitted=2047`; `driver.advance.n` follows the stride interpolation.

`admission:streammae_frames` computes RGB L1 differences against the immediately
preceding frame in that recording, before DataDrop or random augmentation.
Averaging each complete 16x16 patch produces a small motion map cached in the
FIFO metadata. A session start or frame-size change produces a zero map.
`view:streammae` samples four patch-aligned first-stage regions and chooses the
one with the highest average motion. This two-stage branch is used with
probability 0.5; otherwise ordinary full-frame random resized cropping applies.
The selected region receives the standard MAE crop/flip plus color jitter.
Equal scores select the first uniformly sampled candidate.

The prepared datasets have different geometry from WT++. These recipes keep
native RGB dimensions and select first-stage regions with height fraction
336/720 and width fraction 592/1280, rounded to complete patches. Partial patches
at the right/bottom edges are excluded from first-stage candidates. The ordinary
full-frame branch can still use those edges. At 224x224 this produces 96x112 (width x height)
regions; at 270x480 (width x height), 128x224 regions. Explicit `crop_size` uses
`[height, width]` pixels. To use the paper geometry, set `frame_size: [720, 1280]`
in both admission and the view, and set view `crop_size: [336, 592]`.

Motion uses the prepared stream's actual sampling rate. The `fps` field only
labels timestamps and never resamples files. No additional orientation transform
is applied. Changing geometry is a new scientific recipe and requires a new run.

For ablations: set encoder `drop_path_rate=0`, view `jitter_probability=0`,
`motion_candidates=1` for random two-stage crops, or `crop_probability=0` for
ordinary cropping. Disable DataDrop with `run_control.window_size=512` and
`driver.warmup.admitted=480` at stride 32. These switches are independent.

Motion maps are stored in snapshots with the FIFO. Resume replays frame refs
without decoding historical pixels and reloads only the preceding frame before
live ingestion resumes. Metadata-only replay requires memory-backed recipe slots;
`stream_head` slots are rejected for this admission component. Augmented crops
are always regenerated from seeded randomness; worker caches contain raw images.
Synthetic tests cover moving regions, uint8 subtraction, session boundaries,
worker equality, per-example stochastic depth, DataDrop membership, and exact
interrupted-versus-uninterrupted CPU training.

## WalkingTours and WT++

`full_wt_original_streammae` uses all ten videos in the original WalkingTours
release. Follow the [shared WT preparation guide](../walkingtours/README.md) to
download and resample those videos, or compare StreamMAE with visual BabyCL.

`full_wtpp_london_streammae` trains ViT-S/16 on the paper's default 12-hour London
stream using the authors' source video or prepared frames supplied by you.
WT++ extends the original WalkingTours collection, whose released video list does
not contain London. The [project page](https://martinovicivan.github.io/StreamMAE/)
currently marks the WT++ release as pending. Use the exact source when it becomes
available or when you have a local copy.

### Prepared frames

Prepare the 60 FPS video by retaining frames 0, 16, 32, and so on, at 1280×720.
The following command writes high-quality JPEGs to a new session directory. Set
`WALKINGTOURS_VIDEO` to the video path and `WALKINGTOURS_ROOT` to your frame root.
Before extraction, use `ffprobe` to check that your input video has a frame rate of 60 FPS:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$WALKINGTOURS_VIDEO"
mkdir -p "$WALKINGTOURS_ROOT"
mkdir "$WALKINGTOURS_ROOT/london"
ffmpeg -nostdin -n -i "$WALKINGTOURS_VIDEO" -an \
    -vf "select='not(mod(n,16))',scale=1280:720" -fps_mode vfr \
    -q:v 2 -start_number 1 "$WALKINGTOURS_ROOT/london/frame_%08d.jpg"
```

The adapter also accepts PNGs, requires consecutive indices starting at 1, and
rejects empty or missing sessions. A truncated extraction
can still have consecutive indices, so set `run_control.expected_frames` from an
independent source inventory. The adapter uses `stream.fps=3.75` for timestamps
without resampling the prepared frames. Keep the prepared data unchanged throughout training and
resume.

### Training

Set `WALKINGTOURS_FRAMES` to the expected extracted frame count from your independent
source inventory, then start the full-pass runner:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_wtpp_london_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" \
    run_control.expected_frames="$WALKINGTOURS_FRAMES"
```

Run a preflight with `run_control.stop_after_steps=10`, then set it to zero to
resume the same run. The existing `mae.sbatch` launcher accepts the WalkingTours
config names and retains its pause/requeue behavior. For extracted files on a
normal filesystem, provide an empty `MAE_OVERLAY_LIST` file and bind the frame root.

The recipe uses stride 1, a 2,048-frame FIFO, and 512 retained examples per update.
Both motion scoring and cropping use 720×1280 frames with a 336×592 first-stage
region, expressed as height × width. Learning rate, model, seed, and
regularization match the SAYCam/BabyView StreamMAE recipes. DataDrop retains the
existing fixed-cardinality sampling implementation. These settings follow the
[paper's training protocol](https://arxiv.org/html/2609.40333v1#S5), but do not
establish numerical equivalence to the authors' implementation.

The loader caches 256 decoded frames per worker to limit full-resolution memory
use. Training saves a rolling snapshot every 5,000 updates and archives snapshots
at 20%, 40%, 60%, 80%, and completion, with equivalent IID epoch archiving disabled.

For another ordered collection, use `full_walkingtours_streammae` and set
`stream.sessions=[first_video,second_video,...]` to your directory names in training
order. For the paper's WT++95h stride of 2, set `run_control.stride=2` and
`driver.warmup.admitted=2046` together. Use a new run directory when changing the
collection or scientific settings.
