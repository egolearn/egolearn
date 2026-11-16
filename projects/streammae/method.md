# MAE and StreamMAE method

These recipes learn image representations from a stream of prepared frames using
masked reconstruction. This page describes the objective, sampling, and augmentation
choices that determine what each run measures. Start with the [overview](README.md)
for installation and commands, or the [matched ablations](ablations.md) for controlled comparisons.

The implementations follow [MAE](https://arxiv.org/abs/2111.06377) and
[StreamMAE](https://arxiv.org/abs/2609.40333). The recipes do not establish downstream
performance or numerical equivalence to either paper's published results.

## MAE reconstruction

The encoder receives only visible image patches. A decoder restores the masked
positions with learned mask tokens and predicts the original patch pixels.
The loss is mean squared error over masked patches, with a default mask ratio
of 0.75 and independent uniform masks for each image.

With `norm_pix_loss=true`, targets are normalized per patch using sample variance
plus `1e-6`. With `false`, the target is the supplied ImageNet-normalized image.
The encoder and decoder must agree on image and patch sizes. Mask seeds depend
on the run seed, loss name, training step, and view index, allowing masks to be
regenerated after resume.

The standard view applies a bicubic random resized crop, horizontal flip, and
ImageNet normalization. The SAYCam and BabyView recipes expect their dataset
orientation and crop to have been applied during preparation.

## Frame selection

`mae_small` and `mae_base` admit every prepared frame once into a 512-frame FIFO
window. They train on the entire window after each new frame, starting after
frames 1–512 have arrived. Windows overlap and continue across session boundaries,
with no long-term replay or motion-dependent augmentation in these two recipes.

The `full_*_streaming` MAE recipes instead advance 32 frames per update. They
fill 480 frames before the first advance, then train on the 512-frame window.
The final advance can be shorter so the stream consumes every selected frame.
The corresponding `full_*_iid` recipes reshuffle the selected dataset each epoch
and drop incomplete batches while matching the streaming optimizer budget.

StreamMAE uses DataDrop to retain 512 distinct frames sampled uniformly from a
2,048-frame FIFO. Dropped frames receive no training view or model forward.
This fixes retention at 25% per update, rather than using independent Bernoulli
masks whose retained count varies. `run_control.window_size` sets FIFO capacity,
while `run_control.batch_size` sets the retained batch.

## Update budgets

For N source frames, reference window R, and stride S, the full-pass runner uses
`ceil((N - R + S) / S)` optimizer updates. The selected inventory must have at
least R frames. In `full_*` recipes, R is the FIFO size; in `matched_*` recipes,
R defaults to 2,048 for every ablation, independently of its FIFO size.

For a hypothetical inventory of 1,000,000 frames at stride 32:

| Recipe | Reference window | Optimizer updates |
| --- | ---: | ---: |
| Full-pass MAE | 512 | 31,235 |
| Full-pass StreamMAE | 2,048 | 31,187 |
| Matched ablations, including vanilla MAE and IID | 2,048 | 31,187 |

Full-pass StreamMAE performs 48 fewer updates than full-pass MAE at stride 32
because its initial window is larger. Use the [matched suite](ablations.md#budget)
when comparing these methods with equal optimizer budgets. Read your actual
update count from the startup log and `manifest.json`, rather than transferring
a count from another dataset inventory.

For batch 512 and stride 32, the IID budget is approximately 16 epochs.
The final epoch can be shorter because the budget accounts for initial buffer
filling and rounding. IID consumption includes repeated draws, while streaming
consumption counts chronological admissions.

## Optimization

Full-pass SAYCam and BabyView recipes use ViT-S/16, batch 512, BF16 forward
passes, and float32 residual streams, losses, parameters, and Adam state.
They share an update-indexed learning rate of `1.6e-4`, with 5% warmup followed
by a constant rate. This isolates data ordering without reproducing the
StreamMAE paper's separate IID learning-rate sweep and cosine schedule.

The basic `mae_small` and `mae_base` recipes use float32 and learning rates of
`1.6e-4` and `8e-5`, respectively. To change their batch size, update FIFO
capacity, slot `n`, and warmup `admitted` together. Warmup admissions equal batch
size minus one. Scale the learning rate by the new batch size divided by 512.
`floor: 1.0` preserves warmup followed by a constant rate, while `constant: true`
also disables warmup.

For full-pass MAE, change `run_control.stride`, `driver.advance.n`, and
`driver.warmup.admitted` together, with warmup equal to batch size minus stride.
For StreamMAE, `driver.advance.n` follows the stride automatically. For stride 1,
set `run_control.stride=1` and `driver.warmup.admitted=2047` with the default window.
The [matched suite](ablations.md#budget) uses its reference window for warmup.

## StreamMAE augmentation

StreamMAE retains the MAE decoder and objective. Its encoder uses stochastic
depth independently for each example on both residual branches, increasing
linearly across blocks to 0.25. Color jitter applies with probability 0.5,
using brightness, contrast, saturation, and hue strengths `(0.4, 0.4, 0.2, 0.1)`.

Motion scores use RGB L1 differences from the immediately preceding frame in the
same recording, before DataDrop or random augmentation. Each complete 16×16 patch
contributes one mean score. Session starts and frame-size changes produce zero
scores. The actual prepared sampling rate therefore affects the motion signal.

With probability 0.5, the view samples four patch-aligned regions and chooses
the one with the highest mean motion. Equal scores select the first uniformly
sampled candidate. The selected region receives the ordinary MAE crop and flip,
plus color jitter. Otherwise, cropping starts from the full frame.

### Crop geometry

SAYCam and BabyView StreamMAE recipes preserve native prepared image dimensions.
Their first-stage regions use height fraction 336/720 and width fraction 592/1280,
rounded to complete patches. Partial patches at the right and bottom edges are
excluded from first-stage candidates but remain available to full-frame cropping.

For a 224×224 image, the first-stage region is 96×112 pixels, expressed as
width × height. For a 270×480 image, it is 128×224. Explicit `crop_size` settings
use `[height, width]` pixels. To use the WT++ geometry, set
`frame_size: [720, 1280]` in both admission and the view, and set
`crop_size: [336, 592]` in the view. Changing geometry requires a new run directory.

### Individual settings

Use the [matched ablation recipes](ablations.md#recipes) for comparisons with
equal budgets. For custom full-pass experiments, the following controls are independent:

| Setting | Effect |
| --- | --- |
| Encoder `drop_path_rate=0` | Disable stochastic depth |
| View `jitter_probability=0` | Disable color jitter |
| View `motion_candidates=1` | Choose a random first-stage region |
| View `crop_probability=0` | Use ordinary full-frame cropping |
| `run_control.window_size=512` with `driver.warmup.admitted=480` at stride 32 | Retain the entire window, disabling DataDrop |

## Components and feature outputs

Import `egolearn.methods.mae` through `_imports_` when composing a custom MAE config.
Use `egolearn.methods.streammae` for the StreamMAE additions.

| Target | Role |
| --- | --- |
| `encoder:mae_vit` | RGB patch projection, fixed 2D sine/cosine positions, CLS token, and pre-norm transformer |
| `predictor:mae` | Visible-token projection, learned mask tokens, and patch reconstruction |
| `loss:mae` | Per-image masking and masked-patch reconstruction loss |
| `view:mae` | Standard image crop, flip, and normalization |
| `admission:streammae_frames` | Frame admission with motion context |
| `view:streammae` | Motion-biased two-stage cropping and appearance augmentation |

StreamMAE requires memory-backed recipe slots to preserve motion context across
resume; `stream_head` slots are unsupported. Admission, view, and encoder patch
sizes must agree, and admission and view must use the same frame geometry.

`model.encode("vision", images)` encodes the complete image for evaluation.
It returns CLS features in `pooled`, patch-only `tokens`, and a `spatial` grid.
Encoder `pooling: mean` instead averages normalized patch tokens for frozen probes.
See the [evaluation workflow](README.md#checkpoint-export-and-evaluation) for
frozen probes and full-network fine-tuning.

This implementation is independently written from the algorithms and uses PyTorch
transformer layers. No CC-BY-NC upstream MAE code is vendored, and external MAE
checkpoint key conversion is unsupported.
