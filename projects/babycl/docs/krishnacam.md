# KrishnaCam training and loading benchmarks

Use these configs to train visual BabyCL models directly from KrishnaCam MP4s
and measure decoding throughput. The silent recordings provide no image-text
pairs, so this experiment uses the instance and temporal objectives only.

## Dataset and environment

Follow the [dataset setup](../../krishnacam/README.md#dataset-setup), then install
the training environment with `uv sync --extra train`. Set your dataset and
artifact directories before running the commands below:

```bash
export KRISHNACAM_ROOT=/path/to/krishnacam/kcam
export KRISHNACAM_MANIFEST=/path/to/krishnacam/manifest.json
export EGOLEARN_ARTIFACTS=/path/to/artifacts
```

The optional manifest maps relative video filenames to `fps`, `num_frames`,
`width`, and `height`. Unset `KRISHNACAM_MANIFEST` to read metadata from the
containers instead. The supplied archive contains 436 recordings at approximately
29.923 frames per second, covering 69.2 hours.

## Training settings

The config trains a ResNeXt-50 from scratch with equally weighted NT-Xent and
TCL losses. Each step draws four visual segments and samples 32 frames from each,
with two augmented views per frame. A frozen ImageNet ResNeXt-50 supplies features
for segmentation, with weights downloaded into `TORCH_HOME` on first use.

Frames retain their native timestamps and frame indices throughout segmentation
and replay. The segmenter embeds every 360th frame, roughly once per 12 seconds,
and packs segments toward a three-minute target. Recording boundaries and
similarity cuts affect the actual segment lengths.

The short FIFO holds 138 segments and spills into a 692-segment reservoir.
These approximate 10% and 50% of the nominal segment count, respectively.
Each batch draws one short-memory segment and three long-memory segments when
both tiers have enough entries. Training starts after the short tier fills and
runs eight optimizer steps per subsequent segment advance.

## GPU pilot

Use the pilot on an allocated GPU to check memory use and throughput:

```bash
uv run python scripts/bench/bench_full_pipeline.py \
    --config projects/babycl/configs/krishnacam_pilot.yaml --steps 32 \
    --override "run_dir=$EGOLEARN_ARTIFACTS/krishnacam_pilot" \
    --output-json "$EGOLEARN_ARTIFACTS/krishnacam_pilot/throughput.json"
```

The pilot uses the full model and batch, with replay capacities reduced to four
short and 16 long segments. You can compare worker counts with repeated
`--override loader.workers=4` or `8`, giving each run a separate directory.
Use multiples of eight for exact step limits, which the driver checks between
segment advances.

The report includes elapsed time with replay warmup, optimizer steps per second,
logical frame requests per second, and peak allocated GPU memory. Frame requests
include segmentation embeddings and replay samples, not codec intermediate
frames. CUDA synchronization brackets the overall measurement. Individual stage
timers measure CPU wall time and can overlap when prefetching is enabled.

## Loading comparison

Run the loading benchmark on a CPU allocation with enough cores for its largest
worker count. Choose a new output directory for each invocation:

```bash
uv run python -m egolearn.data.video_benchmark \
    --root "$KRISHNACAM_ROOT" --manifest "$KRISHNACAM_MANIFEST" \
    --output-dir "$EGOLEARN_ARTIFACTS/krishnacam_loading" \
    --workers 0 4 8 16 --batches 12 --warmup 2 --jpeg-reference
```

The benchmark samples four windows of up to three minutes per batch across up to
eight recordings spread through the dataset, fixing frame indices and augmentation
seeds across all variants. The report compares individual video reads, grouped video reads,
and JPEG loading with the same two BabyCL augmentations.

JPEG preparation exports only the requested frames at original resolution and
quality 95, with preparation time and disk usage reported separately from loading latency.
JPEG compression is lossy, so tensor equality is checked only between video
variants, using the first measured batch. The operating system cache is
uncontrolled, and worker startup is reported separately from steady batch timing.
The benchmark measures loading and CPU augmentation, not GPU training speed.

## Hardware decoding

Select NVIDIA NVDEC with `stream.decoder.device=cuda:0` on an allocated GPU.
The decoder retains intermediate frames on the device and transfers only
requested images for the existing CPU augmentations. Frame selection and
augmentation seeds are unchanged. Metadata probing does not initialize CUDA,
so you can still validate the config on a CPU allocation.

This path requires PyAV 19 or newer, a CUDA-enabled FFmpeg build, and access to
the NVIDIA video decoder libraries. An unavailable or unsupported hardware
decoder raises an error instead of silently falling back to the CPU.

Each worker has its own decoder handles and consumes GPU memory. Start with
`stream.decoder.cache_handles=2` and a small worker pool, then measure scaling
on your GPU. To benchmark hardware decoding with the same sampled requests,
add `--device cuda:0 --cache-handles 2 --skip-individual` to the loading command
and choose a new output directory. Request a GPU allocation for this comparison.

The output directory contains `requests.json`, an incrementally updated
`report.json`, and the optional bounded `jpeg/` cache. Keep these experiment
artifacts outside the repository.

## Lossless prepared videos

For repeated BabyCL runs, prepare smaller videos containing the exact RGB pixels
after the views' fixed bicubic resize and center crop. Frames remain in MP4 files
and are decoded on demand. H.264 RGB lossless encoding avoids another lossy color
conversion, and a keyframe every eight frames bounds random-access decoding work.
These lossless RGB videos require CPU decoding, so use `--device cpu` in the
loading benchmark and `stream.decoder.device=cpu` for GPU training.

This cache is specific to the configured spatial preprocessing. It supports
`babycl_ms_aug` and unflipped `center_crop` views at the prepared size, rejecting
other views when the loader is built. Keep the originals for other augmentations,
resolutions, and evaluations. Preparation records library versions and rejects a
cache built with different decoding or image-processing versions.

On a CPU allocation, choose a separate directory and prepare a subset first:

```bash
export KRISHNACAM_PREPARED=/path/to/krishnacam/prepared
uv run python -m egolearn.methods.babycl.prepare_video \
    --source-root "$KRISHNACAM_ROOT" --prepared-root "$KRISHNACAM_PREPARED" \
    --workers 16 --sessions recording_a.mp4 recording_b.mp4
```

Replace the session names with filenames from the source directory. Omit
`--sessions` to convert all MP4s directly under that directory. Each complete
output is decoded and checked against a hash of every preprocessed source frame.
Frame counts, frame rates, and source-relative frame times must also match.
Output container timestamps start at zero. Repeating preparation
skips verified files whose source and output fingerprints are unchanged.

### Cache benchmarks

Measure disk usage and GPU training throughput before preparing the full dataset.
Lossless RGB videos can exceed the originals' size despite their smaller dimensions.
Add `--prepared-root "$KRISHNACAM_PREPARED" --seek-gap 8` to the loading benchmark
to reuse its original frame requests and seeds. The selected recordings must all
have prepared counterparts.

For GPU training, use `projects/babycl/configs/krishnacam_prepared.yaml`. It keeps
the original stream root and redirects decoding to the cache, so replay snapshots
retain valid original paths. This config uses 32 loader processes; set
`loader.workers` explicitly when comparing pool sizes. To compare against the short pilot, override
`memories.segments.tiers.short.capacity=4` and
`memories.segments.tiers.long.capacity=16` in the full-pipeline benchmark. Add
`--gpu-log /path/to/gpu.csv` to collect one-second GPU samples with training steps;
exclude warmup and worker startup when assessing steady training utilization.

## Full chronological runs

Submit full runs through the [Slurm launcher](../../../scripts/slurm/README.md)
from a clean commit. After configuring and sourcing your local `env.sh`, launch
three independent seeds with an appropriate GPU partition:

```bash
scripts/slurm/submit --sweep --max-parallel 3 \
    --job-name babycl-krishnacam --partition "$GPU_PARTITION" \
    --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 -- \
    egl train projects/babycl/configs/krishnacam.yaml seed=0,1,2 \
    "stream.root_dir=$KRISHNACAM_ROOT" \
    "stream.manifest_json=$KRISHNACAM_MANIFEST"
```

The full config has no optimizer-step cap and consumes one chronological pass.
Its output directories are `$EGOLEARN_ARTIFACTS/babycl_krishnacam/seed_<seed>`.
Checkpoints and resumable snapshots are saved every 500 optimizer steps, with
additional snapshots at configured stream milestones. Choose the wall-time
request from pilot measurements and use `resume=auto` with the same absolute
`run_dir` when continuing an interrupted run.

## IID visual baselines

Use `krishnacam_iid_simclr.yaml` for instance loss alone, or `krishnacam_iid_tc.yaml`
for temporal loss alone. Both configs shuffle all native-rate frames and use the
verified lossless cache, the same ResNeXt-50, and two BabyCL augmentations per frame.
Each update contains 128 frames from a global shuffled order, with no replay memory.
The active loss has weight 1.0. Only its corresponding head is constructed.

Set `driver.max_steps` to the completed streaming run's update count for each
seed. The default is 11,900 steps, and the cosine schedule spans that budget.
This matches updates and batch size, but does not match which frames each model
sees. The run can stop before completing one shuffled epoch.

```bash
egl train projects/babycl/configs/krishnacam_iid_simclr.yaml \
    seed=0 driver.max_steps=11904
```

TC requires fixed labels covering every frame before training begins. Supply
`KRISHNACAM_TEMPORAL_MANIFEST` and set `temporal_classes` to its class count:

```bash
export KRISHNACAM_TEMPORAL_MANIFEST=/path/to/temporal_labels.json
egl train projects/babycl/configs/krishnacam_iid_tc.yaml \
    temporal_classes=2 seed=0 driver.max_steps=11904
```

The example below describes one eight-frame recording with two temporal classes.
Replace its session name, metadata, and intervals with your complete dataset.
Intervals use inclusive start and exclusive end frame indices. They must cover
each recording exactly once, without gaps or overlap. Labels are integers from
zero through `num_classes - 1`, shared across recordings.

```json
{
  "format_version": 1,
  "num_classes": 2,
  "sessions": {
    "recording.mp4": {
      "fps": 30.0,
      "num_frames": 8,
      "segments": [[0, 3, 0], [3, 8, 1]]
    }
  }
}
```

The driver checks session names, frame counts, timestamps, and label coverage
before training. TC uses the same fixed target for a frame regardless of batch
order, with all classes active from the first update. Streaming BabyCL instead
uses batch-local labels. When converting an online segmenter's evolving output,
resolve overlapping revisions explicitly and retain that rule with the manifest's
provenance. Checkpoint evaluation uses the vision encoder, without either training
head.
