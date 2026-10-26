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
The benchmark measures CPU loading and augmentation, not GPU training speed.

The output directory contains `requests.json`, an incrementally updated
`report.json`, and the optional bounded `jpeg/` cache. Keep these experiment
artifacts outside the repository.

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
