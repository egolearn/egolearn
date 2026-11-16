# Original WalkingTours experiments

These recipes train StreamMAE and visual BabyCL on all ten original WalkingTours
videos, using the same prepared frames and video order.

## Prerequisites

Run the commands from the repository root after installing the training dependencies
with `uv sync --extra train`. Frame preparation runs on CPU. Both training recipes
default to CUDA, and StreamMAE uses BF16, requiring a GPU with native BF16 support
(compute capability 8.0 or later).

Download the MP4 files from the [official Figshare release](https://doi.org/10.21942/uva.25189275.v1)
into `WALKINGTOURS_RAW`. The release contains about 81 GiB of video under CC BY 4.0.
[source_manifest.json](source_manifest.json) records the filenames, download URLs,
sizes, and published MD5 checksums used by the preparer.

## Frame preparation

Prepare the videos once for both methods. Set `WALKINGTOURS_ROOT` to a new
directory for the extracted frames and install FFmpeg before running the command.
The release includes H.264, AV1, and VP9 video, so FFmpeg needs software decoders
for all three. For AV1, use a build with `libdav1d`. If you install a separate
FFmpeg build, put its `ffmpeg` and `ffprobe` binaries first on `PATH` during preparation.

```bash
uv run python -m egolearn.data.prepare_walkingtours \
    --source-root "$WALKINGTOURS_RAW" \
    --output-root "$WALKINGTOURS_ROOT" \
    --source-manifest projects/walkingtours/source_manifest.json
```

The preparer verifies each source checksum, resamples to 3.75 FPS, and writes
1280×720 JPEGs. Since the source videos have different frame rates, preparation
uses their timestamps to select frames at a consistent rate. The stream adapter's `fps` setting
only supplies timestamps and does not resample files during training.

Each completed session has consecutive filenames beginning with
`frame_00000001.jpg`. The preparer checks counts against video duration and writes
`inventory.json` after all ten sessions finish. Rerunning the command reuses
completed sessions whose preparation settings and frame counts match.

The published Zurich file ends after about 26.5 minutes despite a 65-minute
container duration. Its size and MD5 match the release, and the final complete
video packet ends at 1589.283 seconds. The manifest records this usable duration
for that exact file, and both recipes use its available frames. Preparation
checks the shorter duration and retains both durations in its output metadata.

The shared order is Venice, Amsterdam, Singapore, Istanbul, Bangkok, Stockholm,
Kuala Lumpur, Zurich, Chiang Mai, and Wildlife. This includes Wildlife from the
original release, while WT++ London is a separate collection excluded here.

## Training

Read the expected frame count from the completed inventory before starting
StreamMAE. You can run these commands from the repository root on a workstation
or GPU compute allocation. On clusters, do not run Python on login nodes.

```bash
WALKINGTOURS_FRAMES=$(uv run python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_frames"])' \
    "$WALKINGTOURS_ROOT/inventory.json")
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_wt_original_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" \
    run_control.expected_frames="$WALKINGTOURS_FRAMES"
uv run python -m egolearn.train \
    --config-dir projects/babycl/configs --config-name walkingtours \
    stream.root_dir="$WALKINGTOURS_ROOT"
```

Both recipes use seed 1337 and consume one complete stream of all ten prepared videos. Their architectures,
objectives, and update budgets differ, so these runs do not match training compute.

| Setting | StreamMAE | Visual BabyCL |
|---|---|---|
| Trainable encoder | ViT-S/16 from scratch | ResNeXt-50 from scratch |
| Training signal | Masked reconstruction | Instance contrast and temporal classification |
| Batch | 512 retained frames from a 2,048-frame FIFO | Four segments × 32 frames × two views |
| Stream advance | One frame per update | Eight updates per admitted segment |
| Replay | FIFO with fixed-cardinality DataDrop | 28 short FIFO and 140 long reservoir segments |
| Segmentation | None | Frozen ImageNet ResNeXt-50, approximately three-minute segments |

BabyCL uses the visual objectives because WT supplies no aligned transcripts.
Its buffer capacities approximate 10% and 50% of a nominal 14-hour stream divided
into three-minute segments, although actual segment lengths vary with visual content.
The segmenter needs the torchvision ImageNet weights on first use or in its cache.

StreamMAE uses the WT++ crop geometry and stride with the original WT source videos.
It saves resumable state every 5,000 updates and at five progress milestones.
BabyCL saves weights and resumable state every 500 updates, plus progress snapshots.
Use distinct run directories when changing the data or scientific settings.

## Cluster runs

Commit the intended code and use [the Slurm launcher](../../docs/cluster.md) to freeze
the source and environment for each run. Run preparation on CPUs before submitting
the two GPU jobs, or use an `afterok` dependency on the preparation job.

For StreamMAE preflights, `run_control.stop_after_steps=16` exits with code 76 after
saving state. Set it to zero to resume the same run with its full training budget.
For StreamMAE allocations shorter than the full run, set `run_control.max_seconds` below
the wall limit and export `EGOLEARN_REQUEUE_ON_PAUSE=1` when submitting.
The launcher requeues exit code 75 after the runner saves its state.
BabyCL uses the standard runner and does not implement this pause/requeue control.
