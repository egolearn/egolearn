# WT++ preparation and training

Use these recipes with a WT++ source supplied by you or another explicitly
ordered video collection. For the ten released original WalkingTours videos,
follow the [original WalkingTours guide](../walkingtours/README.md) instead.
The original release does not contain the WT++ London stream.

`full_wtpp_london_streammae` uses the StreamMAE paper's 12-hour London source.
The [project page](https://martinovicivan.github.io/StreamMAE/) lists WT++ as
pending release, so you need your own copy of that source video or its prepared
frames. Follow the [installation and GPU prerequisites](README.md#prerequisites)
before training. Frame extraction requires FFmpeg and `ffprobe`.

## Frame preparation

Set `WALKINGTOURS_VIDEO` to the source video and `WALKINGTOURS_ROOT` to a new frame
directory. For the paper's 60 fps source, retain frames 0, 16, 32, and so on,
at 1280×720. Check that `ffprobe` reports 60 fps before running extraction:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$WALKINGTOURS_VIDEO"
mkdir -p "$WALKINGTOURS_ROOT"
mkdir "$WALKINGTOURS_ROOT/london"
ffmpeg -nostdin -n -i "$WALKINGTOURS_VIDEO" -an \
    -vf "select='not(mod(n,16))',scale=1280:720" -fps_mode vfr \
    -q:v 2 -start_number 1 "$WALKINGTOURS_ROOT/london/frame_%08d.jpg"
```

The adapter accepts JPEGs or PNGs with consecutive indices starting at
`frame_00000001`. It rejects empty or missing sessions. A truncated extraction
can still have consecutive indices, so obtain an expected frame count from an
independent source inventory. The adapter uses `stream.fps=3.75` for timestamps
without resampling files. Keep the prepared frames unchanged throughout training
and resume.

## Training

Set `WALKINGTOURS_FRAMES` to that expected count, then start the full-pass runner:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_wtpp_london_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" \
    run_control.expected_frames="$WALKINGTOURS_FRAMES"
```

Append `run_control.stop_after_steps=10` for a preflight, then set it to zero to
resume the same run. The [Slurm launcher](training.md#slurm-launcher) accepts these
recipe names and requeues controlled pauses. For frames on an ordinary filesystem,
use an empty `MAE_OVERLAY_LIST` file and include the frame root in `MAE_BINDPATH`.

## Protocol and snapshots

The recipe uses stride 1, a 2,048-frame FIFO, and 512 retained examples per update.
Both motion scoring and cropping use 720×1280 frames with a 336×592 first-stage
region, expressed as height × width. Learning rate, model, seed, and regularization
match the SAYCam/BabyView StreamMAE recipes. DataDrop samples a fixed number of
frames at each update, as described in the [method guide](method.md#frame-selection).

These settings follow the [paper's training protocol](https://arxiv.org/html/2609.40333v1#S5).
They do not establish numerical equivalence to the authors' implementation.
The loader caches 256 decoded frames per worker to limit full-resolution memory use.

Training saves a rolling snapshot every 5,000 updates and archives snapshots at
20%, 40%, 60%, 80%, and completion. Equivalent IID epoch archiving is disabled.
Use [checkpoint export and evaluation](README.md#checkpoint-export-and-evaluation)
to evaluate a completed run or saved milestone.

## Other ordered collections

Use `full_walkingtours_streammae` and set
`stream.sessions=[first_video,second_video,...]` to your directory names in training
order. For the paper's WT++95h stride of 2, set `run_control.stride=2` and
`driver.warmup.admitted=2046` together. Choose a new run directory when changing
the collection or scientific settings.
