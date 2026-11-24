# StreamMAE on WalkingTours

This page explains how to train StreamMAE on city-walk videos: the ten original
WalkingTours videos, the WT++ London stream from the StreamMAE paper, or your own
videos in a fixed order. Read the [MAE and StreamMAE overview](README.md) first
for installation, the GPU requirement and resuming a run.

## Training protocol

The three WalkingTours recipes follow the [paper's WT++ protocol](https://arxiv.org/html/2609.40333v1#S5).
They train a ViT-S/16 from scratch with seed 1337, advancing one frame per update
and training on 512 frames sampled from a 2,048-frame window. Frames are prepared
at 3.75 fps and 1280 × 720 pixels, and the motion-biased crop starts from a
592 × 336 region. The learning rate, model and regularization match the SAYCam
and BabyView StreamMAE recipes.

Training saves `snapshot_latest.pth` every 5,000 updates and keeps
`snapshot_020pct.pth` through `snapshot_100pct.pth` at each fifth of the run.
Export and evaluate them as described in the overview under
[export and evaluation](README.md#export-and-evaluation).

## Original WalkingTours

Download and prepare the ten videos with the
[WalkingTours dataset guide](../../docs/datasets/walkingtours.md), and set
`WALKINGTOURS_ROOT` to its frame directory. Read the expected frame count from
the inventory that preparation writes, then start training:

```bash
WALKINGTOURS_FRAMES=$(uv run python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_frames"])' \
    "$WALKINGTOURS_ROOT/inventory.json")
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_wt_original_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" \
    run_control.expected_frames="$WALKINGTOURS_FRAMES"
```

The run writes to `runs/streammae_wt_original/seed_1337`, or to the same path
under `$EGOLEARN_ARTIFACTS` when that variable is set. The original release has
different videos from WT++, so this recipe applies the WT++ protocol to them.
[BabyCL on WalkingTours](../babycl/docs/walkingtours.md) trains on the same frames
in the same order, with a different model, objective and update budget.

## WT++ London stream

The `full_wtpp_london_streammae` recipe trains on the 12-hour London video from
the StreamMAE paper. The [StreamMAE project page](https://martinovicivan.github.io/StreamMAE/)
does not provide WT++ for download, so you need your own copy of the source video.
Frame extraction needs FFmpeg and `ffprobe`.

Set `WALKINGTOURS_VIDEO` to the source video and `WALKINGTOURS_ROOT` to a new
directory. Check that the video runs at 60 fps, then keep every sixteenth frame
at 1280 × 720 to get 3.75 fps:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$WALKINGTOURS_VIDEO"
mkdir -p "$WALKINGTOURS_ROOT/london"
ffmpeg -nostdin -n -i "$WALKINGTOURS_VIDEO" -an \
    -vf "select='not(mod(n,16))',scale=1280:720" -fps_mode vfr \
    -q:v 2 -start_number 1 "$WALKINGTOURS_ROOT/london/frame_%08d.jpg"
```

A truncated extraction still produces consecutive filenames, so take the
expected frame count from an independent source, such as the video's frame count
divided by 16 and rounded up.
Set `WALKINGTOURS_FRAMES` to that count and train:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_wtpp_london_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" \
    run_control.expected_frames="$WALKINGTOURS_FRAMES"
```

## Other ordered collections

To train on other videos, extract each one into its own directory under
`WALKINGTOURS_ROOT` with the frame naming above. Use the
`full_walkingtours_streammae` recipe and list the directories in training order:

```bash
uv run python -m egolearn.methods.mae.train \
    --config-dir projects/streammae/configs --config-name full_walkingtours_streammae \
    stream.root_dir="$WALKINGTOURS_ROOT" 'stream.sessions=[first_video,second_video]'
```

For the paper's 95-hour WT++ setting with stride 2, also pass
`run_control.stride=2 driver.warmup.admitted=2046`. Use a new run directory
whenever you change the videos or these settings.
