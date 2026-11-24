# WalkingTours

This page explains how to turn the ten original WalkingTours videos into the
frame dataset that the [BabyCL](../../projects/babycl/docs/walkingtours.md) and
[StreamMAE](../../projects/streammae/walkingtours.md) WalkingTours recipes train
on. Both recipes read the same prepared frames in the same video order.

## Requirements

- The ten MP4 files from the [official Figshare release](https://doi.org/10.21942/uva.25189275.v1),
  about 81 GiB under CC BY 4.0, downloaded into one directory
- EgoLearn installed with `uv sync --extra train`
- FFmpeg and `ffprobe` with software decoders for H.264, VP9 and AV1, first on
  your `PATH` (for AV1, use a build with `libdav1d`)

This release includes Wildlife but not the WT++ London stream, which the
[StreamMAE WalkingTours guide](../../projects/streammae/walkingtours.md) covers.

## Frame preparation

Set `WALKINGTOURS_RAW` to the download directory and `WALKINGTOURS_ROOT` to a
new output directory, then run the preparer on CPUs:

```bash
uv run python -m egolearn.data.prepare_walkingtours \
    --source-root "$WALKINGTOURS_RAW" \
    --output-root "$WALKINGTOURS_ROOT"
```

The preparer checks each video against the MD5 checksums in the packaged
[source manifest](../../src/egolearn/data/manifests/walkingtours.json). It then
writes 1280×720 JPEG frames at 3.75 frames per second, one directory per video:

```text
$WALKINGTOURS_ROOT/
  inventory.json
  venice/
    preparation.json
    frame_00000001.jpg
    ...
  amsterdam/
  ...
```

The preparer writes `inventory.json` after all ten videos finish. If a run
stops, rerunning the same command reuses the videos that finished. Keep the
prepared frames unchanged while you train or resume.

## Video order

The recipes stream the videos in this order: Venice, Amsterdam, Singapore,
Istanbul, Bangkok, Stockholm, Kuala Lumpur, Zurich, Chiang Mai and Wildlife.

The published Zurich file stops after about 26.5 minutes, although its header
claims 65 minutes. The file matches the release's MD5, so the preparer and both
recipes use the 26.5 minutes that exist.
