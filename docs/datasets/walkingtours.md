# WalkingTours

Prepare the ten original WalkingTours videos as a shared frame dataset for
[BabyCL](../../projects/babycl/docs/walkingtours.md) and
[StreamMAE](../../projects/streammae/walkingtours.md). Both recipes use the same
prepared frames and video order, with different models and training budgets.

## Source videos

Download the MP4 files from the [official Figshare release](https://doi.org/10.21942/uva.25189275.v1)
into `WALKINGTOURS_RAW`. The release contains about 81 GiB of video under CC BY 4.0.
The packaged [source manifest](../../src/egolearn/data/manifests/walkingtours.json)
records filenames, download URLs, sizes, and published MD5 checksums.

This release includes Wildlife and does not contain the WT++ London stream.
WT++ is a separate collection. For a supplied WT++ source, follow the
[StreamMAE WT++ preparation guide](../../projects/streammae/wtpp.md).

## Preparation requirements

Install EgoLearn with the `train` extra and run preparation on CPUs. From a
repository checkout, use `uv sync --extra train` and prefix the Python command
below with `uv run`. With an activated installation, you can run the command
outside the checkout using the bundled source manifest.

Install FFmpeg and `ffprobe` before extracting frames. The videos include H.264,
AV1, and VP9, so FFmpeg needs software decoders for all three. For AV1, use a
build with `libdav1d`. Put the chosen `ffmpeg` and `ffprobe` binaries first on
`PATH`, and set `WALKINGTOURS_ROOT` to a new directory for the prepared frames.

## Frame preparation

Run the preparer after downloading all ten source videos:

```bash
python -m egolearn.data.prepare_walkingtours \
    --source-root "$WALKINGTOURS_RAW" \
    --output-root "$WALKINGTOURS_ROOT"
```

The preparer verifies each source checksum, resamples to 3.75 fps, and writes
1280×720 JPEGs. Since the videos have different frame rates, it uses timestamps
to select frames at a consistent rate. The stream adapter's `fps` setting only
supplies timestamps and does not resample files during training.

Each completed session contains consecutive filenames beginning with
`frame_00000001.jpg` and a `preparation.json` record. The preparer checks frame
counts against video duration and writes `inventory.json` after all ten sessions
finish. Rerunning the command reuses completed sessions whose source identity,
preparation settings, and frame counts match.

```text
$WALKINGTOURS_ROOT/
  inventory.json
  venice/
    preparation.json
    frame_00000001.jpg
    frame_00000002.jpg
    ...
  amsterdam/
    ...
  ...
```

Pass `--source-manifest /path/to/manifest.json` to use a custom manifest with the
same format as the packaged one. Keep the prepared files and inventory unchanged
throughout training and resume.

## Video order and source issue

The shared order is Venice, Amsterdam, Singapore, Istanbul, Bangkok, Stockholm,
Kuala Lumpur, Zurich, Chiang Mai, and Wildlife. The manifest, output inventory,
and packaged stream configuration use this order.

The published Zurich file ends after about 26.5 minutes despite a 65-minute
container duration. Its size and MD5 match the release, and the final complete
video packet ends at 1589.283 seconds. The manifest records this usable duration
for that exact file, and both recipes use its available frames. Preparation
checks the shorter duration and retains both durations in its output metadata.

## Training guides

Use the [BabyCL experiment guide](../../projects/babycl/docs/walkingtours.md) or
the [StreamMAE experiment guide](../../projects/streammae/walkingtours.md) after
preparation finishes. Their architectures, objectives, and update budgets differ,
so sharing the input dataset does not make their training compute equal.
