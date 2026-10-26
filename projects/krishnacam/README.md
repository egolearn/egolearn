# KrishnaCam video experiments

Use these pilot configs to compare chronological replay with offline shuffled
training directly from KrishnaCam MP4 recordings. Both train a visual SimCLR
model with the same ResNet-18 encoder, augmentations, and contrastive loss.

## Dataset setup

Download the video archive linked from the
[KrishnaCam project page](https://krsingh.cs.ucdavis.edu/krishna_files/papers/krishnacam/krishnacam.html)
and extract it outside the repository, preserving the `kcam/` directory with
its timestamped MP4 filenames. Keep the videos in their original encoding and
allow about 172 GB for both the archive and extracted files.

Install the training dependencies and point the config at your extracted videos:

```bash
uv sync --extra train
export KRISHNACAM_ROOT=/path/to/krishnacam/kcam
```

The source sorts filenames to obtain recording order, which you can override
with `stream.sessions` when you need a subset or a different order. Frame rates
and frame counts come from each container's metadata. Subsampling by six
therefore approximates 5 fps without changing timestamps or generating JPEGs.

KrishnaCam has no audio, so these configs use only a visual objective.
The linked project page does not specify a checkpoint redistribution license;
confirm the applicable release terms before publishing trained weights.

## Pilot runs

Check configuration and video metadata before starting training:

```bash
uv run python -m egolearn.train --config-dir projects/krishnacam/configs \
    --config-name streaming dry_run=true
uv run python -m egolearn.train --config-dir projects/krishnacam/configs \
    --config-name offline dry_run=true
```

Both commands require the extracted dataset because the build reads container
headers to count observations, without decoding video pixels during a dry run.

For a short run on an allocated GPU, use:

```bash
uv run python -m egolearn.train --config-dir projects/krishnacam/configs \
    --config-name streaming driver.max_steps=20
uv run python -m egolearn.train --config-dir projects/krishnacam/configs \
    --config-name offline driver.max_steps=20
```

For a CPU smoke test, append `device=cpu loader.workers=0 driver.prefetch=false`.
Keep editable source unchanged until the run ends. Submit longer cluster runs
through the repository's [Slurm launcher](../../scripts/slurm/README.md), which
archives a clean commit.

The defaults cap each pilot at 1,000 optimizer steps. Streaming admits every
sixth frame, warms up with 256 admitted frames, and uses equal draws from a
1,024-frame FIFO and an 8,192-frame reservoir. It performs four optimizer steps
per 384 source observations. Offline training shuffles every sixth frame for
one epoch. Both use batches of 64 and initialize the encoder from scratch.

These initial settings provide a starting point for experiments, without
downstream evaluation probes or a claim of reproducing BabyCL results.
Offline training schedules its learning rate over the planned optimizer
steps, while streaming schedules it over the full source timeline. Preserve
the step limit and session selection when comparing loading settings.
Streaming checks the step limit between four-step advances, so choose a
multiple of four when you need an exact stopping point.

Streaming subsampling continues across recording boundaries, while offline
subsampling restarts at the beginning of each recording. They therefore select
slightly different frame sets when a recording's length is not divisible by six.

## Loading measurements

Compare `loader.workers=0`, `2`, and `4` with identical seeds and session lists.
Disable `driver.prefetch` when using zero workers. Training reports `data.load`
and `batch.compose` wall time alongside model stages. Include startup and total
elapsed time when assessing prefetch, since the stages can overlap.

The loader sends up to 16 frame requests together, allowing the video source
to sort reads by file and frame index within each group. It restores their
original order before augmentation, preserving the training batch contents. Clip frames also share a decode window. Requests
retain their individual augmentation seeds across worker counts.

`stream.decoder.cache_handles` limits open videos per process, with four as the
default. Nearby forward reads reuse the decoder; backward reads and gaps larger
than `stream.decoder.seek_gap` seek to a preceding keyframe. Each decoder uses
one CPU thread by default, avoiding a separate thread pool for every open video.

The PyAV backend requires constant-rate video with frame-count metadata.
Unsupported timestamps raise an error when encountered. It does not build an
index for variable-rate recordings. Random replay can still decode many
intermediate frames between keyframes, so measure throughput on representative
sessions before choosing worker counts or comparing with extracted JPEGs.
