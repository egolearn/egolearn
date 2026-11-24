# KrishnaCam video baselines

This project trains a visual SimCLR model directly from the KrishnaCam MP4
recordings, once as a chronological stream with replay and once offline on
shuffled frames. The configs are a starting point for video experiments and
reproduce no published result. For BabyCL on the same videos, see
[BabyCL on KrishnaCam](../babycl/docs/krishnacam.md).

## Dataset setup

Download the video archive linked from the
[KrishnaCam project page](https://krsingh.cs.ucdavis.edu/krishna_files/papers/krishnacam/krishnacam.html)
and extract it outside the repository, keeping the `kcam/` directory, its
timestamped filenames and the original encoding. Allow about 172 GB for the
archive and the extracted files, which hold 436 recordings and 69 hours of video
at about 30 fps. Then install the training dependencies and point the configs at
the videos:

```bash
uv sync --extra train
export KRISHNACAM_ROOT=/path/to/krishnacam/kcam
```

The configs read the videos with `stream:video` in filename order, and you can
set `stream.sessions` to a list of filenames to select a subset or another
order. The `decoder:pyav` backend decodes frames on demand and requires
constant-frame-rate video with frame-count metadata.

The project page states no license for redistributing trained weights, so
confirm the release terms before you publish a checkpoint.

## Training

Validate both configs with a dry run, then remove `dry_run=true` to train on a
GPU:

```bash
uv run egl train projects/krishnacam/configs/streaming.yaml dry_run=true
uv run egl train projects/krishnacam/configs/offline.yaml dry_run=true
```

For a short check on a CPU, add
`device=cpu loader.workers=0 driver.prefetch=false driver.max_steps=20`.

## Configuration

Both configs train a ResNet-18 from scratch with the same augmentations and
contrastive loss. They use batches of 64 frames and stop after 1,000 updates,
which you can change with `driver.max_steps`.

| Setting | `streaming` | `offline` |
| --- | --- | --- |
| Frames | Every sixth, in recording order | Every sixth, shuffled for one epoch |
| Memory | 1,024-frame FIFO and 8,192-frame reservoir | None |
| Updates | Four per 384 source frames | One per batch |

Streaming subsampling continues across recording boundaries, while offline
subsampling restarts in each recording, so the two select slightly different
frames. Random replay can decode many frames between keyframes, so measure
throughput before you choose `loader.workers`.
