# BabyCL on KrishnaCam

This page shows how to train visual BabyCL and its shuffled-frame baselines
directly from the KrishnaCam MP4 recordings.

## Training

Prepare the videos as described in
[KrishnaCam dataset setup](../../krishnacam/README.md#dataset-setup) and install
the training dependencies with `uv sync --extra train`. Set the dataset and
output locations, validate the config, then train one seed on a GPU:

```bash
export KRISHNACAM_ROOT=/path/to/krishnacam/kcam
export EGOLEARN_ARTIFACTS=/path/to/artifacts
uv run egl train projects/babycl/configs/krishnacam.yaml dry_run=true
uv run egl train projects/babycl/configs/krishnacam.yaml seed=0
```

The run makes one chronological pass and writes to
`$EGOLEARN_ARTIFACTS/babycl_krishnacam/seed_<seed>`. It saves resumable
snapshots every 500 updates, so you can continue an interrupted run by repeating
the command with `resume=auto`. Submit long runs with the
[cluster launcher](../../../docs/cluster.md). To decode on an NVIDIA GPU, add
`stream.decoder.device=cuda:0`, which needs PyAV with a CUDA-enabled FFmpeg.

## Differences from SAYCam

| Setting | SAYCam | KrishnaCam |
| --- | --- | --- |
| Input | Extracted 5 fps frames and transcripts | MP4 at about 30 fps, no audio |
| Objectives | Instance, segment SupCon and image–text | Instance and segment SupCon |
| Batch | 3 segments of 32 frames and 32 pairs | 4 segments of 32 frames |
| Segment memory | 406 FIFO, 2,030 reservoir | 138 FIFO, 692 reservoir |
| Run length | The paper run's step count | One full pass |

The KrishnaCam memory holds about 10% and 50% of the dataset's 69 hours in
three-minute segments.

## Lossless prepared videos

For repeated runs, you can prepare lossless videos that store the frames after
BabyCL's fixed resize and center crop, which makes decoding cheaper. Prepare
the videos, then train from them:

```bash
export KRISHNACAM_PREPARED=/path/to/krishnacam/prepared
uv run python -m egolearn.methods.babycl.prepare_video \
    --source-root "$KRISHNACAM_ROOT" --prepared-root "$KRISHNACAM_PREPARED" --workers 16
uv run egl train projects/babycl/configs/krishnacam_prepared.yaml seed=0
```

The copies can be larger than the originals, so you may want to add
`--sessions` with a few filenames to check the size first. Preparation verifies
each output against its source frames. The prepared videos decode on the CPU
and suit only BabyCL's training views, so keep the originals for evaluation.

## Shuffled-frame baselines

Two offline configs train on shuffled frames from the prepared videos, with no
replay memory. `krishnacam_iid_simclr.yaml` trains the instance objective alone
and `krishnacam_iid_tc.yaml` the temporal objective alone, with 128 frames per
update. Set `driver.max_steps` (default 11,900) to the update count of the
streaming run you compare against:

```bash
uv run egl train projects/babycl/configs/krishnacam_iid_simclr.yaml \
    seed=0 driver.max_steps=11900
export KRISHNACAM_TEMPORAL_MANIFEST=/path/to/temporal_labels.json
uv run egl train projects/babycl/configs/krishnacam_iid_tc.yaml \
    temporal_classes=2 seed=0 driver.max_steps=11900
```

The temporal baseline reads fixed labels for every frame from a manifest, and
`temporal_classes` must equal its class count. This example has one eight-frame
recording and two classes:

```json
{
  "format_version": 1,
  "num_classes": 2,
  "sessions": {
    "recording.mp4": {"fps": 30.0, "num_frames": 8, "segments": [[0, 3, 0], [3, 8, 1]]}
  }
}
```

List every recording, including empty ones with `"num_frames": 0` and
`"segments": []`. Each `[start, end, label]` has an exclusive `end`, and a
recording's segments cover all its frames without gaps or overlap.
