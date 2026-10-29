# Data loading

Use a packed frame store to reduce file opens when training from extracted
SAYCam or BabyView frames. This guide covers building and checking a store,
configuring loader workers, and choosing image-loading settings for reproduction.

You need an existing frame directory and access to its source data. Packing
copies the encoded frames into shards, so allow space for another copy of the
selected frames. The commands below use placeholder paths for your local data.

For direct MP4 decoding, use `stream:video` with `decoder:pyav`. The
[KrishnaCam guide](../projects/babycl/docs/krishnacam.md) covers lossless prepared
videos and matched-request loading benchmarks without extracted JPEGs.

## Packed frame store (`stream:packed_frames`)

A packed store replaces per-frame files with shard files and an index,
preserving the source frames and their order without re-encoding them.
Use it when cold random reads or worker startup dominate data loading. See
[the performance study](design/data_loading_performance.md) for measurements
and their limits.

### Building a store

For timestamp-named SAYCam frames, pack the sessions you want to use:

```bash
uv run python -m egolearn.data.packed pack /path/to/saycam_5fps /path/to/saycam_5fps_packed \
    --include "S_*" --workers 32
```

The input directory contains one flat subdirectory per session, with session
selection and frame ordering following the `stream:saycam` adapter's rules.
Choose options that match your source:

| Option | Default | Use |
| --- | --- | --- |
| `--include` | All session directories | Select session names with a glob pattern |
| `--sessions` | Unset | Supply an explicit ordered list, overriding `--include` |
| `--naming` | `timestamp` | Use `frame_index` for names such as `frame_00000001.jpg` |
| `--fps` | `5.0` | Set the frame rate used to derive times from frame-index names |
| `--workers` | `0` | Pack sessions in parallel with this many processes; `0` runs in-process |
| `--shard-bytes` | `256M` | Set the target shard size; `M` means MiB |

The command prints progress and a final frame and shard count.
It writes `manifest.json` after the other files, so a store without that
manifest is incomplete.

If the destination contains an earlier or interrupted build, packing fails
unless you use `--force` to replace it. This removes the store's metadata
files and all `shards/*.bin` files before packing.

#### BabyView frame times

BabyView filenames encode frame indexes, so packing requires
`--naming frame_index` and the frame rate used during extraction. For frames
extracted at 5 fps:

```bash
uv run python -m egolearn.data.packed pack /path/to/babyview_5fps /path/to/babyview_5fps_packed \
    --naming frame_index --fps 5.0 --workers 32
uv run python -m egolearn.data.packed info /path/to/babyview_5fps_packed
```

Replace `5.0` with your extraction frame rate and confirm that `info` reports
that rate and `naming=frame_index`. With timestamp naming, a file such as
`frame_00000123.jpg` is read as 123 seconds instead of 24.4 seconds at 5 fps.
Verification uses the store's own naming and frame rate, so it cannot detect
an incorrect choice made during packing.

### Verification

Compare the store with its source before changing a training configuration:

```bash
uv run python -m egolearn.data.packed verify /path/to/saycam_5fps_packed /path/to/saycam_5fps \
    --sample 20000
uv run python -m egolearn.data.packed info /path/to/saycam_5fps_packed
```

Verification checks the selected sessions, frame names, order, and times
against the complete source listing. `--sample 20000` limits the byte comparison
to 20,000 randomly selected frames, or all frames if the store is smaller.
Omit `--sample` to compare every frame's encoded bytes.

A successful check prints `ok: <count> frames byte-identical to <source>`.
The `info` command prints the number of frames, sessions, and shards, along
with the frame rate and naming scheme. Verification fails if the source
has changed, including newly populated sessions matching the build's selection.

### Using a store

The packer and packed adapter do not accept `session_glob`, `natural_order`, or
`strict_filenames`. The V-JEPA 2 presets use these frame-directory options and
cannot be switched directly to a packed store. In particular, nested BabyView
sessions and their natural order are not supported by the current packed format.

Change the stream target and root in your existing training configuration:

```yaml
stream:
  _target_: "stream:packed_frames"
  root_dir: /path/to/saycam_5fps_packed
  include: "S_*"
  transcript_json: /path/to/expand_correct_seg.json
```

Keep the same session selection and transcript file as the frame-directory
run. Omit `transcript_json` for a stream without transcripts. You can use
`sessions: [...]` instead of `include` to select an explicit ordered subset.

The manifest supplies the frame rate and naming scheme, so existing `fps`
and `naming` keys in your configuration must match the store.
With the same session selection and transcripts, both adapters produce the
same sessions, times, channels, frame references, and decoded pixels. Existing
memory contents, snapshots, and storyboard indexes retain their frame references.

For BabyView, add explicit `naming` and `fps` keys so the adapter checks the
manifest against your intended settings. A store built from 5 fps frames uses:

```yaml
stream:
  _target_: "stream:packed_frames"
  root_dir: /path/to/babyview_5fps_packed
  naming: frame_index
  fps: 5.0
```

Retain your transcript and session-selection settings in this configuration.

If you want to use node-local storage, measure the copy time against the
expected training benefit first. Copy the whole store directory and set
`root_dir` to the copy.

### Store contents

Keep the metadata and shards together when moving a store:

| File | Contents |
| --- | --- |
| `manifest.json` | Format, frame rate, naming scheme, shard list, session order, and build selection |
| `index.npy` | An int64 `[N, 3]` array of shard number, byte offset, and byte length |
| `times.npy` | A float64 `[N]` array of frame times in seconds |
| `names.npy` | Source frame filenames, used to validate precomputed indexes |
| `shards/<session>.<k>.bin` | Encoded frame bytes in frame order |

## Loader workers and prefetch

Set `loader.workers` to run frame and clip decoding and augmentation in worker
processes. The default, `0`, loads in the trainer process. Choose a worker
count within your job's CPU allocation and measure batch-loading time.
The packing command's `--workers` option controls only store construction.

Frame-directory workers omit the listing cache during spawn and list each
session when first accessed. Keep source directories unchanged throughout a run.

To overlap the next batch's preparation with training, enable
`driver.prefetch`. It requires `loader.workers >= 1`. For packaged presets
that omit the prefetch key, append these overrides to your training command:

```text
loader.workers=8 +driver.prefetch=true
```

In streaming mode, prefetch takes effect only when
`driver.rate.resample=true` and `driver.rate.steps_per_admit > 1`.
It prepares a fresh batch within the current
[stream advance](design/architecture.md#training-and-concurrency) and does not
cross into the next one. These rate settings change the training protocol, so preserve
your intended sampling and update schedule when enabling prefetch.

In offline mode, prefetch prepares the next batch within the current epoch.
Both modes reject prefetch when a loss or hook declares that it changes
future batch sampling.

## Gaussian blur implementation (`blur_impl`)

Use `blur_impl=torchvision` when you need the reference blur for a numerical
comparison. The `simclr_strong` and `babycl_ms_aug` image views otherwise use
the faster separable implementation.

| Value | Behavior | Pixel agreement with the reference |
| --- | --- | --- |
| `separable` | Default; applies the same Gaussian kernel in horizontal and vertical passes | Can differ by one intensity level |
| `torchvision` | Uses torchvision's full two-dimensional Gaussian convolution | Reference implementation |
| `none` | Skips filtering but consumes the same sigma draw | Blur ablation |

All three consume the same random numbers, preserving subsequent crop, flip,
color jitter, grayscale, and solarization decisions. Append the overrides
for your preset to its training command:

```text
# offline_iid_saycam
+views.v1.blur_impl=torchvision +views.v2.blur_impl=torchvision
# BabyCL and Memory Storyboard
+views.ms_v1.blur_impl=torchvision +views.ms_v2.blur_impl=torchvision
```

The leading `+` adds a key that the packaged presets do not declare.
BabyCL's `pair` view keeps its Pillow radius blur, while `clip_simclr` uses
torchvision's tensor blur. Neither accepts `blur_impl`.

## Flushing denormals in loader workers (`loader.flush_denormal`)

Leave `loader.flush_denormal=true` enabled unless your CPU cannot support it.
This enables a worker-wide arithmetic mode that flushes subnormal values to
zero. Small blur-kernel weights benefit from this mode because subnormal
arithmetic is slow on some CPUs.
Enabling this setting leaves rounded 8-bit blur outputs unchanged in the
repository's tests, including comparisons across worker counts.

This setting applies only inside loader worker processes, leaving the trainer's
arithmetic and in-process loading with `loader.workers=0` unchanged.
If worker initialization reports that the CPU cannot flush denormals, set
`loader.flush_denormal=false`.
