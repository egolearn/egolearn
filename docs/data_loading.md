# Data loading

This page explains how to speed up training from extracted SAYCam or BabyView
frames, with a packed frame store and loader workers. It also covers the blur
setting to use when you compare against a reference implementation. To train
directly from MP4 files instead, use `stream:video` with `decoder:pyav`, as the
[KrishnaCam guide](../projects/babycl/docs/krishnacam.md) describes.

## Packed frame store

A packed store holds the frames of a frame directory in a few large shard
files plus an index. Frames keep their original encoded bytes and order, so a
store decodes to the same pixels, and gives the same frame references, as the
directory it was built from.

A frame directory costs one file open per frame, which is slow for cold
random reads on a network filesystem, the pattern of streaming replay and of
the first offline epoch. A packed store reads each frame with one positional
read into a shard. It does not speed up augmentation, which often dominates
once frames are cached, and it needs room for a second copy of the frames.

### Building a store

Pack the sessions you want from a directory that holds one subdirectory of
frames per session, then compare the store with its source:

```bash
uv run python -m egolearn.data.packed pack /path/to/saycam_5fps /path/to/saycam_5fps_packed \
    --include "S_*" --workers 32
uv run python -m egolearn.data.packed verify /path/to/saycam_5fps_packed /path/to/saycam_5fps \
    --sample 20000
uv run python -m egolearn.data.packed info /path/to/saycam_5fps_packed
```

`--include` selects sessions with a glob pattern, and `--workers` packs that
many sessions in parallel. `verify` checks the sessions, frame names, order and
times, compares the bytes of 20,000 random frames, and prints
`ok: <count> frames byte-identical to <source>`. `info` prints the store's
counts, frame rate and naming scheme. Packing into an existing store fails
unless you pass `--force`, and `--help` lists the other options.

SAYCam frame names are timestamps, which is the default. BabyView frame names
are frame indexes, so pack BabyView frames with `--naming frame_index` and the
extraction frame rate, such as `--fps 5.0`. `verify` cannot catch a wrong
naming or rate, so check what `info` reports.

### Using a store

Change the stream target and root in your training config, and keep the same
session selection and transcripts as the frame-directory run:

```yaml
stream:
  _target_: "stream:packed_frames"
  root_dir: /path/to/saycam_5fps_packed
  include: "S_*"
  transcript_json: /path/to/expand_correct_seg.json
```

The store supplies the frame rate and naming, and any `fps` or `naming` keys
left in your config must match it. The packed stream does not support the
`session_glob`, `natural_order` and `strict_filenames` options of the
frame-directory stream. The V-JEPA 2 presets set these options, so they cannot
switch to a packed store.

## Loader workers and prefetch

Set `loader.workers` to decode and augment in worker processes instead of the
training process, which is the default of `0`. `driver.prefetch` prepares the
next batch while the current one trains and needs at least one worker:

```text
loader.workers=8 +driver.prefetch=true
```

In offline runs, prefetch works within each epoch. In streaming runs, it takes
effect only with `driver.rate.resample=true` and `driver.rate.steps_per_admit`
above 1, and it never prepares a batch across a
[stream advance](design/architecture.md#training-and-concurrency). These rate
settings change the training protocol, so do not change them only to enable
prefetch.

## Gaussian blur implementation

The `simclr_strong` and `babycl_ms_aug` views take a `blur_impl` setting. The
default, `separable`, is faster than the reference `torchvision` blur and can
differ from it by one intensity level per pixel, and `none` turns blur off for
ablations. All three draw the same random numbers, so the other augmentations
do not change. BabyCL's SAYCam configs already use `torchvision`. To use it in
another preset, add the setting to its views:

```text
# offline_iid_saycam
+views.v1.blur_impl=torchvision +views.v2.blur_impl=torchvision
# Memory Storyboard
+views.ms_v1.blur_impl=torchvision +views.ms_v2.blur_impl=torchvision
```

Loader workers flush denormal floats to zero, because tiny blur weights are
slow on some CPUs. If worker start-up reports that your CPU cannot do this, set
`loader.flush_denormal=false`.
