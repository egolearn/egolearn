# Data loading: blur, denormals, and the packed frame store

Where a frame's CPU time goes on the SAYCam 5 fps frame dumps (224 px JPEGs,
one process, medians, from an October 2026 study on the lab cluster: single
runs on a node whose other cores were busy):

| stage | cost |
| --- | ---: |
| read one frame, cold, through the FUSE-mounted squashfs | 2.6 ms |
| read one frame, warm (page cache) | 0.04 ms |
| JPEG decode | 0.3 ms |
| two `simclr_strong` views, reference torchvision blur | 10.5 + 25 ms |
| two `simclr_strong` views, separable blur (default) | 10.5 ms |
| 256 random frames, 32 reader processes, squashfs, cold | 1,206 ms |
| 256 random frames, 32 reader processes, packed store on scratch, cold | 36 ms |

Augmentation dominates the per-frame cost once the data is warm, and storage
dominates when it is not. This page covers the three controls that address
those two: the blur implementation, the worker arithmetic mode, and the
packed frame store.

## Gaussian blur implementation (`blur_impl`)

The SimCLR-style image views (`simclr_strong` in the core toolbox,
`babycl_ms_aug` in the BabyCL method) draw sigma from U(0.1, 2.0) and blur
with a kernel of about a tenth of the view size (23 taps at 224 px). Each view
takes a `blur_impl` argument:

| value | what it does | pixels |
| --- | --- | --- |
| `separable` (default) | the reference kernel, sigma draw, reflection padding and rounding point, applied as a horizontal and a vertical pass | within one intensity level of the reference |
| `torchvision` | torchvision's `GaussianBlur`, one full 2-D convolution | the bit-exact reference |
| `none` | skips the filter but consumes the sigma draw | an ablation; every later random decision is unchanged |

All three consume the same random numbers, so switching the implementation
never changes crops, flips, jitter, grayscale or solarization. The separable
pass is the default because it is about 4× cheaper per image and the
one-level difference is below the uint8 noise the pipeline already carries
through its PIL round trips. Set it per view when a bit-exact comparison with
an older run matters:

```text
views.v1.blur_impl=torchvision views.v2.blur_impl=torchvision      # IID preset
views.ms_v1.blur_impl=torchvision views.ms_v2.blur_impl=torchvision  # BabyCL
```

The BabyCL `pair` view keeps its own Pillow radius blur, which is part of that
method's numerical contract and has no implementation switch. Clip views
(`clip_simclr`) run torchvision's tensor blur on the whole clip and are not
affected by `blur_impl`.

## Flushing denormals in loader workers (`loader.flush_denormal`)

Small sigma draws put the kernel's tail weights in float32's subnormal range,
where x86 arithmetic runs through a microcode assist roughly a hundred times
slower than normal multiply-adds. About one draw in five hits it, which is
why the reference blur has a 35 ms median and a 120 ms tail. Those weights are
below 1e-38 and cannot move a pixel that is rounded to 8 bits.

`loader.flush_denormal: true` (the default) makes every worker process set the
CPU's flush-to-zero mode in its initializer. The loaded tensors are identical
with or without it (`tests/test_blur.py` checks the blur outputs on
adversarial images and compares worker-pool loading with in-process loading),
so `loader.workers` stays a pure throughput knob. The mode is never set in the
trainer process, and in-process loading (`loader.workers: 0`) leaves the
interpreter's arithmetic untouched. Set `loader.flush_denormal: false` on a
CPU without the mode (the worker initializer raises with that advice).

## Packed frame store (`stream:packed_frames`)

The frame-directory adapters open one file per frame. That is fine warm and
poor cold: random reads through a FUSE-mounted squashfs serialise in its
single daemon, and per-file JPEGs on a network filesystem pay a lookup per
open and need millions of inodes. Replay in streaming (memory refs are never
read when admitted) and the first epoch of offline training are exactly that
cold random pattern. A packed store concatenates the encoded frames into a
few large shard files with a numpy index, so a frame is one `pread` and a
contiguous clip is one read.

Measured on the cluster's network scratch, ms per 256 random frames with 32
reader processes (cold / warm), and ms per 16-frame clip:

| layout | random frames | clips |
| --- | ---: | ---: |
| squashfs of per-frame JPEGs | 1,206 / 10 | 0.7 / 0.4 |
| per-file JPEGs on scratch | 80 / 64 | 3.8 / 3.2 |
| packed store on scratch | 36 / 9 | 0.5 / 0.3 |
| parquet shards on scratch (64-row groups) | 875 / 26 | 1.2 / 0.9 |
| packed store on node-local NVMe | 13 / 8 | 0.4 / 0.3 |

### Building a store

```bash
uv run python -m egolearn.data.packed pack /path/to/saycam_5fps /path/to/saycam_5fps_packed \
    --include "S_*" --workers 32
uv run python -m egolearn.data.packed verify /path/to/saycam_5fps_packed /path/to/saycam_5fps --sample 20000
uv run python -m egolearn.data.packed info /path/to/saycam_5fps_packed
```

`pack` lists sessions and frames with the same rules as `stream:saycam`
(`--include`, `--sessions`, `--naming timestamp|frame_index`, `--fps`), packs
one session per process, and writes the manifest last, so a directory without
one is an incomplete build. Frames keep their original encoded bytes: nothing
is re-encoded, and `verify` checks byte identity and frame times against the
source directories. `--force` replaces an existing store (only the files the
previous build wrote, or an interrupted build left behind). The SAYCam S
child (3.65 M frames, 76 GB) packed in 4 minutes with 32 workers into 1,242
shard files of up to 256 MB (`--shard-bytes`), one or more per session;
verifying 20,000 random frames against the source took 77 s.

A store directory holds:

| file | contents |
| --- | --- |
| `manifest.json` | format tag, the source's fps and naming, the shard list, the session table (id, first global frame, count) in stream order |
| `index.npy` | int64 `[N, 3]`: shard, byte offset, byte length of each frame |
| `times.npy` | float64 `[N]`: media time of each frame |
| `names.npy` | the source frame file names, so precomputed indexes (BabyCL storyboards) validate their frame order against the store as they do against a directory |
| `shards/<session>.<k>.bin` | the frames' encoded bytes, back to back |

### Using a store

```yaml
stream:
  _target_: "stream:packed_frames"
  root_dir: /path/to/saycam_5fps_packed
  include: "S_*"                                      # or sessions: [...]
  transcript_json: /path/to/expand_correct_seg.json   # as with stream:saycam
```

`fps` and the naming scheme come from the manifest. Sessions, times, channels,
refs and pixels are the same as `stream:saycam` on the directories the store
was built from, so snapshots, memory contents and storyboards from a
frame-directory run keep working, and the two adapters can be swapped by
config. The Loader's clip decode fetches a clip's frames in one call
(`Stream.open_frames`), which the store serves as one read; the store's index
is memory-mapped and only its path is pickled, so the stream reaches Loader
workers as a few kilobytes plus the transcript instead of a per-session
listing cache. That also removes the worker start-up stall the listing cache
causes: with 16 workers on a 32-session subset the pool was ready in 19 s
instead of 202 s, because a spawn whose arguments exceed the pipe buffer
blocks the trainer until the child has imported torch. Each worker keeps its
shard files open (up to a quarter of its open-file limit, so all of them
under a job step's limit; an interactive shell's 1024 degrades to an LRU).

Staging the store on a job's node-local disk (copy the directory, point
`root_dir` at the copy) turns every read into the last row of the table above.
It is worth it only for jobs long enough to amortise the copy: random reads
from the packed store on scratch are already decode-bound once warm.
