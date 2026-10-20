# Data-loading performance

Packed storage improves cold reads and worker startup, while augmentation can
still dominate batch loading. This page records the measurements behind the
defaults and explains where those benefits stop.

These measurements come from an October 2026 study of SAYCam 5 fps frame
dumps on the lab cluster, with costs that depend on the hardware and filesystem.
The source frames are 224 px JPEGs. The per-frame measurements are medians from
single runs using one process on a node whose other cores were busy.
The tables do not report variation across repeated runs.

For the packing commands and configuration options, see
[data loading](../data_loading.md).

## Per-frame costs

Warm reads are cheap enough that image augmentation dominates the measured
per-frame time, while cold storage reads need separate attention.

| Stage | Median cost |
| --- | ---: |
| Read one cold frame through the FUSE-mounted squashfs | 2.6 ms |
| Read one warm frame from the page cache | 0.04 ms |
| JPEG decode | 0.3 ms |
| Two `simclr_strong` views with reference torchvision blur | 35.5 ms |
| Two `simclr_strong` views with separable blur | 10.5 ms |

The blur implementations draw sigma from U(0.1, 2.0) and use a kernel about
one tenth of the view width, giving 23 taps at 224 px. A separable
filter replaces the full two-dimensional convolution with horizontal and
vertical passes. Both use the same sigma draw, reflection padding, and final
rounding point, but the accumulation order can change a pixel by one intensity
level.

Small sigma draws can put tail weights into the float32 subnormal range.
The worker-wide flush-to-zero mode avoids their arithmetic cost on supported
CPUs and also applies to other subnormal values in those workers. Those tiny
weights do not affect the rounded 8-bit blur outputs in the test cases.
The trainer process retains its arithmetic settings.

## Storage layouts

Packed stores reduce the filesystem work required to read random frames.
Per-frame files require separate opens, while FUSE-mounted squashfs reads
also pass through a shared daemon. Packing concatenates the original encoded
bytes and indexes their offsets within larger files.

The following measurements include reading and decoding with 32 reader
processes. Random batches contain 256 frames, and clips contain 16 frames.
Each cell reports cold / warm time in milliseconds.

| Layout | Random frames | Clips |
| --- | ---: | ---: |
| Squashfs of per-frame JPEGs | 1,206 / 10 | 0.7 / 0.4 |
| Per-file JPEGs on network scratch | 80 / 64 | 3.8 / 3.2 |
| Packed store on network scratch | 36 / 9 | 0.5 / 0.3 |
| Parquet shards on network scratch, 64-row groups | 875 / 26 | 1.2 / 0.9 |
| Packed store on node-local NVMe | 13 / 8 | 0.4 / 0.3 |

A packed frame needs one positional read, and neighboring frames within a
shard can share that read. This avoids opening one file per frame, although
a clip crossing a shard boundary still requires more than one read.

## Batch-loading costs

The training loader narrows the storage difference because it also receives
worker tensors and applies augmentations. These measurements used a fresh CPU
node, 16 workers, and a 33-session subset containing 94,000 frames. Batch
times are medians over eight batches of 256 random frames, except for the
clip workload.

| Through the loader, 16 workers | Squashfs | Packed store |
| --- | ---: | ---: |
| Worker pool ready | 213 s | 18 s |
| First batch on a cold node, decode only | 31.0 s | 0.48 s |
| Later cold batches, decode only | 217 ms | 335 ms |
| Warm batches, decode only | 195 ms | 198 ms |
| Cold / warm, IID SimCLR two views | 567 / 521 ms | 539 / 500 ms |
| Cold / warm, 64 clips of 16 frames | 848 / 806 ms | 764 / 665 ms |

The large startup difference reflects the frame-directory adapter's listing
cache. Passing that cache to spawned workers can block the trainer while each
child imports torch. A packed stream passes the store path and maps the index
inside each worker.

The first squashfs batch also fetched directory metadata and data blocks
through the daemon, while later batches benefited from read-ahead over the small
subset. The packed store avoided that first-batch delay but did not beat
squashfs on every later batch. Warm decode-only results were nearly equal,
and two-view augmentation reduced the storage advantage.

## Packing and staging costs

Building a packed store takes time and requires another copy of the encoded
frames. In the study, packing the SAYCam S child took four minutes with
32 workers. Its 3.65 million frames occupied 76 GB and produced 1,242 shards
with a target size of 256 MiB. Verifying a sample of 20,000 frames against
the source took 77 seconds.

Node-local NVMe reduced cold random-read time in the storage comparison,
excluding the time needed to copy the store onto the node.
Staging therefore makes sense only when the saved loading time exceeds the
copy cost. Warm reads on network scratch were already close to the measured
decode limit.
