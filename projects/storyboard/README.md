# Memory Storyboard

Memory Storyboard learns visual representations from a chronological stream using
temporal segmentation and two memory tiers. This project provides the
[Memory Storyboard](https://arxiv.org/abs/2501.12254) method and its SAYCam training
configuration.

## Training

Use the [SAYCam config](configs/saycam.yaml) with your frame-directory root configured
under `stream.root_dir`. The recipe expects raw, upside-down 5 fps SAYCam frames
and flips them during loading; BabyCL instead expects upright input frames.
See [dataset access](../../docs/evaluation/data.md#saycam-access)
for the source data requirements.

```bash
egl train projects/storyboard/configs/saycam.yaml dry_run=true
egl train projects/storyboard/configs/saycam.yaml
```

Each batch draws 64 frames from the short memory and 448 from the long memory.
Instance discrimination uses both slots together, while temporal classification
uses the long-memory slot. The default schedule performs one training update after
each advance of 64 stream frames.

The configuration is a code-level port, without a numerical-parity claim.
The packaged recipe uses float32 arithmetic in one process. The shared trainer
supports opt-in mixed precision through `driver.amp`; numerical parity with the
original mixed-precision recipe has not been established.

## Memory and segmentation

The short FIFO tier retains every eighth frame, while the long weighted reservoir
receives every frame. Both tiers admit directly from the stream, so the long memory
does not depend on short-tier eviction.

Segmentation uses the learner's current backbone. When the short window has fully
turned over, the method reclusters its frames and revises their class labels.
Prototype merging samples up to 32 frames per class from the long memory and uses
a dynamic quantile threshold to decide which classes to merge.

You can change memory capacities and training rate in the config. The default
`driver.rate.resample: false` makes additional updates reuse the batch, matching the
original method's repeated-gradient setting. No Memory Storyboard-specific KrishnaCam
recipe is packaged. The [KrishnaCam guide](../krishnacam/README.md) covers general
video training; adapting this method also requires matching timing and preprocessing.

## Evaluation

Evaluate saved checkpoints with the [standalone evaluator](../../docs/evaluation.md).
The released Memory Storyboard ImageNet evaluation uses 112-pixel inputs, while the
packaged ImageNet evaluation defaults to 224 pixels. Set `benchmarks.0.size=112`
when you need that resolution, and match the remaining protocol settings before
comparing scores.

## Implementation

The [method package](../../src/egolearn/methods/storyboard/) contains the original
greedy hierarchical clustering algorithm, reclustering and prototype merging, and
the learner-backed embedding component. Component docstrings document the preserved
clustering behavior and adaptations to the framework.

Labels are resolved through a shared label store when batches are assembled.
The recluster trigger is checked per frame rather than per training update,
preserving the original threshold crossing within a stream advance. The
[architecture guide](../../docs/design/architecture.md#method-differences) compares
this design with BabyCL.
