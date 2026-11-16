# egolearn

EgoLearn trains visual and image–text models from first-person video streams.
Streaming runs retain observations in bounded replay memory, while offline runs
use the complete selected dataset. You can evaluate saved models independently
or schedule small benchmarks during training.

## Start here

- **[Overview and quickstart](overview.md)**: install, run a dry run, train.
- **[Command line](cli.md)**: `egl train` and `egl eval`.
- **[Evaluate a model](evaluation.md)**: benchmarks, data access, and the
  evaluation API.
- **[Model checkpoints](evaluation/hub.md#released-models)**: choose weights and request access.
- **[Status and limitations](status.md)**: supported workflows and their constraints.
- **[Design](design/architecture.md)**: how streams, memories, models, and
  probes fit together.

## Methods

- **[BabyCL](projects/babycl.md)**: continual visual and verbal learning from
  SAYCam in a single chronological pass.
- **[Memory Storyboard](projects/storyboard.md)**: streaming visual
  self-supervised learning with temporal segmentation.
- **[V-JEPA 2](projects/vjepa2.md)**: video representation learning with offline
  and streaming recipes.
- **[KrishnaCam](projects/krishnacam.md)**: visual training directly from MP4 recordings.

## Benchmarks

Evaluate object recognition with Labeled-S, BabyView, and ImageNet, or test
image–text alignment with VTWT and Baby Winoground. Shape bias, spatial relations,
IntPhys 2, and GRASP measure generalization beyond category labels.
EgoSchema and Ego4D NLQ evaluate served generative models. The
[evaluation guide](evaluation.md) lists protocols and data requirements.
[Benchmark suites](benchmark_suites.md) run reproducible comparisons across models.
