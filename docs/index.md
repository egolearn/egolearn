# egolearn

Learning from first-person video streams: streaming self-supervised learning,
continual learning and replay, temporal segmentation, multimodal learning,
memory, and egocentric video question answering.

egolearn reads egocentric video as a time-ordered stream, decides what to keep
in bounded replay buffers, trains models from those buffers, and scores them
with developmental and vision benchmarks while they train or afterwards.

## Start here

- **[Overview and quickstart](overview.md)**: install, run a dry run, train.
- **[Command line](cli.md)**: `egl train` and `egl eval`.
- **[Evaluate a model](evaluation.md)**: benchmarks, data access, and the
  evaluation API.
- **[Design](design/architecture.md)**: how streams, memories, models, and
  probes fit together.

## Methods

- **[BabyCL](projects/babycl.md)**: continual visual and verbal learning from
  SAYCam in a single chronological pass.
- **[Memory Storyboard](projects/storyboard.md)**: streaming visual
  self-supervised learning with temporal segmentation.
