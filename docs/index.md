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
- **[Design](design/architecture.md)**: how streams, memories, models, and
  probes fit together.

## Methods

- **[BabyCL](projects/babycl.md)**: continual visual and verbal learning from
  SAYCam in a single chronological pass.
- **[Memory Storyboard](projects/storyboard.md)**: streaming visual
  self-supervised learning with temporal segmentation.
