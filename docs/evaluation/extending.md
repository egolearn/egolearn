# Benchmark development

Add a benchmark when you need a new evaluation protocol for existing models.
You write the data loading and scoring, and the evaluation runner handles model
loading, device placement, seeds, run records and the result file.

## Example probe

A benchmark is a class with a `run()` method that returns a mapping from metric
names to floats. This probe reports the mean norm of a model's pooled image
features on a fixed tensor of images:

```python
import torch

from egolearn.config.registry import PROBES
from egolearn.evaluation import Probe


@PROBES.register("feature_norm")
class FeatureNormProbe(Probe):
    """Mean L2 norm of pooled image features on a saved image tensor."""

    requires_capabilities = ("image_representation",)
    version = "1"

    def __init__(self, name: str, images: str, encoder: str = "image"):
        self.name = name
        self.images_path = images
        self.encoder = encoder
        self.images = None

    def prepare(self) -> None:
        if self.images is None:
            self.images = torch.load(self.images_path)

    def configuration(self) -> dict:
        return {"images": self.images_path, "encoder": self.encoder}

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        self.prepare()
        features = model.encode(self.encoder, self.images.to(ctx.device)).pooled
        return {"feature_norm": float(features.norm(dim=1).mean())}
```

Put the class in an importable module, then list that module under `_imports_`
and use the registered target in a [custom config](../evaluation.md#custom-configs):

```yaml
_imports_:
  - my_package.feature_norm
benchmarks:
  - _target_: "probe:feature_norm"
    name: feature_norm
    images: /path/to/images.pt
```

## Benchmarks and probes

Subclass `Probe` and register with `PROBES` when the evaluation can also run
inside training. The training loop calls `run()` without `prepare()`, which is
why the example calls `prepare()` itself. Subclass `Benchmark` and register with
`BENCHMARKS` for an evaluation that runs only on its own, such as fine-tuning.
Its target is then `benchmark:<name>`.

## Benchmark rules

- Validate settings in the constructor, and read the dataset in `prepare()`.
  Repeated calls to `prepare()` should be cheap.
- Leave the model's weights and buffers unchanged.
- Recompute features and fit any classifier again on every call to `run()`,
  because one benchmark instance can score several models.
- Return finite floats. A NaN or infinite metric fails the benchmark.
- Declare the model capabilities you use in `requires_capabilities`, such as
  `image_representation` or `text_representation`. The runner checks them for
  the whole suite before any benchmark starts.
- Reach the model only through its named encoders and heads.
- Draw random numbers from a generator seeded with `ctx.derive_seed(name)` or
  a seed setting of your own. Inside training, the global random state belongs
  to the training run.
- Return the settings that affect the score from `configuration()`, and the
  dataset size and identity from `metadata()`.
- Increase `version` when a change to scoring or preprocessing makes new scores
  incomparable with old ones.

The [ImageNet linear benchmark](../../src/egolearn/evaluation/benchmarks/imagenet_linear/benchmark.py)
is a complete example with dataset preparation, metadata and a shared scorer.

## Tests

Test the score on small synthetic inputs whose answer you know, so the tests
run without the source dataset. Mark tests that need a GPU, long runs or a paid
model with the `gpu`, `slow` or `llm` pytest markers.
