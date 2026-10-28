# Benchmark development

Add a benchmark when you need a new evaluation protocol for an existing model.
The standalone runner handles model loading, device checks, run records, and
result serialization.

Use a `Benchmark` for standalone evaluation, or its `Probe` subclass when the
same evaluation is also suitable for training hooks.
Training pipelines accept probes and reject standalone benchmarks.

## Registration

Implement `egolearn.evaluation.Benchmark` and register the class with
`BENCHMARKS.register("name")` from `egolearn.config.registry`. Its configuration
target becomes `benchmark:name`.

Include the module in the evaluation config's `_imports_` list, or the Python
API's `imports` argument, to register it before the runner constructs the
benchmark. For a probe, register with `PROBES` and use a
`probe:name` target.

The [ImageNet benchmark](../../src/egolearn/evaluation/benchmarks/imagenet_linear/benchmark.py)
provides a complete example with dataset preparation, a shared scorer, and
provenance. Keep checkpoint loading in the runner so your benchmark can also
evaluate a model already in memory.

## Preparation and scoring

Give the benchmark a unique `name`, validate settings in its constructor, and
defer dataset enumeration to `prepare()`. Make preparation idempotent so
repeated calls can reuse dataset metadata.

Implement `run(model, ctx)` to return a mapping from metric names to finite
scalar values. Standalone evaluation passes an `EvaluationContext` with
`device`, `seed`, and `derive_seed(name)`, without a training clock.
Its runner calls `prepare()` before `run()` and resets the seed for each
benchmark independently of suite order.

Leave the evaluated model's weights and buffers unchanged throughout your
benchmark's execution.
Train a fresh classifier and recompute model-dependent features on every run,
even when reusing a prepared benchmark instance. A decoder can enable its own
gradients within `run()` while leaving the pretrained encoder frozen.

### Training probes

The training hook calls `probe.run(model, driver.ctx)` directly, without calling
`prepare()`. If a probe needs preparation, call its idempotent `prepare()` at the
start of `run()` so it works in both training and standalone evaluation.

During training, `ctx` is the training `Context`, including its clock and RNG.
The hook enters evaluation mode and disables gradients, but does not apply the
standalone runner's per-benchmark seed reset or parameter-gradient freezing.
Keep training RNG state and model state unchanged within your probe.
Use a protocol-specific seed with isolated RNG state when scoring requires
random draws.

## Model requirements

Declare `requires_capabilities` as a tuple of model capabilities, such as
`image_representation` or `text_representation`. The runner checks the whole
suite before preparing any benchmark.

Select encoders and heads through the model's named interfaces, and make their
names configurable when the protocol applies to different model compositions.
The benchmark should report incompatible inputs instead of adapting the model's
weights during evaluation.

## Provenance and versions

Override `configuration()` to describe constructed instances, including protocol
settings that affect their scores, while config-driven evaluations also record
constructor defaults.

Override `metadata()` to report dataset sizes, split identity, and other
protocol-specific provenance, stating what each hash covers. A manifest hash identifies
the manifest's bytes, while a split hash can identify ordered paths and labels
without identifying image contents.

Set `version` and increment it when scoring or preprocessing changes the
protocol. Keep reporting changes separate from protocol changes so results
remain comparable for a stated version.

## Tests

Test the score on small synthetic inputs with known outcomes. Also test required
capabilities, invalid inputs, repeat preparation, and unchanged model state where
those behaviors apply.

Use the existing benchmark tests as examples, marking tests that need GPUs,
long runs, or paid models with the repository's corresponding pytest markers.
Data preparation and full benchmark runs must remain separate from tests that
run without the source dataset.
