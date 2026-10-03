# Evaluating trained models

The standalone evaluator runs benchmarks on a loaded `ComposedModel` or an
EgoLearn checkpoint. It constructs no training stream, replay memory, training
optimizer, or driver. Install the `train` extra for the PyTorch dependencies.

## Python API

```python
from egolearn.evaluation import evaluate, evaluate_checkpoint, load_model

benchmarks = [
    {
        "_target_": "benchmark:imagenet_linear_sgd",
        "name": "imagenet",
        "data_root": "/datasets/imagenet",
        "encoder": "image",
    }
]
imports = ["egolearn.evaluation.benchmarks.imagenet_linear"]

result = evaluate_checkpoint(
    "run/final.pth",
    benchmarks=benchmarks,
    imports=imports,
    device="cuda",
    seed=0,
)
print(result.results)
result.save_json("evaluation.json")

# Model loading is also available separately.
model = load_model("run/final.pth", device="cuda")
result = evaluate(model, benchmarks=benchmarks, imports=imports, seed=0)
```

`evaluate` infers the device from the loaded model. If supplied, `device` must
match the model's current device; move the model explicitly before evaluating.
The runner temporarily freezes the model and restores each module's training
mode, parameter gradient flags, and Python/NumPy/PyTorch RNG states, including
on failure. Benchmarks must leave model weights and buffers unchanged. A linear
decoding benchmark can enable gradients for its own newly initialized decoder;
the pretrained encoder remains frozen.

The result records metrics by benchmark name, constructor settings including
defaults for config-driven calls, protocol versions, dataset metadata, seeds,
package version, source commit and dirty flag, and resolved model config.
`code_revision` reports `"unavailable"` outside a source checkout or when Git
cannot be queried. Checkpoint evaluations record the file name, SHA-256, and
training step when available; no checkpoint directory is stored. Benchmark names
must be unique within an evaluation.

Existing small evaluations remain available with their `probe:*` configuration
targets. Pass those configs in `benchmarks` to evaluate them after training.
There is one scoring implementation for training probes and standalone use.
Linear probes record their own seed (for SGD), hyperparameters, class names,
split sizes and split SHA-256, including when passed as constructed objects.
Generic split hashes cover ordered relative paths, labels and classes, not image
bytes. Manifest-backed probes also record the manifest's byte hash; Labeled-S
records the verified canonical split hash and source.

If a benchmark fails, `EvaluationError.partial_result` retains completed
benchmarks and their provenance; `benchmark` identifies the failed benchmark
and `__cause__` retains its original error. The CLI saves this partial result to
`output_json` before exiting unsuccessfully. In Python:

```python
from egolearn.evaluation import EvaluationError

try:
    result = evaluate_checkpoint("run/final.pth", benchmarks=benchmarks, imports=imports)
except EvaluationError as error:
    error.partial_result.save_json("partial-evaluation.json")
    raise
```

## BabyVLM benchmark data

The Visual Two-Word Test (VTWT, `probe:vtwt`) and Baby Winoground
(`probe:baby_winoground`) come from Wang et al., *BabyVLM: Data-Efficient
Pretraining of VLMs Inspired by Infant Learning* (2025). See the
[paper website](https://shawnking98.github.io/BabyVLM/) and the authors'
[evaluation repository](https://github.com/ShawnKing98/babylmms-eval).

| Benchmark | Public labels | Required images |
| --- | --- | --- |
| VTWT | [VTWT labels][vtwt-labels] | SAYCam frames |
| Baby Winoground | [Baby Winoground labels][bw-labels] | SAYCam frames and synthetic negatives |

Follow the [SAYCam access and preparation instructions][saycam-preparation] in
this repository's README to obtain the frames. For Baby Winoground, also obtain
the `BabyVLM_V1` folder from [Databrary][babyvlm-images] for its synthetic negative
images. The repository distributes labels; the images are obtained separately
under the applicable Databrary data-use terms. Researchers can apply for Databrary
access with approval from their institution's IRB.

For EgoLearn, configure VTWT's `csv_path` and `image_dir`; for Baby Winoground,
configure `csv_path`, `pos_dir` (SAYCam frames), and `neg_dir` (synthetic images).
The probes expect CSV inputs with the schemas documented in the
[VTWT module](../src/egolearn/evaluation/benchmarks/vtwt/__init__.py) and
[Baby Winoground module](../src/egolearn/evaluation/benchmarks/baby_winoground/__init__.py).
Both also require the evaluated model's `vocab_json`, `vision_head`, and
appropriate `image_norm` (`imagenet` for EgoLearn-trained BabyCL models,
`clip` for the CVCL family).

[vtwt-labels]: https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/vtwt
[bw-labels]: https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/baby_winoground
[saycam-preparation]: ../README.md#saycam-dataset-access
[babyvlm-images]: https://nyu.databrary.org/volume/564/folder/77120

## Checkpoint compatibility

Both `CheckpointHook` files (`step_*.pth`, `final.pth`) and training snapshots
(`snapshot_*.pth`) are supported with an explicit `trusted_snapshot=True` opt-in. New files produced from a pipeline-built model
include a versioned construction config: encoders, heads, predictors, frozen
roles, registration imports, and resolved values such as vocabulary size.

For older checkpoints, or checkpoints of manually assembled models, pass
`model_config="outputs/<date>/<time>/.hydra/config.yaml"`. An explicit model-section mapping also
works. Use a resolved config, not an uncomposed YAML containing Hydra defaults.
Only model-related nodes are resolved: training dataset environment variables
and probe data paths are not needed. An old `vocab_size: auto` setting still
needs the original tokenizer config/file or an explicit vocabulary size.

Weights load strictly, and the export contract is checked when present.
Pretrained encoder initialization is disabled while restoring a checkpoint.
Training RNGs, optimizer state, and stream replay are never restored. Existing
checkpoint files are read in full; only the model is kept. Ordinary checkpoints
use restricted unpickling; there is no automatic unsafe fallback for snapshots.
See the [checkpoint trust rule](../SECURITY.md#checkpoint-trust).

## Command line

```bash
uv run python -m egolearn.evaluate --config-name imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  device=cuda output_json=imagenet_results.json
```

For an older checkpoint, append
`'model_config=outputs/<date>/<time>/.hydra/config.yaml'`, substituting the training
run's date and time (or your explicit `hydra.run.dir`). For a trusted full
snapshot, also pass `trusted_snapshot=true`.
The packaged ImageNet config exposes its protocol settings under `benchmarks.0`,
for example `benchmarks.0.encoder=image` or `benchmarks.0.batch_size=256`.
`python -m egolearn.evaluate --help` lists the default evaluation configuration.
Custom suites can use `--config-dir` and `--config-name`, just like training.

## ImageNet migration

Full ImageNet linear decoding is now `benchmark:imagenet_linear_sgd`, implemented
by `ImageNetLinearBenchmark` in `evaluation/benchmarks/imagenet_linear/benchmark.py`.
Remove `probe:imagenet_linear_sgd` from training configs and place the new target
under `benchmarks` in a separate evaluation config. The old ImageNet probe class
and registry entry have been removed. Training pipelines reject standalone
benchmark entries in their `probes` list.

The protocol is unchanged: full ImageFolder train/val splits, frozen encoder,
stochastic training crops, SGD/LARC, and cosine learning-rate decay. Dataset
enumeration happens in `prepare()` or the first `run()`, never in the constructor.
Passing the same benchmark instance to multiple evaluations reuses its split
metadata. Each evaluation trains a fresh classifier and recomputes features for
the supplied model. The default remains 224 pixels; use `size: 112` for the
released Memory Storyboard evaluation resolution.

## Adding benchmarks

Implement `Benchmark`, register it in `BENCHMARKS`, and expose its module through
`_imports_`. A constructor validates settings; an idempotent `prepare()` loads
dataset metadata; `run(model, ctx)` returns scalar metrics. The evaluation context
provides `device`, `seed`, and `derive_seed(name)`, without training clocks.
Declare `requires_capabilities` so incompatible models fail before preparation.
Override `configuration()` for provenance when users pass constructed instances,
and `metadata()` for split sizes or dataset identity. Increment `version` when
the evaluation protocol changes.

A small evaluation suitable for training hooks can implement `Probe`, which
extends `Benchmark`, or use a probe wrapper around its shared scorer. A benchmark
implementation should receive a model; checkpoint loading belongs to the runner.
