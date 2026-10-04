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
        "encoder": "vision",
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
package version, and resolved model config. Checkpoint evaluations also record
the checkpoint path, SHA-256, and training step when available. Benchmark names
must be unique within an evaluation.

Existing small evaluations remain available with their `probe:*` configuration
targets. Pass those configs in `benchmarks` to evaluate them after training.
There is one scoring implementation for training probes and standalone use.

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
(`snapshot_*.pth`) are supported. New files produced from a pipeline-built model
include a versioned construction config: encoders, heads, predictors, frozen
roles, registration imports, and resolved values such as vocabulary size.

For older checkpoints, or checkpoints of manually assembled models, pass
`model_config="run/.hydra/config.yaml"`. An explicit model-section mapping also
works. Use a resolved config, not an uncomposed YAML containing Hydra defaults.
Only model-related nodes are resolved: training dataset environment variables
and probe data paths are not needed. An old `vocab_size: auto` setting still
needs the original tokenizer config/file or an explicit vocabulary size.

Weights load strictly, and the export contract is checked when present.
Pretrained encoder initialization is disabled while restoring a checkpoint.
Training RNGs, optimizer state, and stream replay are never restored. Existing
checkpoint files still deserialize their stored contents before unrelated
training state is released; this API does not change that file-format cost.
As with training resume, load checkpoints only from trusted sources.

## Command line

```bash
uv run python -m egolearn.evaluate --config-name imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  device=cuda output_json=imagenet_results.json
```

For an older checkpoint, append `model_config=run/.hydra/config.yaml`.
The packaged ImageNet config exposes its protocol settings under `benchmarks.0`,
for example `benchmarks.0.encoder=vision` or `benchmarks.0.batch_size=256`.
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

## Hub artifacts and access tiers

`uv run python -m egolearn.hub fetch shape-bias --revision <tag-or-commit>`
fetches `agentic-learning-ai-lab/egolearn-shape-bias` into
`$EGOLEARN_DATA/shape-bias` (default `data/shape-bias`). The download is staged,
checked against `checksums.sha256`, and installed only after verification.
`.egolearn-fetch.json` records the requested revision, resolved commit and
verification status. Missing checksums require `--allow-unverified`; corrupt
checksums always fail. `--repo` and `--dest` override the defaults.

Data paths in component configs, including benchmark manifests, may use
`hf://<org>/<repo>@<revision>[/path/in/repo]`. Revisions are tags or commit IDs
without slashes. `egolearn.hub.resolve` is torch-free, passes local paths through,
and logs the resolved snapshot commit. Direct Python callers can use it before
constructing a dataset. Config resolution downloads to the standard HF cache;
use `fetch` when checksum verification is required. `HF_HOME` controls that cache.

For example, a probe's `manifest` may be
`hf://agentic-learning-ai-lab/egolearn-shape-bias@v1/manifest.json`.
Checkpoint and resume inputs use model repos with the same syntax, ending in an
explicit filename such as `@v1/final.pth`; local checkpoint paths still work.
Only load trusted checkpoints: Hub checksums establish file integrity, not trust
in Python pickle contents. Model config files use model repos too.

| Dataset / artifact | Access tier | What may be hosted |
| --- | --- | --- |
| Prepared shape-bias stimuli | public | Licensed rendered stimuli, manifests and checksums |
| Spatial-relations v1 and monitoring subsets | public | Licensed stimuli, fixed trials and manifests |
| BabyView 4AFC and linear evaluation manifests | public | Split/trial metadata only, without pixels or restricted captions |
| Labeled-S split metadata | public | Split identifiers only; source frames stay restricted |
| ImageNet | upstream-licensed | Manifests only; users obtain images from the licensed upstream source |
| SAYCam, Labeled-S frames, restricted BabyView and BabyVLM images | IRB | Never on the Hub, including private repos or pixel-embedding manifests |

These are publication candidates, not a claim that collections already exist.
Pinned upstream downloads remain available. A Hub Collection can group approved
repos after their owners publish them; this change publishes no data or weights.

To upload a prepared directory (not a training run directory):

```bash
uv run python -m egolearn.hub push prepared/ --name shape-bias --license CC-BY-4.0
uv run python -m egolearn.hub push weights/ --name babycl-saycam-seed0 \
  --repo-type model --license MIT
uv run python -m egolearn.hub verify prepared/
uv run python -m egolearn.hub checksum prepared/
```

`push` uploads a staging copy with checksums and a minimal card, leaves the source
unchanged, and creates a **private** repo. `--public` is required to create or
upload to a public repo; it does not change an existing repo's visibility.
Choose a license you have authority to grant and inspect the directory before
uploading: non-hidden payload files are included. Do not upload IRB data, embedded
frames, credentials or personal run metadata. Checkpoint names follow
`egolearn-<method>-<data>-seed<n>`. SAYCam-trained weights stay in **private** model
repos until publication rights under the Databrary agreement have been confirmed.
Authentication uses `HF_TOKEN`; no token belongs in configs or command arguments.
