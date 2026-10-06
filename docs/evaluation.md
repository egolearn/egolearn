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

## SAYCam access

SAYCam videos, SAYCam-S frames, and Labeled-S images are hosted on
[Databrary](https://nyu.databrary.org/volume/564/). Researchers must obtain access
with approval from their institution's IRB and follow the applicable Databrary
data-use terms. See the [CVCL dataset notes][cvcl-data] and
[BabyVLM preparation notes][babyvlm-preparation] for the source releases.

| Benchmark | Image source | Labels and code terms | Preparation |
| --- | --- | --- | --- |
| Labeled-S 4AFC | Labeled-S release on Databrary | Frozen BabyCL trials; [CVCL vocabulary][cvcl-vocab] (MIT) | [Manifest and vocabulary](#labeled-s-4afc-data) |
| VTWT | Multimodal SAYCam-S frames on Databrary | [VTWT labels][vtwt-labels]; [Apache-2.0][babyvlm-license] task code | [Frames and CSV inputs](#babyvlm-benchmark-data) |
| Baby Winoground | SAYCam-S frames and `BabyVLM_V1` synthetic images on Databrary | [Baby Winoground labels][bw-labels]; [Apache-2.0][babyvlm-license] task code | [Frames, negatives, and CSV inputs](#babyvlm-benchmark-data) |

[cvcl-data]: https://github.com/wkvong/multimodal-baby#datasets
[babyvlm-preparation]: https://github.com/ShawnKing98/babylmms-eval#data-preparation
[babyvlm-license]: https://github.com/ShawnKing98/babylmms-eval/blob/main/LICENSE

## Labeled-S 4AFC data

The Labeled-S configuration of `probe:four_afc` uses BabyCL's frozen 4,150 trials
over 23 categories. Obtain the Labeled-S evaluation images through
[SAYCam access](#saycam-access), retaining
their `<category>/<filename>` layout and released 224×224 crops. These are
separate from the SAYCam-S training frames used by VTWT below.

Use the original `eval_object_categories_4150.json` from the BabyCL evaluation
assets. The [BabyCL paper](https://arxiv.org/html/2606.05115v1) describes its code
release as supplementary material; if you do not have the frozen metadata,
request it from the authors. The input is a JSON object with a `data` list; each
row has `target_category`, `target_img_filename`, and three `foil_img_filenames`.
Converting this file preserves trial membership, foil order, and the target at
index 0. Generating random replacement trials would give a different evaluation.

For `vocab_json`, use the exact word-to-integer mapping used to train the model.
BabyCL's CVCL vocabulary is available as
[`multimodal/vocab.json`][cvcl-vocab] in the MIT-licensed CVCL repository (pinned
revision `d64c1bb164ed4c25f0cad8a1c3b7a4d68efae326`). Save that file unchanged if
your model was trained with it. A vocabulary built from the 23 evaluation labels
would assign different embedding indices. For a model trained with a custom
vocabulary, retain its training vocabulary and check that all target labels are
present; the checkpoint's embedding weights alone do not recover word names.

Export the paths to the category-folder root as `LABELED_S_ROOT`, the original
trial JSON as `LABELED_S_TRIALS`, and the model's vocabulary as `VOCAB_JSON`.
The converter checks the frozen trial/category counts, target vocabulary coverage,
and local image paths, and refuses to overwrite an existing output:

```bash
uv run python -m egolearn.evaluation.benchmarks.labeled_s_4afc.prepare \
  --data_root "$LABELED_S_ROOT" --trials_json "$LABELED_S_TRIALS" \
  --vocab_json "$VOCAB_JSON" --output data/labeled_s_4afc_manifest.json
```

Pass the result as `manifest`, the same vocabulary as `vocab_json`, and the
model's image-text projection head as `vision_head` (BabyCL: `embed512`); see
the [probe configuration](../src/egolearn/evaluation/benchmarks/labeled_s_4afc/__init__.py).
The manifest contains absolute local paths; regenerate it if the image root
moves. Keep it and the restricted images outside version control.

[cvcl-vocab]: https://github.com/wkvong/multimodal-baby/blob/d64c1bb164ed4c25f0cad8a1c3b7a4d68efae326/multimodal/vocab.json

## BabyVLM benchmark data

The Visual Two-Word Test (VTWT, `probe:vtwt`) and Baby Winoground
(`probe:baby_winoground`) come from Wang et al., *BabyVLM: Data-Efficient
Pretraining of VLMs Inspired by Infant Learning* (2025). See the
[paper website](https://shawnking98.github.io/BabyVLM/) and the authors'
[evaluation repository](https://github.com/ShawnKing98/babylmms-eval).

After obtaining [SAYCam access](#saycam-access), follow the
[BabyVLM preparation notes][babyvlm-preparation]:

1. Download `frames.txt` from the
   [Multimodal SAYCam-S release](https://nyu.databrary.org/volume/564/session/67447).
2. Rename it to `frames.zip` and extract it into a local image directory. The
   release contains 600,285 images (about 14 GB).
3. For Baby Winoground, also download the `BabyVLM_V1` folder from
   [Databrary][babyvlm-images] for the synthetic negative images.

The public repository supplies labels separately from these images. Its
[license][babyvlm-license] assigns Apache-2.0 to the added multimodal tasks;
this does not replace the image releases' access terms.

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
uv run egl eval imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  device=cuda output_json=imagenet_results.json
```

For an older checkpoint, append `model_config=run/.hydra/config.yaml`.
The packaged ImageNet config exposes its protocol settings under `benchmarks.0`,
for example `benchmarks.0.encoder=vision` or `benchmarks.0.batch_size=256`.
`egl eval --help` lists the default evaluation configuration. Custom suites
can use `egl eval path/to/suite.yaml`. See the [CLI guide](cli.md) for overrides
and configuration search paths.

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
