# Checkpoint evaluation

`egl eval` scores a trained checkpoint or exported model on one or more
benchmarks and writes the metrics, with the settings and provenance needed to
compare runs, to a JSON file. It rebuilds only the model, so you do not need the
training data. Install the training extra with `uv sync --extra train` first.

## Example

The shape-bias benchmark runs on CPU with a public checkpoint and stimuli that
you render locally:

```bash
uv run egl data prepare shape-bias
uv run egl eval shape_bias_eval \
  checkpoint=hf://models/egolearn/babycl-resnext50-krishnacam@21c05a86e3042f767a36c03311d594c368275d67/model.safetensors \
  "benchmarks.0.manifest=${EGOLEARN_DATA:-data}/shape-bias/stimulus_manifest.json" \
  device=cpu output_json=shape_bias_results.json
```

The first command renders 256 stimuli under `data/shape-bias`. The second
downloads the weights, scores them and writes `shape_bias_results.json`. The
[first checkpoint evaluation](evaluation/first_evaluation.md) walks through the
same run and explains the score.

Every evaluation takes a packaged config name or a YAML file, a `checkpoint`,
and Hydra overrides such as data paths. Add `--cfg job` to print the composed
config without loading anything, and see the [command-line guide](cli.md) for
the override syntax.

## Benchmark selection

Names ending in `_eval` are packaged configs. The `probe:` targets go in a
[custom config](#custom-configs).

| Benchmark | Config or target | Data | Headline metric (chance) |
| --- | --- | --- | --- |
| [ImageNet linear decoding](#imagenet-linear-decoding) | `imagenet_linear_eval` | ImageNet `train/` and `val/` class folders | `top1` (0.001) |
| [ImageNet fine-tuning](#full-network-fine-tuning) | `imagenet_full_finetune_eval` | ImageNet `train/` and `val/` class folders | `top1` (0.001) |
| Shape bias | `shape_bias_eval` | [Rendered stimuli](evaluation/data.md#shape-bias) | `shape_bias` (0.5 is no preference) |
| Spatial relations | `spatial_relations_eval` | [Monitoring subsets](evaluation/spatial_relations.md) | `<name>/accuracy` (0.5) |
| Labeled-S 4AFC | `probe:four_afc` | [Labeled-S trials](evaluation/data.md#labeled-s-4afc-data) | `four_afc_acc` (0.25) |
| Labeled-S linear decoding | `probe:labeled_s_linear` | [Labeled-S images](evaluation/data.md#labeled-s-linear-data) | `top1` (1/26) |
| Labeled-S linear, BabyCL paper split | `probe:labeled_s_stratified_linear` | [Labeled-S images](evaluation/data.md#labeled-s-linear-data) | `top1` (1/26) |
| mini-ImageNet linear decoding | `probe:mini_imagenet_linear` | 100-class `train/` and `val/` folders | `top1` (0.01) |
| BabyView 4AFC | `probe:four_afc` | [BabyView object crops](evaluation/data.md#babyview-object-crops) | `four_afc_acc` (0.25) |
| BabyView linear decoding | `probe:linear` | [BabyView object crops](evaluation/data.md#babyview-object-crops) | `top1` (1/72) |
| Konkle 4AFC | `probe:four_afc` | [Konkle object photographs](evaluation/data.md#konkle-4afc-data) | `four_afc_acc` (0.25) |
| VTWT | `probe:vtwt` | [SAYCam frames and labels](evaluation/data.md#babyvlm-benchmark-data) | `vtwt_acc` (0.5) |
| Baby Winoground | `probe:baby_winoground` | [SAYCam frames and labels](evaluation/data.md#babyvlm-benchmark-data) | `bw_group_acc` (0.25) |
| IntPhys 2 | `probe:intphys2` | [Main split videos](evaluation/intphys2.md) | `voe_acc`, `voe_auroc` (0.5) |
| GRASP Level 2 | `probe:grasp` | [Level 2 videos](evaluation/grasp.md) | `voe_acc`, `voe_auroc` (0.5) |
| EgoSchema | `egoschema_eval` | [Questions and videos](evaluation/egoschema.md) | `acc` (0.2) |
| Ego4D NLQ | `ego4d_nlq_eval` | [Annotations and clips](evaluation/ego4d_nlq.md) | `r1_iou0.3`, `r1_iou0.5` |

Accuracies are fractions between 0 and 1. The 4AFC, VTWT and Baby Winoground
benchmarks compare images with words, so they need a model with a text encoder
and its training vocabulary. EgoSchema and Ego4D NLQ score a generative model
behind an OpenAI-compatible server, such as vLLM, instead of a checkpoint.

The shape-bias and spatial-relations configs read
[public Hub datasets](evaluation/hub.md#packaged-collections). To work offline,
prepare the data locally and override the manifest path as in the example. To run many
checkpoints and seeds, use [benchmark suites](benchmark_suites.md).

## Custom configs

A config of your own runs a `probe:` benchmark, or several benchmarks in one
pass. It inherits `/evaluate`, imports the module that registers each target and
lists the benchmarks. Save this as `labeled_s_eval.yaml` with your paths:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.labeled_s_4afc
benchmarks:
  - _target_: "probe:four_afc"
    name: labeled_s_4afc
    manifest: data/labeled_s_4afc_manifest.json
    vocab_json: /path/to/training_vocab.json
    vision_head: embed512
```

```bash
uv run egl eval labeled_s_eval.yaml checkpoint=run/final.pth --cfg job
uv run egl eval labeled_s_eval.yaml checkpoint=run/final.pth \
  output_json=labeled_s_results.json
```

Each benchmark needs a unique `name`, and `vision_head` names the model's
image–text projection head (`embed512` for BabyCL). Relative paths resolve from
the directory where you run `egl eval`. The
[probe reference](reference/components/probe.md) lists each target's arguments
and its module for `_imports_`.

Training configs accept the same `probe:` entries in their `probes` list. The
`benchmark:` targets behind the ImageNet, EgoSchema and Ego4D NLQ configs run
only in standalone evaluation.

## ImageNet protocols

Both ImageNet configs read licensed images from `train/` and `val/` class
folders under `data_root`, and both default to `device=cuda`.

### ImageNet linear decoding

`imagenet_linear_eval` trains a linear classifier on frozen features with SGD
and LARC for 10 epochs and reports validation `top1`:

```bash
uv run egl eval imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  output_json=imagenet_results.json
```

Images are 224 pixels by default, and `benchmarks.0.size=112` matches the
Memory Storyboard evaluation. The classifier's own seed is `benchmarks.0.seed`,
which the top-level `seed` does not change.

### Full-network fine-tuning

`imagenet_full_finetune_eval` trains a copy of the encoder with a new
classification head and reports final-epoch `top1` and `top5`:

```bash
uv run egl eval imagenet_full_finetune_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  benchmarks.0.output_dir=runs/finetune/example output_json=finetune.json
```

The packaged recipe trains for 100 epochs with AdamW, an effective batch of 256
and standard ImageNet augmentation. It is one fixed protocol, not each model's
published recipe. With `output_dir` set, rerunning the same command resumes
from the last completed epoch, so use a new directory for each checkpoint, seed
and configuration.

## Linear decoding protocols

The linear-decoding targets fit different classifiers, so compare scores only
within one row. All of them keep the encoder frozen and report held-out `top1`.

| Targets | Features and classifier | Default view and size |
| --- | --- | --- |
| `probe:linear`, `probe:labeled_s_linear` | Fixed features standardized on the train split, L-BFGS logistic regression | `center_crop`, 224 |
| `probe:labeled_s_stratified_linear`, `probe:mini_imagenet_linear` | Fixed unit-length features, L-BFGS logistic regression with `C=0.316` | `center_crop`, 224 |
| `probe:linear_sgd` | Fresh training crops each epoch, SGD with LARC | `imagenet_center_crop`, 112 |
| `benchmark:imagenet_linear_sgd` | Full ImageNet, fresh training crops each epoch, SGD with LARC | `imagenet_center_crop`, 224 |

## Checkpoint inputs

`checkpoint` accepts a `final.pth` or `step_*.pth` file from a training run, a
directory written by `egl export`, or an `hf://models/...` reference to an
exported `model.safetensors`. These carry the config that rebuilds the model.
For an older checkpoint without it, add the training run's saved Hydra config,
such as `model_config=outputs/<date>/<time>/.hydra/config.yaml`.

Evaluation loads only the weights, never optimizer, RNG or replay state. A full
`snapshot_*.pth` also needs `trusted_snapshot=true`, which allows arbitrary
pickled code, so use it only for snapshots you trust (see
[checkpoint trust](../SECURITY.md#model-export-and-checkpoint-trust)).

## Results and run records

The output file, `evaluation.json` unless you set `output_json`, groups metrics
by benchmark name under `results`. It also records each benchmark's settings
and dataset metadata, the seed, the checkpoint identity and the code revision.

If a benchmark fails, `egl eval` saves the completed benchmarks with
`status: "partial"` and exits with an error. A setup failure, such as a missing
checkpoint, leaves an existing output file unchanged, so check the exit status
before you read it.

Each run also writes a record to `runs/eval/records/<id>/record.json`, which
[benchmark suites](benchmark_suites.md) read to build tables. Set `record_json`
to a new path to move it, or to `null` to skip it. For evaluation from Python,
see the [Python API](evaluation/api.md).
