# Checkpoint evaluation

Evaluate an EgoLearn checkpoint or exported model without rebuilding its training
pipeline. Install its dependencies with `uv sync --extra train`, then choose a
compatible benchmark and obtain access to that benchmark's data.

Evaluation requires the model's construction config when the checkpoint does
not include one, but it does not require the original training stream.
Full training snapshots require the explicit trust setting described under
[checkpoint compatibility](#checkpoint-compatibility).

The base `evaluate` config defaults to `device=cpu`. The `imagenet_linear_eval`
config used below defaults to `device=cuda`. To run that example on CPU,
replace `device=cuda` with `device=cpu`.

## Command line

For ImageNet linear decoding, provide a checkpoint and an ImageFolder root with
`train/` and `val/` directories:

```bash
uv run egl eval imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  device=cuda output_json=imagenet_results.json
```

Replace the example paths with your artifacts before running the command.
The evaluator trains a fresh linear classifier on frozen image representations
and writes metrics and provenance to `imagenet_results.json`.
Progress lines can precede the JSON on standard output, so use `output_json`
for machine-readable results.

Protocol settings appear under `benchmarks.0` in the packaged ImageNet config.
Set `benchmarks.0.encoder` to an encoder name or modality; its default is
`image`. Append `benchmarks.0.batch_size=256` to reduce the batch size. The
default image resolution is 224 pixels, while `benchmarks.0.size=112` matches
the released Memory Storyboard evaluation resolution.

Use `benchmarks.0.seed` to change ImageNet's classifier initialization, shuffle
order, and augmentations. The top-level `seed` sets the evaluation context but
does not replace this separate protocol seed.

Use `egl eval --help` to inspect the default evaluation config.
`egl eval path/to/suite.yaml` accepts a custom evaluation configuration with
the search paths and overrides described in
[command-line configuration](cli.md#configuration-and-overrides).
For repeated runs with pinned inputs and several seeds, use
[benchmark suites](benchmark_suites.md).

## Benchmark selection

Choose a protocol compatible with the model and prepare its inputs before
running evaluation.

| Evaluation | Configuration or target | Inputs and protocol | Headline metric and baseline |
| --- | --- | --- | --- |
| ImageNet linear decoding | `imagenet_linear_eval` | Licensed ImageNet images in matching `train/` and `val/` class folders | `top1`; 1/1,000 uniform-guess accuracy |
| Shape bias | `shape_bias_eval` | [Prepared stimuli and scoring](#shape-bias) | `shape_bias`; 0.5 means equal shape/texture preference |
| Spatial relations | `spatial_relations_eval` | [Monitoring subsets and scoring](evaluation/spatial_relations.md) | `<name>/accuracy`; 0.5 chance |
| Labeled-S 4AFC | `probe:four_afc` | [4,150 frozen trials over 23 categories](evaluation/data.md#labeled-s-4afc-data) | `four_afc_acc`; 0.25 chance |
| Labeled-S linear decoding | `probe:labeled_s_linear` | [Canonical 26-class split](evaluation/data.md#labeled-s-linear-data) | `top1`; 1/26 uniform-guess accuracy |
| BabyView 4AFC | `probe:four_afc` | [Frozen trials on vocabulary-matched categories](#babyview-protocols) | `four_afc_acc`; 0.25 chance |
| BabyView linear decoding | `probe:linear` | [Frozen split over all 72 categories](#babyview-protocols) | `top1`; 1/72 uniform-guess accuracy |
| VTWT | `probe:vtwt` | [CSV labels, frames, and training vocabulary](evaluation/data.md#babyvlm-benchmark-data) | `vtwt_acc`; 0.5 chance |
| Baby Winoground | `probe:baby_winoground` | [CSV labels, positive/negative images, and training vocabulary](evaluation/data.md#babyvlm-benchmark-data) | `bw_group_acc`; 0.25 for independent random choices |
| IntPhys 2 | `probe:intphys2` | [Video inputs and score limitations](evaluation/intphys2.md) | `voe_acc`, `voe_auroc`; 0.5 chance |
| GRASP Level 2 | `probe:grasp` | [Pair manifest and score limitations](evaluation/grasp.md) | `voe_acc`, `voe_auroc`; 0.5 chance |

Accuracies are fractions rather than percentages. Baby Winoground requires both
image-phrase choices to be correct, so its group baseline assumes independent
random choices for the two images.

The shape-bias and spatial-relations configs supply their own Hub input paths.
If you have repository access, only `checkpoint=` is required. Private
collections require authentication through `HF_TOKEN`.

Small evaluations use `probe:*` targets both during training and in standalone
evaluation, where you put these configs in the `benchmarks` list. Full ImageNet
linear decoding uses `benchmark:imagenet_linear_sgd` and runs outside the
training loop.

## Standalone probe configurations

For a probe without a packaged evaluation config, save a YAML file that inherits
`/evaluate`, imports its registration module, and places entries under
`benchmarks`. For example, save this as `labeled_s_eval.yaml` after replacing
the input paths:

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
    view: center_crop
    size: 224
```

Select the image-text head from your model; `embed512` is BabyCL's head.
Check configuration composition without opening the checkpoint or dataset,
then run evaluation:

```bash
uv run egl eval labeled_s_eval.yaml checkpoint=run/final.pth --cfg job
uv run egl eval labeled_s_eval.yaml \
  checkpoint=run/final.pth output_json=labeled_s_results.json
```

The CLI resolves the config file relative to your current directory and searches
its directory before the installed defaults. Relative data and output paths
also resolve from the current directory unless you explicitly enable Hydra's
working-directory change.

Import the module for each configured protocol. The data guides provide
complete [BabyVLM](evaluation/data.md#babyvlm-evaluation-config) and
[IntPhys 2](evaluation/intphys2.md#registered-evaluation-probe) examples.

| Protocol | Target | Registration module |
| --- | --- | --- |
| Labeled-S 4AFC | `probe:four_afc` | `egolearn.evaluation.benchmarks.labeled_s_4afc` |
| Labeled-S linear | `probe:labeled_s_linear` | `egolearn.evaluation.benchmarks.labeled_s_linear` |
| BabyView 4AFC and linear | `probe:four_afc`, `probe:linear` | `egolearn.evaluation.benchmarks.babyview_4afc` |
| Generic fixed-feature linear | `probe:linear` | `egolearn.evaluation.linear_probe` |
| Generic SGD linear | `probe:linear_sgd` | `egolearn.evaluation.linear_sgd_probe` |
| VTWT | `probe:vtwt` | `egolearn.evaluation.benchmarks.vtwt` |
| Baby Winoground | `probe:baby_winoground` | `egolearn.evaluation.benchmarks.baby_winoground` |
| IntPhys 2 | `probe:intphys2` | `egolearn.evaluation.benchmarks.intphys2` |
| GRASP | `probe:grasp` | `egolearn.evaluation.benchmarks.grasp` |

## Linear decoding protocols

Match the split, representation, preprocessing, and classifier protocol before
comparing linear-decoding scores. The defaults differ across implementations:

| Target | Classifier input and fitting | Default view | Default size |
| --- | --- | --- | --- |
| `probe:linear`, `probe:labeled_s_linear` | Fixed features, train-split standardization, deterministic L-BFGS logistic regression | `center_crop` | 224 pixels |
| `probe:linear_sgd` | Fresh stochastic training crops each epoch, SGD/LARC | `imagenet_center_crop` for evaluation | 112 pixels |
| `benchmark:imagenet_linear_sgd` | Full ImageNet train/val splits, stochastic training crops, SGD/LARC | `imagenet_center_crop` for evaluation | 224 pixels |

All classifiers leave the encoder frozen and return held-out `top1`.
SGD protocols report final-epoch validation accuracy, plus
`train_top1_running` from predictions collected during that epoch's updates.
Use `top1` for the held-out score. Fixed-feature probes consume no randomness,
while SGD protocols use their own `seed`.

Set `view` and `size` explicitly for comparisons, along with the same `encoder`
and optional `head`. Matching those settings does not make the classifier
protocols equivalent. Labeled-S linear decoding uses 2,886 training and 2,901
evaluation images over 26 classes. It differs from BabyCL's reported 24-class
split and embedding representation.

### BabyView protocols

The BabyView evaluation uses 5,921 object crops over 72 categories.
Its 4AFC trials include only categories present in the training vocabulary,
which leaves 61 categories with the SAYCam vocabulary. Linear decoding uses
all 72 categories with a frozen stratified split and no vocabulary filter.

Prepare manifests from an extracted release containing `manifest.csv` and its
crop images, using your model's training vocabulary and a new output directory:

```bash
uv run egl data prepare babyview-4afc \
  --data_root /path/to/babyview_crops --vocab_json /path/to/training_vocab.json \
  --out_dir data/babyview
```

Append `--download` to fetch and extract the public crops release first.
The preparer writes `babyview_4afc_trials.json`, `babyview_linear.json`, and a
provenance report. Use `view: center_crop` and `size: 224` for both protocols.
The released crops have varying aspect ratios, so changing preprocessing
changes the evaluation.

With those manifests prepared, save this as `babyview_eval.yaml` and replace
the input paths and image-text head:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.babyview_4afc
benchmarks:
  - _target_: "probe:four_afc"
    name: babyview_4afc
    manifest: /path/to/babyview_4afc_trials.json
    vocab_json: /path/to/training_vocab.json
    vision_head: embed512
    view: center_crop
    size: 224
  - _target_: "probe:linear"
    name: babyview_linear
    manifest: /path/to/babyview_linear.json
    encoder: image
    head: null
    view: center_crop
    size: 224
```

Run `uv run egl eval babyview_eval.yaml checkpoint=run/final.pth` with your
checkpoint path. The 4AFC benchmark requires image and text representations;
the linear benchmark only requires image representations.

## Shape bias

Shape bias measures a representation's preference for matching shape over
matching texture. The probe encodes 256 prepared stimuli and compares all
57,600 valid anchor/shape-match/texture-match triplets.

`shape_bias` is the fraction whose shape match has strictly higher cosine
similarity. Ties count as texture choices. A value of 0.5 indicates equal
preference; this is not an object-classification accuracy.

| Metric | Meaning |
| --- | --- |
| `shape_bias` | Shape-choice fraction |
| `shape_bias_texture` | Texture-choice fraction, including ties |
| `shape_bias_margin` | Mean shape-minus-texture cosine margin |
| `shape_bias_ties` | Number of equal-similarity triplets |
| `shape_bias_n` | Number of triplets |

See [Hub collections](evaluation/hub.md#packaged-collections) for the stimuli,
license, citation, and use of a verified local copy.

## Checkpoint compatibility

Use `step_*.pth`, `final.pth`, or an exported model directory as the checkpoint
input. Pipeline-built checkpoints include the resolved model construction
config needed to rebuild the model.
Exports from `egl export` contain `model.safetensors` and `model.json`, which
jointly identify the weights and construction metadata.

For a checkpoint without construction metadata, append the training run's saved
config path. Hydra writes it under
`outputs/<date>/<time>/.hydra/config.yaml` by default, or under your explicit
`hydra.run.dir`:

```bash
uv run egl eval imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  model_config=outputs/example/.hydra/config.yaml \
  device=cuda output_json=imagenet_results.json
```

Use a composed training config or an explicit model mapping in Python,
because a YAML file containing unresolved Hydra defaults is insufficient.
Only model-related nodes are resolved, so training-data environment variables
are unnecessary unless a model setting depends on them. An older
`vocab_size: auto` setting still needs its tokenizer config and vocabulary file,
or an explicit vocabulary size.

The loader checks the model contract when present and loads weights strictly.
It disables pretrained initialization and does not restore training RNGs,
optimizer state, or stream replay. Legacy checkpoint files still deserialize
their stored contents before unrelated training state is released.

Ordinary checkpoints use restricted unpickling, while exported safetensors
weights load without pickle. For a full `snapshot_*.pth` from a trusted source,
append `trusted_snapshot=true` to permit unrestricted unpickling. Evaluation
uses the model weights without restoring the snapshot's training state.
See [model export](cli.md#exporting-weights) and
[checkpoint trust](../SECURITY.md#model-export-and-checkpoint-trust).

## Results and failures

A successful result has `status: "complete"` and groups metrics by benchmark
name, which must be unique within an evaluation. It records benchmark settings,
versions, dataset metadata, seeds, model config, and available source and
checkpoint provenance.

If a benchmark fails during preparation or execution, the CLI writes completed
benchmarks to `output_json` and exits unsuccessfully. The result has
`status: "partial"` and names the failed benchmark. Non-finite metrics fail the
benchmark before its results are recorded.

Check the process exit status before using the output file. Setup failures,
such as a missing checkpoint or invalid benchmark config, leave an existing
`output_json` unchanged, so that file can belong to an earlier run.
The [API reference](evaluation/api.md#evaluationerror) describes partial results
and Python error handling.

## Run records

Evaluation also writes an atomic `record.json` under
`runs/eval/records/<id>/`, relative to the process's working directory.
The record is created before loading the checkpoint. Each attempt gets a
separate record containing run
provenance and benchmark settings, including completed benchmarks before a
later failure.

Set `record_json=/path/to/new-record.json` for a specific destination, which
must not already exist, or set `record_json=null` to disable recording.
An unwritable automatic destination aborts evaluation before checkpoint loading.
Choose a writable explicit path or disable recording with `record_json=null`.
See
[run records](benchmark_suites.md#run-records-and-scope) for input identity and
training provenance.

## Python API

Use [`evaluate_checkpoint()`](evaluation/api.md#evaluate_checkpoint) to load and
evaluate a checkpoint, or [`evaluate()`](evaluation/api.md#evaluate) for a model
already in memory. The [API reference](evaluation/api.md) covers device handling,
model-state preservation, result metadata, and failures.

## SAYCam access

SAYCam and related evaluation images require approved data access. See
[SAYCam access](evaluation/data.md#saycam-access) before preparing Labeled-S,
VTWT, or Baby Winoground inputs.

## Labeled-S 4AFC data

Follow [Labeled-S preparation](evaluation/data.md#labeled-s-4afc-data) to convert
the frozen BabyCL trials using your model's training vocabulary.

## BabyVLM benchmark data

Follow [BabyVLM preparation](evaluation/data.md#babyvlm-benchmark-data) for the
VTWT and Baby Winoground images, CSV schemas, and normalization settings.

## ImageNet migration

Use `benchmark:imagenet_linear_sgd` under `benchmarks` in a standalone evaluation
config and remove `probe:imagenet_linear_sgd` entries from training configs.
Training pipelines reject standalone benchmarks in their `probes` list.

The protocol uses full ImageFolder train/val splits, a frozen encoder,
stochastic training crops, SGD/LARC, and cosine learning-rate decay. Dataset
enumeration occurs during preparation. Reusing a benchmark instance reuses
split metadata, while each evaluation trains a fresh classifier and recomputes
representations for the supplied model.

## Adding benchmarks

See [benchmark development](evaluation/extending.md) for the implementation
contract, registration, capability checks, and protocol versioning.

## Hub artifacts and access tiers

See [Hub artifacts](evaluation/hub.md) for verified downloads, `hf://` input
paths, cache management, access tiers, and publication rules.
