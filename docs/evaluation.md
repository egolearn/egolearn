# Evaluating trained models

The standalone evaluator runs benchmarks on a loaded `ComposedModel` or an
EgoLearn checkpoint or exported model directory. It constructs no training stream, replay memory, training
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
training step when available; no checkpoint directory is stored. For exports,
`weights_sha256` and `metadata_sha256` identify each file, while `sha256` hashes
the concatenated hexadecimal digests (weights first), covering both files. Benchmark names
must be unique within an evaluation.

Existing small evaluations remain available with their `probe:*` configuration
targets. Pass those configs in `benchmarks` to evaluate them after training.
There is one scoring implementation for training probes and standalone use.
Linear probes record their own seed (for SGD), hyperparameters, class names,
split sizes and split SHA-256, including when passed as constructed objects.
Generic split hashes, also recorded by ImageNet, cover ordered relative paths,
labels and classes, not image bytes. Manifest-backed probes also record the
manifest's byte hash; Labeled-S records the verified canonical split hash and source.
Linear-probe path settings retain only the dataset directory or file name, so
recorded configs do not expose machine-specific parent directories.
For every benchmark, the runner reduces top-level absolute local path settings
to file or directory names. This cleanup leaves relative values and structured
metadata intact; config-driven calls retain their original top-level `hf://`
references. Benchmarks still receive the original paths for loading data.

If benchmark preparation, execution, metric conversion, or metadata collection
fails, `EvaluationError.partial_result` retains completed benchmarks and their
provenance; `benchmark` identifies the failed benchmark
and `__cause__` retains its original error. Serialized results record
`status: "partial"` and `failed_benchmark`; successful suites record
`status: "complete"`. The CLI replaces `output_json` with this partial result
before exiting unsuccessfully, including when no benchmark has completed.
Non-finite metrics (NaN or infinity) fail the benchmark before it is recorded.
Setup errors (checkpoint loading, imports, benchmark construction, or suite
validation) propagate normally and leave any existing output file unchanged.
Consumers must check the process exit status before using `output_json`; a file
from an earlier run may still exist after a setup failure. In Python:

```python
from egolearn.evaluation import EvaluationError

try:
    result = evaluate_checkpoint("run/final.pth", benchmarks=benchmarks, imports=imports)
except EvaluationError as error:
    error.partial_result.save_json("partial-evaluation.json")
    raise
```

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
uv run egl data prepare labeled-s-4afc \
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

`CheckpointHook` files (`step_*.pth`, `final.pth`) use restricted loading by
default. Training snapshots (`snapshot_*.pth`) require an explicit
`trusted_snapshot=True` opt-in. New files produced from a pipeline-built model
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
checkpoint files still deserialize their stored contents before unrelated
training state is released; this API does not change that file-format cost.
Legacy checkpoints use restricted unpickling. Full snapshots need an explicit
`trusted_snapshot=True` opt-in (Hydra: `trusted_snapshot=true`) and a trusted source.
Exported directories from `egl export` load safetensors weights without pickle;
their model configuration and contract are read from `model.json`. Evaluation
records digests covering both files and retains source provenance. See
[model export](cli.md#exporting-weights) and [checkpoint trust](../SECURITY.md#model-export-and-checkpoint-trust).

## Command line

```bash
uv run egl eval imagenet_linear_eval \
  checkpoint=run/final.pth data_root=/datasets/imagenet \
  device=cuda output_json=imagenet_results.json
```

For an older checkpoint, append
`'model_config=outputs/<date>/<time>/.hydra/config.yaml'`, substituting the training
run's date and time (or your explicit `hydra.run.dir`). For a trusted full
snapshot, also pass `trusted_snapshot=true`.
The packaged ImageNet config exposes its protocol settings under `benchmarks.0`,
for example `benchmarks.0.encoder=image` or `benchmarks.0.batch_size=256`.
`shape_bias_eval` and `spatial_relations_eval` evaluate shape bias and the spatial
monitoring subsets on hosted data, so they need only `checkpoint=` (see
[Hub artifacts](#hub-artifacts-and-access-tiers)).
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

## Hub artifacts and access tiers

`uv run egl data fetch shape-bias --revision <tag-or-commit>`
fetches `agentic-learning-ai-lab/egolearn-shape-bias` into
`$EGOLEARN_DATA/shape-bias` (default `data/shape-bias`). The download is staged,
checked against `checksums.sha256`, and installed only after verification.
`.egolearn-fetch.json` records the requested revision, resolved commit and
verification status. Missing checksums require `--allow-unverified`; corrupt
checksums always fail. `--repo` overrides the dataset repository. To choose the
storage root, use `egl data --root DIR fetch shape-bias --revision <tag-or-commit>`;
`--root` goes before `fetch`, and the collection is installed in `DIR/shape-bias`.

Data paths in component configs, including benchmark manifests, may use
`hf://<org>/<repo>@<revision>[/path/in/repo]`. Revisions are tags or commit IDs
without slashes. `egolearn.hub.resolve` is torch-free, passes local paths through,
and logs the resolved snapshot commit. Direct Python callers can use it before
constructing a dataset. Config resolution downloads to the standard HF cache;
use `fetch` when checksum verification is required. `HF_HOME` controls that cache.
Dataset references materialize the full snapshot once so manifests can safely
read sibling files. This can download data during `dry_run=true`; that option
skips training-stream iteration, not artifact resolution. Model file references
must name a file and download only that file. Filename-less model references fail
before any download. Loading an exported checkpoint also fetches its sibling
`model.json` at the weights' immutable commit. Materialized copies require additional cache space
and are not tracked by `hf cache` cleanup tools. Their path is
`<Hub cache>/datasets--<org>--<repo>/egolearn-materialized/<generation>/<commit>`;
remove the `egolearn-materialized` directory manually when no job uses its copies.
Existing complete copies can be reused from a read-only shared cache. First-time
materialization needs a writable cache; alternatively use `egl data --root DIR fetch`
to put a real-file collection on writable storage. If an offline snapshot was partial,
newly cached files produce a new materialization rather than reusing that subset.

An explicit `hf://models/<org>/<repo>@<revision>/vocab.json` lets a tokenizer or
benchmark read vocabulary from a model repo; `hf://datasets/` forces a dataset
repo. Without a prefix, component inputs default to datasets and checkpoint/model
config inputs default to models. Checkpoint evaluation records the original
reference, repo, revision and resolved commit alongside the content digest.

For example, a probe's `manifest` may be
`hf://agentic-learning-ai-lab/egolearn-shape-bias@v1/stimulus_manifest.json`.
Checkpoint and initialization inputs use model repos with the same syntax, ending
in an explicit filename such as `@v1/model.safetensors`; local exported directories
and legacy checkpoint paths still work. Keep full snapshots in local run directories.
Resume also retains support for legacy Hub snapshot references and always uses
unrestricted pickle, so only resume snapshots you trust. Model Hub publishing now
accepts only exported weights. Hub checksums establish file integrity, not trust
in legacy Python pickle contents. Model config files use model repos too.

| Dataset / artifact | Access tier | What may be hosted |
| --- | --- | --- |
| Prepared shape-bias stimuli | public | Licensed rendered stimuli, manifests and checksums |
| Spatial-relations v1 and monitoring subsets | public | Licensed stimuli, fixed trials and manifests |
| BabyView 4AFC and linear evaluation manifests | public | Split/trial metadata only, without pixels or restricted captions |
| Labeled-S split metadata | public | Split identifiers only; source frames stay restricted |
| ImageNet | upstream-licensed | Manifests only; users obtain images from the licensed upstream source |
| IntPhys 2 Main split | upstream-licensed | Nothing; users download the CC-BY-NC videos from the upstream release |
| SAYCam, Labeled-S frames, restricted BabyView and BabyVLM images | IRB | Never on the Hub, including private repos or pixel-embedding manifests |

Two collections are published, privately until the first release, under CC BY 4.0
with the authors' permission. Cite the papers when using them.

| Repo | Contents | Packaged config |
| --- | --- | --- |
| `agentic-learning-ai-lab/egolearn-shape-bias` | Prepared stimuli (Tartaglini, Vong & Lake, 2022) | `shape_bias_eval` |
| `agentic-learning-ai-lab/egolearn-spatial-relation-categorization` | `monitoring/` subsets (Davidson, Orhan & Lake, 2024) | `spatial_relations_eval` |

Both are tagged `v1`; fetch them with `egl data fetch shape-bias --revision v1` or
`egl data fetch spatial-relation-categorization --revision v1`. The other rows are
publication candidates, not a claim that collections already exist. Pinned upstream
downloads remain available.

To upload a prepared directory (not a training run directory):

```bash
uv run python -m egolearn.hub push prepared/ --name shape-bias --license CC-BY-4.0
uv run egl export runs/example/final.pth exports/babycl-saycam-seed0
uv run python -m egolearn.hub push exports/babycl-saycam-seed0 --name babycl-saycam-seed0 \
  --repo-type model --license MIT
uv run egl data verify prepared/
uv run python -m egolearn.hub checksum prepared/
```

`push` uploads a staging copy with checksums and a minimal card, leaves the source
unchanged, and creates a **private** repo. `--public` is required to create or
upload to a public repo; it does not change an existing repo's visibility.
The supported license identifiers are checked before repository creation (an
unsupported identifier reports the accepted list). Choose a license you have
authority to grant and inspect the directory before uploading. Model repos require
the validated `model.safetensors`/`model.json` pair from `egl export` and reject
other payloads, including `.pth` files. Dataset uploads include non-hidden payload
files. Do not upload IRB data, embedded
frames, credentials or personal run metadata. Checkpoint names follow
`egolearn-<method>-<data>-seed<n>`. SAYCam-trained weights stay in **private** model
repos until publication rights under the Databrary agreement have been confirmed.
Authentication uses `HF_TOKEN`; no token belongs in configs or command arguments.
