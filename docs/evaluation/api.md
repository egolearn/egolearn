# Evaluation API

The evaluation API runs benchmarks on a checkpoint or a loaded `ComposedModel`.
It returns metrics and provenance without constructing a training stream,
replay memory, training optimizer, or driver.

| API | Behavior | Typical use |
| --- | --- | --- |
| [`evaluate_checkpoint()`](#evaluate_checkpoint) | Load a model and run benchmarks | Evaluate saved weights |
| [`load_model()`](#load_model) | Reconstruct a model in evaluation mode | Reuse one loaded model |
| [`evaluate()`](#evaluate) | Run benchmarks on a loaded model | Evaluate an in-memory model |
| [`EvaluationResult`](#evaluationresult) | Store metrics and provenance | Save or inspect results |
| [`EvaluationError`](#evaluationerror) | Retain results after a benchmark failure | Handle partial evaluations |
| [`record_json`](#record_json) | Control automatic run records | Choose a record destination |

## `evaluate_checkpoint`

`evaluate_checkpoint(checkpoint, *, benchmarks, model_config=None, device="cpu",
seed=0, imports=(), record_json="auto", trusted_snapshot=False)` loads a model
and returns an `EvaluationResult`.

```python
from egolearn.evaluation import evaluate_checkpoint

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
```

The example requires a compatible checkpoint and licensed ImageNet images at
the supplied paths.

| Argument | Default | Behavior |
| --- | --- | --- |
| `checkpoint` | Required | Local checkpoint, exported directory, or explicit Hub model-file reference; `None` builds the model from `model_config` alone, for models without EgoLearn-trained weights such as a client of a served VLM |
| `benchmarks` | Required | Nonempty sequence of `Benchmark` instances or configuration mappings |
| `model_config` | `None` | Construction mapping or config file; required when absent from the checkpoint |
| `device` | `"cpu"` | PyTorch device on which to load and evaluate the model |
| `seed` | `0` | Integer evaluation seed |
| `imports` | `()` | Modules imported to register model and benchmark components |
| [`record_json`](#record_json) | `"auto"` | Run-record destination, or `None` to disable recording |
| `trusted_snapshot` | `False` | Boolean opt-in to unrestricted loading of a trusted full snapshot |

Benchmark names must be unique, and the model must supply their declared
capabilities. An incompatible suite fails before benchmark preparation.
Each benchmark starts with the evaluation seed, independently of suite order.
Benchmarks can also define protocol-specific seeds. In particular, linear SGD
uses its own configured seed for classifier initialization, shuffling, and
augmentations. In a config-driven ImageNet evaluation, set `benchmarks.0.seed`
to change those draws; the top-level `seed` does not override it.

Checkpoint files load with restricted unpickling unless `trusted_snapshot=True`.
Safetensors exports load without pickle. Construction and strict weight-loading
requirements are described under
[checkpoint compatibility](../evaluation.md#checkpoint-compatibility).

## `load_model`

`load_model(checkpoint, *, model_config=None, device="cpu", imports=(),
trusted_snapshot=False)` returns a reconstructed `ComposedModel` in evaluation
mode.

Its arguments have the same meanings as
[`evaluate_checkpoint()`](#evaluate_checkpoint). The model moves to `device`,
and loading preserves the caller's RNG state. This function does not run
benchmarks or create an evaluation record.

```python
from egolearn.evaluation import load_model

model = load_model("run/final.pth", device="cuda")
```

## `evaluate`

`evaluate(model, *, benchmarks, device=None, seed=0, imports=(),
record_json="auto")` returns an `EvaluationResult` for an existing model.

```python
from egolearn.evaluation import evaluate

# Reuse the model, benchmarks, and imports from the examples above.
result = evaluate(model, benchmarks=benchmarks, imports=imports, seed=0)
```

Arguments shared with [`evaluate_checkpoint()`](#evaluate_checkpoint) behave
the same way. With `device=None`, the runner infers the device from the model.
An explicit device must match every model parameter and buffer. Move the model
before evaluation if a different device is needed.

The runner temporarily sets evaluation mode and disables parameter gradients.
It restores module training modes, gradient flags, and Python, NumPy, and
PyTorch RNG states, including after a failure. Benchmarks must leave model
weights and buffers unchanged. A linear benchmark can enable gradients for a
new decoder while the pretrained model remains frozen.

## `EvaluationResult`

`EvaluationResult` contains metrics grouped by benchmark name and the metadata
needed to interpret them. `to_dict()` returns a serializable mapping.
`save_json(path)` creates parent directories and writes JSON, replacing an
existing file.

| Field | Contents |
| --- | --- |
| `results` | Metric names and finite floating-point values, grouped by benchmark |
| `configs` | Benchmark constructor settings, including defaults for config-driven calls |
| `versions` | Protocol version for each completed benchmark |
| `metadata` | Dataset and protocol metadata supplied by each completed benchmark |
| `seed`, `device` | Evaluation seed and device |
| `checkpoint` | Artifact identity for checkpoint evaluation; `None` for an in-memory model |
| `model_config` | Available resolved model construction config |
| `package_version`, `code_revision` | Package version and source revision/dirty state |
| `schema_version` | Result schema version, currently `2` |
| `status`, `failed_benchmark` | Completion status and failed benchmark name, if any |

### Artifact identity

Checkpoint identity includes the file name, SHA-256, and training step when
available. It excludes the local parent directory. Exports additionally record
`weights_sha256` and `metadata_sha256`. Their combined `sha256` hashes the
concatenated hexadecimal digests, weights first.

Hub checkpoints retain their original reference, repository, requested revision,
and resolved commit. Available training provenance is retained separately from
the evaluation run's provenance. Source revision is `"unavailable"` when it
cannot be determined from the source checkout.

### Benchmark metadata

Linear probes record hyperparameters, class names, split sizes, and split hashes.
SGD probes also record their decoder seed. Generic split hashes cover ordered
relative paths, labels, and classes; they do not hash image bytes.
Manifest-backed linear probes additionally record the manifest's byte hash.
Labeled-S linear decoding records its verified canonical split hash and source.

Top-level absolute paths in recorded benchmark settings become file or directory
names. Relative paths and structured metadata remain intact. Config-driven
calls retain their original top-level `hf://` references, while benchmarks
receive the resolved paths needed to load data. These result fields differ from
the full local configuration retained in [run records](#record_json).

## `EvaluationError`

`EvaluationError` reports a failure during benchmark preparation, execution,
metric conversion, or metadata collection. Its `benchmark` attribute names
the failed benchmark. `partial_result` retains completed benchmarks and their
provenance, and `__cause__` retains the original exception.

```python
from egolearn.evaluation import EvaluationError, evaluate_checkpoint

try:
    result = evaluate_checkpoint("run/final.pth", benchmarks=benchmarks, imports=imports)
except EvaluationError as error:
    error.partial_result.save_json("partial-evaluation.json")
    raise
```

Partial results have `status="partial"` and `failed_benchmark` set. This also
applies when the first benchmark fails and no metrics have completed.
NaN and infinite metrics fail before the benchmark is recorded.

Checkpoint loading, imports, benchmark construction, and suite validation errors
propagate directly. They do not produce an `EvaluationError`. The CLI writes
partial results on benchmark failure, but leaves `output_json` unchanged on
setup failure. Check its exit status before accepting a result file.

## `record_json`

`evaluate()` and `evaluate_checkpoint()` default to `record_json="auto"`.
Each call writes a separate atomic record at
`runs/eval/records/<id>/record.json`, relative to the process's working directory.
The record is created before checkpoint loading or benchmark execution.
CLI evaluation and its underlying API call share one record.

An explicit path must name a new file, and its destination must be writable.
An unwritable automatic destination aborts the evaluation before loading its
checkpoint, so choose a writable explicit destination in that case.
Use `record_json=None` in Python or `record_json=null` on the CLI to disable
recording.

Records include the local resolved config, input provenance, and completed
benchmark metadata. A later benchmark failure retains earlier metrics in a
record marked `failed`. It cannot enter a normal benchmark-suite table.
See [run records](../benchmark_suites.md#run-records-and-scope) for input hashes,
checkpoint provenance, and interrupted attempts.
