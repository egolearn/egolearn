# Python evaluation API

`egolearn.evaluation` runs the same evaluations as `egl eval` from Python. Use
it in your own scripts, or to score a model that is already in memory. The
[evaluation API reference](../reference/api/evaluation.md) documents every
argument and field.

## Saved checkpoints

`evaluate_checkpoint()` loads a checkpoint, runs a list of benchmarks and
returns an `EvaluationResult`:

```python
from egolearn.evaluation import evaluate_checkpoint

benchmarks = [
    {
        "_target_": "benchmark:imagenet_linear_sgd",
        "name": "imagenet",
        "data_root": "/datasets/imagenet",
    }
]
imports = ["egolearn.evaluation.benchmarks.imagenet_linear"]

result = evaluate_checkpoint(
    "run/final.pth", benchmarks=benchmarks, imports=imports, device="cuda"
)
print(result.results["imagenet"]["top1"])
result.save_json("evaluation.json")
```

Each benchmark is a mapping in the same form as an entry under `benchmarks` in a
[custom config](../evaluation.md#custom-configs), or a `Benchmark` instance.
`imports` names the modules that register the targets, like `_imports_` in a
config. The checkpoint can be any input that `egl eval` accepts, and
`model_config` and `trusted_snapshot` work as described under
[checkpoint inputs](../evaluation.md#checkpoint-inputs).

## Models in memory

`load_model()` rebuilds a checkpoint's model in evaluation mode, and
`evaluate()` scores a model you already have:

```python
from egolearn.evaluation import evaluate, load_model

model = load_model("run/final.pth", device="cuda")
result = evaluate(model, benchmarks=benchmarks, imports=imports)
```

`evaluate()` runs on the model's device unless you pass `device`. Benchmarks do
not change the weights, and the call restores the model's training mode,
gradient flags and random number generator states afterwards, even when a
benchmark fails.

## Results and failures

`result.results` maps each benchmark name to its metrics. The result also holds
each benchmark's settings, protocol version and dataset metadata, along with the
seed, device, checkpoint identity and code revision. `save_json()` writes the
same JSON file as `egl eval`.

If a benchmark fails, the call raises `EvaluationError`. Its `partial_result`
holds the benchmarks that completed, with `status` set to `"partial"`:

```python
from egolearn.evaluation import EvaluationError, evaluate_checkpoint

try:
    result = evaluate_checkpoint("run/final.pth", benchmarks=benchmarks, imports=imports)
except EvaluationError as error:
    error.partial_result.save_json("partial-evaluation.json")
    raise
```

Errors in checkpoint loading, imports or benchmark construction happen before
any benchmark runs and raise their own exception types.

Both `evaluate_checkpoint()` and `evaluate()` write a run record to
`runs/eval/records/<id>/record.json` under the current directory. Pass
`record_json` with a new file path to choose the location, or
`record_json=None` to skip the record.
