# Initialization from a checkpoint

`init_from` copies model weights from an earlier run into a new one. Use it
when you want to start from a trained encoder while the optimizer, the stream
and any weights you do not copy start fresh. If you want to continue an
interrupted run instead, use `resume`, as described in
[cluster jobs](cluster.md#resuming-a-run).

## Example

This command starts a new toy run from the vision encoder exported by the
[toy model export](cli.md#toy-model-export) example:

```bash
uv run egl train toy_streaming run_dir=runs/toy-init \
  +init_from.checkpoint=exports/toy '+init_from.include=[encoders.vision.]'
```

The pipeline summary printed at start-up has an `init_from` line with the
number of tensors copied and the source file's step and hash. Add
`dry_run=true` to check a selection without training.

## Options

| Key | Meaning |
| --- | --- |
| `init_from.checkpoint` | An export directory, a training checkpoint, a full snapshot, or a Hub file such as `hf://models/org/repo@revision/model.safetensors` |
| `init_from.include` | Tensor name prefixes to copy. Default: every tensor |
| `init_from.exclude` | Tensor name prefixes to skip |
| `init_from.strict` | Default `true`: every selected tensor must exist in both models with the same shape. `false` skips mismatches |
| `init_from.trusted_snapshot` | Set to `true` to load a full snapshot, which is unpickled without restrictions. Use it only for files you trust |

## Caveats

- `init_from` and `resume` cannot be combined. If you resume a run whose
  config sets `init_from`, add `init_from=null`.
- The run fails if your selection copies no tensors.
- V-JEPA 2 copies its context encoder into its target encoder at step 0, so
  any loaded target-encoder weights are replaced.
