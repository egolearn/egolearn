# Initialize from another run

Set `init_from` to copy model weights into a fresh run. For example, keep a trained
vision encoder while initializing new heads from the new run's seed:

```bash
uv run python -m egolearn.train --config-name toy_streaming \
  '+init_from.checkpoint=runs/previous/final.pth' \
  '+init_from.include=[encoders.vision.]'
```

Checkpoints and full snapshots are accepted. `include` and `exclude` select tensor
name prefixes; omitting `include` selects all tensors. Strict loading is the default:
selected components must be complete and shapes must match. Explicitly setting
`strict: false` skips unmatched tensors, but selecting no tensors always fails.
Parameters and buffers are copied; optimizer, stream, and RNG state start fresh.
Use `resume` to continue a complete run instead. `init_from` and `resume` are mutually
exclusive; set `init_from=null` when resuming a config that specifies initialization.
Only load trusted checkpoint files.
The source checkpoint hash, selected prefixes, and tensor count are recorded in the
pipeline summary and subsequent checkpoints' model metadata.


V-JEPA initializes its EMA target from the context encoder at step 0, replacing any
loaded target weights. A resumed snapshot preserves its target state.
