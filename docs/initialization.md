# Initialize from another run

Set `init_from` to copy model weights into a fresh run. For example, keep a trained
vision encoder while initializing new heads from the new run's seed:

```bash
uv run python -m egolearn.train --config-name toy_streaming \
  '+init_from.checkpoint=runs/previous/final.pth' \
  '+init_from.include=[encoders.vision.]'
```

Checkpoints are accepted by default. Full snapshots require
`+init_from.trusted_snapshot=true`. See the [trust rule](../SECURITY.md#checkpoint-trust). `include` and `exclude` select tensor
name prefixes; omitting `include` selects all tensors. Strict loading is the default:
selected components must be complete and shapes must match. Explicitly setting
`strict: false` skips unmatched tensors, but selecting no tensors always fails.
Parameters and buffers are copied; optimizer, stream, and RNG state start fresh.
Use `resume` to continue a complete run instead. `init_from` and `resume` are mutually
exclusive; set `init_from=null` when resuming a config that specifies initialization.
The source checkpoint file name and hash, selected prefixes, and tensor count are recorded in the
pipeline summary and subsequent checkpoints' model metadata.

`init_from.checkpoint` also accepts a model Hub reference such as
`hf://org/model@v1/final.pth`. Only the named file is downloaded. The source
reference, repository, requested revision, and resolved commit are retained in
the initialization metadata, including after snapshot resume.

V-JEPA initializes its EMA target from the context encoder at step 0, replacing any
loaded target weights. A resumed snapshot preserves its target state.
