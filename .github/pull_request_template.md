## Summary

<!-- What does this PR do, and why? -->

## Test plan

<!-- How was this verified? e.g. `uv run pytest`, new tests, manual run -->

## Checklist

- [ ] No secrets or API keys (LLM keys are env vars only), no cluster-specific personal paths
- [ ] No binaries or files >500 KB (checkpoints/datasets belong on the Hub, see docs/roadmap.md)
- [ ] `uv.lock` regenerated if dependencies changed
- [ ] CHANGELOG.md updated under [Unreleased] for user-visible changes
- [ ] Ported third-party-derived code keeps its attribution header (see docs/design/streaming.md)
