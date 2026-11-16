<!-- Follow CONTRIBUTING.md, "The PR message"; remove comments and empty optional sections, then run scripts/check_pr_message.py on the draft. -->

## Summary

<!-- A short paragraph on the problem and outcome, one bullet per change, then Closes #N or Part of #N when applicable; describe the final diff. -->

## Decisions

<!-- Optional: one bullet per decision a reviewer may weigh differently, with its reason; delete this section otherwise. -->

## Results

<!-- For experiment/config/result changes: config, seeds, hardware, wall time, a table against the baseline, run/artifact links, and a reproduction command; if not run, say what is pending. Delete otherwise. -->

## Test plan

<!-- Commands that ran, where, and the outcome; state pending checks, or say if CI covers everything. No cluster job IDs. -->

## Checklist

- [ ] No secrets or API keys (LLM keys are env vars only), no cluster-specific personal paths
- [ ] No binaries or files >500 KB (follow the access tiers in docs/evaluation/hub.md before publishing artifacts)
- [ ] `uv.lock` regenerated if dependencies changed
- [ ] CHANGELOG.md updated under [Unreleased] for user-visible changes
- [ ] Ported third-party-derived code keeps its attribution header (see CONTRIBUTING.md#licensing-and-attribution)
