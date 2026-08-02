# Contributing

## Setup

```bash
uv sync
uv run pre-commit install
```

## Branching and merging

- `main` is protected: all changes land via PR with green CI.
- Branch off `main`: `feat/<topic>`, `fix/<topic>`, or `proj/<name>` for paper projects.
- **Merge commits only — squash and rebase merges are disabled at the repo level.**
  Commit hashes must stay shared with the private `-dev` mirror after the public flip
  (see RELEASING.md); rewriting history breaks that permanently.
- Keep long-lived branches fresh with `git merge main`, never rebase.
- Never force-push a shared branch.

## Pull requests

- All five CI checks must pass: `lint`, `types`, `test`, `lock`, `gitleaks`.
- Every PR needs an approving review from a code owner (`.github/CODEOWNERS`),
  who are auto-requested. Your own approval doesn't count, even as a code owner.
- Fill in the PR template; update `CHANGELOG.md` under `[Unreleased]` for
  user-visible changes.

## Porting policy

Code from the source repos (memory-storyboard, lifelong-memory, poodle,
midway-network, and unreleased lab codebases) is ported **only after triage**: a
concrete need, agreed scope, and a code-owner-reviewed PR. Ports must arrive
de-duplicated, seeded, tested, and free of hardcoded paths — see the porting rules
in `docs/design/streaming.md`, including the licensing rules for third-party-derived
files.

## Style

- `ruff` is the single source of truth for lint + formatting (line length 100).
  `uv run ruff check --fix . && uv run ruff format .` before pushing.
- Type hints are encouraged, not required; `uv run mypy` must pass (lenient config).
- Imports are always absolute (`from egolearn...`). Code must work from an
  installed wheel — no repo-root-relative paths, no `sys.path` hacks.
- Dataset roots come from config/env — never hardcoded module constants.

## Tests

| Tier | Marker | Where it runs | Command |
| --- | --- | --- | --- |
| unit | *(none)* | CI, every PR | `uv run pytest` |
| video | `@pytest.mark.video` | CI with tiny synthetic clips; self-skips without the extra | included in `uv run pytest` |
| slow | `@pytest.mark.slow` | nightly / manual | `uv run pytest -m slow` |
| gpu | `@pytest.mark.gpu` | cluster only, never CI | `uv run pytest -m gpu` |
| llm | `@pytest.mark.llm` | manual only (paid APIs), never CI | `uv run pytest -m llm` |

Every ported or new library component lands with tests in the appropriate tier.

## Dependencies

- Runtime deps go into the optional extra that owns them (see the planned-extras
  comment in `pyproject.toml`); dev-only tools go in the `dev` dependency group.
- Regenerate the lockfile after any dependency change: `uv lock`. The `lock` CI
  check fails the PR otherwise.

## Experiments

- **Don't edit code after launching a job** (lab wiki rule) — jobs run launch-time
  code, not queue-time code. The launcher phase adds per-run snapshotting.
- W&B: entity `nyu-mengye-group`, always config-driven (`wandb.*` keys in the hydra
  config) — never hardcode project names in code.
- LLM API keys are environment variables only — never CLI flags, configs, or code.

## Confidentiality

Research is confidential until published. See SECURITY.md for the never-commit list —
remember that today's private history becomes public at first release.
