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
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

### The PR message

Describe the final change for someone who did not watch it happen. Update the
message when the diff changes; review discussion stays in comments.

- **Title**: one line, at most 72 characters, saying what changes. No
  sentence-ending period (a dot inside a version is fine), issue number, or WIP.
- **Summary**: a short paragraph on the problem and the outcome, then one bullet
  per change naming the file or component and what it does now. End with
  `Closes #N` or `Part of #N` when there is an issue.
- **Decisions** (optional): decisions a reviewer may weigh differently, one
  bullet each with the reason. Delete the section when there are none.
- **Results** (when an experiment, config, or reported number changes): what ran
  (config, seeds, hardware, wall time), a table of numbers against the baseline,
  links to runs and artifacts, and a command to reproduce them. If nothing has
  run yet, say so and explain what is pending. Delete the section otherwise.
- **Test plan**: commands that ran, where, and the outcome; cluster jobs by ID.
  If CI covers everything, say so. State any checks that have not run.
- **Checklist**: the template's five top-level `- [x]` boxes, each ticked after verification.
  Add notes for conditional items, such as no dependency change or no code port.

Summary comes first; Test plan and Checklist are the last two sections. Other
sections belong between Summary and Test plan. Leave out review history, people,
dates, a copy of the changelog, and the title repeated as a heading. Delete all
template comments and empty optional sections. Aim for at most 40 non-blank
lines; the checker warns past 40 and rejects more than 60.

Draft the body in a file and check it before opening or updating the PR:

```bash
uv run python scripts/check_pr_message.py --title "describe the change" --body-file body.md
```

Optionally, include changed files to enable the Results warning. In Bash 4.4 or
newer, preserve each path using NUL delimiters and a quoted array:

```bash
mapfile -d '' -t changed_files < <(git diff --name-only -z origin/main...HEAD)
uv run python scripts/check_pr_message.py --title "describe the change" --body-file body.md \
    --files "${changed_files[@]}"
```

The diff command lists committed changes; include any uncommitted paths in
`--files` when checking earlier. Put `--files` last: every following argument is
a literal filename, even `--help` or `--`. Quote any paths you add manually.
To check an existing PR from this checkout, use `uv run python scripts/check_pr_message.py --pr N`
(requires authenticated `gh`). Checking a local draft needs only Python's
standard library, with no GitHub access or training dependencies.

This is a local helper, **not a CI gate**. It checks title, sections, order,
nonempty content, checklist boxes, leftover comments, and length, not prose
quality. Missing issue links, 41–60 non-blank lines, and a missing Results
section when a changed path is under `projects/` or is a YAML file under
`configs/` are warnings; other findings are errors. Warnings exit successfully.
Without `--files`, the Results warning is skipped. The path heuristic includes
`.yaml` and `.yml`; reported numbers elsewhere are the author's responsibility.
Fenced examples count as content, never headings, checklist boxes, or comments.

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
