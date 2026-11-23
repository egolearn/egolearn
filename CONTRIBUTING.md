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
- **Test plan**: commands that ran, where, and the outcome.
  If CI covers everything, say so. State any checks that have not run.
- **Checklist**: the template's five top-level `- [x]` boxes, each ticked after verification.
  Add notes for conditional items, such as no dependency change or no code port.

Summary comes first; Test plan and Checklist are the last two sections. Other
sections belong between Summary and Test plan. Leave out review history, people,
cluster job IDs, dates, a copy of the changelog, and the title repeated as a
heading. Delete all template comments and empty optional sections. Aim for at
most 40 non-blank lines; the checker warns past 40 and rejects more than 60.

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

## Docs site

The landing page is `landing/index.html`, served at the site root; the MkDocs
site from `docs/` (`mkdocs.yml`) is served under `/docs/`. Pages outside `docs/`
(the root README, CHANGELOG, CONTRIBUTING, SECURITY, and the project READMEs) are
added at build time by `scripts/docs/hooks.py`, which also points links to files
outside the site at GitHub. Preview locally:

```bash
uv run --group docs mkdocs serve
```

For the complete local build, see [site development](scripts/docs/README.md).

Images in `landing/media/` are either drawn or generated for the page, or come
from published, openly licensed sources and the datasets' own project pages; the
page credits each one.

### Generated reference

The Reference section is generated in memory on each build; do not add its
pages under `docs/`. Component entries come from registry decorators, with
constructor arguments and defaults read from source, including inherited
constructors. Add a docstring to describe a component in the catalog.

Each package under `evaluation/benchmarks/` defines a literal
`BENCHMARK_METADATA` mapping in its `__init__.py`. It contains five nonempty
strings: `name`, `source`, `license`, `access`, and `prepare`. Use Markdown links
for sources and code spans for preparation commands. Distinguish data terms
from code licenses, and state when no preparer exists. Access is `public`,
`gated`, `IRB`, or `upstream-licensed`, following the
[access tiers](docs/evaluation/hub.md#access-tiers).

The curated Python API list lives in `scripts/docs/reference.py`. Entries name
their defining modules, such as `egolearn.pipeline.build`, rather than lazy
package exports. Mkdocstrings uses
[Griffe's static analysis](https://mkdocstrings.github.io/python/usage/configuration/general/#allow_inspection)
with inspection disabled, so the docs environment needs neither EgoLearn nor
Torch installed. The CLI reference reads the parser's argument notation and
help text; the config reference reads YAML header comments. Source edits also
trigger a rebuild during `mkdocs serve`.

## Porting policy

Propose a concrete use and scope before porting another implementation, and obtain
code-owner review. Adapt the port to configured paths, seeded randomness, and the
shared interfaces, with tests for its supported behavior.

### Licensing and attribution

EgoLearn is MIT-licensed. Contribute code only when you have the rights needed to
include it under the project's licensing policy. Keep third-party license notices
and attribution headers with adapted code, and update [third-party notices](THIRD_PARTY_NOTICES.md)
when a port requires them.

Do not vendor GPL or CC-BY-NC code into the library. Existing permissive ports keep
their original notices, including Apache-2.0 for mmcv/DINO-derived components and
BSD notices for RAFT and pytorch-examples-derived utilities. Use permissively
licensed alternatives for MAE-lineage utilities licensed CC-BY-NC.

## Versions and releases

Versions follow SemVer. During 0.x development, minor releases add features and
patch releases fix bugs. Record user-visible changes under `[Unreleased]` in
[CHANGELOG.md](CHANGELOG.md); released versions have dated headings.

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
Prefer a small set of behavior tests over a test for every function or input
spelling. Extend an existing test when it already exercises the behavior; keep
separate cases for known regressions and distinct failure modes. Share setup for
the same contract, and avoid Cartesian products of independent options unless
their interaction matters. Keep numerical correctness, reproducibility, checkpoint
safety, and data-integrity checks when trimming redundant smoke tests.

## Dependencies

- Runtime deps go into the optional extra that owns them (see the planned-extras
  comment in `pyproject.toml`); dev-only tools go in the `dev` dependency group.
- Regenerate the lockfile after any dependency change: `uv lock`. The `lock` CI
  check fails the PR otherwise.

## Experiments

- Submit experiments through `scripts/slurm/submit`, which archives a clean commit
  and runs the job from that snapshot. Keep archived code and environments unchanged
  until dependent jobs finish; see [snapshots and records](docs/cluster.md#snapshots-and-records).
- For local or interactive runs from an editable checkout, keep its source unchanged
  until the run finishes, since spawned workers import that source.
- Configure W&B through `wandb.*` settings and environment variables, using your
  own entity and project when logging runs.
- LLM API keys are environment variables only — never CLI flags, configs, or code.

Keep restricted data, credentials, and unpublished material outside Git. See
[repository hygiene](SECURITY.md#never-commit) for the contribution requirements.
