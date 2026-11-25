# CLAUDE.md

Learning from egocentric video streams (streaming SSL, continual learning, replay,
memory, video QA). `docs/status.md` describes supported workflows and limitations;
check the source before assuming a component exists.

## Commands

```bash
uv sync                                          # setup (.venv + dev tools)
uv run pytest                                    # tests (slow/gpu/llm excluded by default)
uv run ruff check --fix . && uv run ruff format .
uv run mypy
uv run pre-commit run --all-files
```

## Layout

- `src/egolearn/` — installable src-layout package; hydra configs are packaged at
  `src/egolearn/configs` and composed via `pkg://egolearn.configs`.
- `projects/<name>/` — per-paper entry points/configs; import from `egolearn` only.
- `scripts/` — operational scripts and cluster launchers, always committed.
- `tests/` — tiers via markers: unmarked unit, `video`, `slow`, `gpu`, `llm`.

## Hard rules

- **Never port code from the source repos without an explicit request** — porting
  is triaged per-PR by the code owners (CONTRIBUTING.md "Porting policy").
- **Merge commits only** — never rebase, squash, or force-push shared history.
- Open PRs against `egolearn/egolearn`.
- Never commit secrets, personal cluster paths, dataset frames (SAYCam/BabyView are
  IRB-controlled; the only exception is credited, already-public images in
  `landing/media/`), or files >500 KB (SECURITY.md has the full list).
- LLM API keys: environment variables only, never CLI flags or configs.
- Imports are absolute (`from egolearn...`); code must work from an installed
  wheel — no repo-root-relative paths. Dataset roots come from config, never code.
- Dependency changes: add to the owning extra / dev group in `pyproject.toml`,
  then regenerate with `uv lock` (the `lock` CI check enforces this).
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
- PR messages follow CONTRIBUTING.md "The PR message": draft the body in a
  file, run `scripts/check_pr_message.py`, and keep the description aligned with
  the final diff. Review history, people, and dates stay out of the message.
- Third-party-derived files keep attribution headers; never vendor GPL or
  CC-BY-NC code (CONTRIBUTING.md has the licensing and attribution rules).
