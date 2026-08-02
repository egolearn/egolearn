# egolearn

[![ci](https://github.com/agentic-learning-ai-lab/egolearn/actions/workflows/ci.yml/badge.svg)](https://github.com/agentic-learning-ai-lab/egolearn/actions/workflows/ci.yml)

Learning from first-person video streams: streaming self-supervised representation
learning, continual learning and replay, temporal segmentation, multimodal learning,
memory retrieval, and egocentric video question answering.
[Agentic Learning AI Lab](https://agenticlearning.ai), New York University.

> **Private — unpublished research.** This repo is confidential until its first public
> release (lab policy: research is confidential until published). Do not share code,
> results, or ideas outside the lab.

## Quickstart

```bash
git clone git@github.com:agentic-learning-ai-lab/egolearn.git
cd egolearn
uv sync                      # creates .venv, installs package + dev tools
uv run pre-commit install    # local gates: ruff, gitleaks, large-file blocker
uv run pytest                # unit tests (slow/gpu/llm tiers excluded by default)
```

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is provisioned automatically.

## What this will become

An ecosystem consolidating the lab's egocentric-learning line — Memory Storyboard,
LifelongMemory, BabyCL, and the video-SSL toolchain — into one integrated codebase:
base utilities for streaming and video processing, replay buffers, temporal
segmentation, self-supervised losses, and evaluation on multimodal readout and QA.
Currently scaffold-only; components arrive per [docs/roadmap.md](docs/roadmap.md),
and **nothing is ported without triage** — see the porting policy there.

## Layout

| Path | Purpose |
| --- | --- |
| `src/egolearn/` | The installable library (subpackages arrive with their code — see roadmap) |
| `src/egolearn/configs/` | Packaged hydra configs (`pkg://egolearn.configs`) — entry points never require running from the repo root |
| `projects/` | One directory per paper: thin entry points + configs composing on the library ([conventions](projects/README.md)) |
| `tests/` | Tiered tests: unit (default), `video`, `slow`, `gpu`, `llm` markers |
| `scripts/` | Committed operational scripts and cluster launchers |
| `docs/` | Roadmap and design notes |

## Lifecycle

Private now → **public at first release, with full git history** (so: public-repo hygiene
from day 1, enforced by gitleaks and pre-commit). After the flip, unpublished research
continues on branches of the private `egolearn-dev` mirror and merges here at paper
release. Details and exact workflow: [RELEASING.md](RELEASING.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) (branching, gates, style, tests) and
[SECURITY.md](SECURITY.md) (what must never be committed) before your first PR.
