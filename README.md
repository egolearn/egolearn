<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/icon-dark.svg">
  <img alt="egolearn: a robot head with a visor next to a video block" src="docs/assets/icon-light.svg" width="124">
</picture>

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
uv sync --extra train        # creates .venv, installs package + torch + dev tools
uv run pre-commit install    # local gates: ruff, gitleaks, large-file blocker
uv run pytest                # unit tests (slow/gpu/llm tiers excluded by default)

# validate a full pipeline without reading any data
uv run python -m egolearn.train --config-dir projects/babycl/configs \
    --config-name saycam_default dry_run=true
# train (dataset roots come from the command line / config, never code)
uv run python -m egolearn.train --config-dir projects/babycl/configs \
    --config-name saycam_default stream.root_dir=$DATA_ROOT
```

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is provisioned automatically.

## What's here

The streaming learning framework. In plain terms: dataset adapters read
egocentric video as a time-ordered stream (SAYCam, BabyView, raw video,
synthetic test data); admission components watch the stream and decide what
goes into bounded replay buffers; a batch recipe describes how training
batches are sampled from those buffers; SSL and image-text losses train the
model; probes score it while it trains (4AFC, visual two-word test, Baby
Winoground). Buffers and sampling only ever handle lightweight references,
never image tensors, so they stay cheap; an admission component that needs
to look at image content pulls the frames it wants through the loader (both
shipped methods do this to decide where segments end). Training can follow the stream in order (the streaming mode) or
shuffle everything like ordinary offline training (the IID mode). Runs can
stop and resume exactly where they left off, and every run prints a
per-stage timing report. Memory Storyboard and BabyCL are included as
faithful ports of the original code, each expressed as a config file. The
full walkthrough: [docs/design/architecture.md](docs/design/architecture.md).

The rest of the consolidation (LifelongMemory's QA/memory line, more evals)
arrives per [docs/roadmap.md](docs/roadmap.md), and **nothing is ported
without triage** — see the porting policy there.

## Layout

| Path | Purpose |
| --- | --- |
| `src/egolearn/` | The installable library: `core` `config` `data` `memory` `model` `training` `evaluation`, plus `methods/<name>/` (per-paper components) |
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
