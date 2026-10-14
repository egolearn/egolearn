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

Run Python setup and commands on a workstation or compute node. For cluster
login-node submission, start with the [shell launcher](docs/cluster.md).

```bash
git clone git@github.com:agentic-learning-ai-lab/egolearn.git
cd egolearn
uv sync --extra train        # creates .venv, installs package + torch + dev tools
uv run pre-commit install    # local gates: ruff, gitleaks, large-file blocker
uv run pytest                # unit tests (slow/gpu/llm tiers excluded by default)

# validate wiring without iterating the stream; Hub references may download artifacts
uv run egl train projects/babycl/configs/saycam_default.yaml dry_run=true
# train (dataset roots come from the command line / config, never code)
uv run egl train projects/babycl/configs/saycam_default.yaml stream.root_dir=$DATA_ROOT
```

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is provisioned automatically.

## Command line

After `uv sync`, use `uv run egl` (also available as `egolearn`), or `egl` in an activated environment:

```bash
egl --help
egl data list
egl doctor
egl train toy_streaming dry_run=true
egl train projects/babycl/configs/saycam_default.yaml seed=42
egl eval imagenet_linear_eval checkpoint=runs/example/final.pth data_root=/path/to/imagenet
```

See the [CLI guide](docs/cli.md) for data preparation, diagnostics, submission,
benchmark suites, configuration paths, and multirun.

## SAYCam dataset access

The SAYCam videos used in BabyCL and Memory Storyboard are hosted on
[Databrary](https://nyu.databrary.org/volume/564/).
See the [evaluation guide](docs/evaluation.md#saycam-access) for IRB approval,
access requirements, and benchmark data preparation.

## What's here

The streaming learning framework. In plain terms: dataset adapters read
egocentric video as a time-ordered stream (SAYCam, BabyView, raw video,
synthetic test data); admission components watch the stream and decide what
goes into bounded replay buffers; a batch recipe describes how training
batches are sampled from those buffers; SSL and image-text losses train the
model; probes score it while it trains (4AFC, linear probing, visual two-word
test, Baby Winoground, shape bias). Buffers and sampling only ever handle lightweight
references, never image tensors, so they stay cheap; an admission component that needs
to look at image content pulls the frames it wants through the loader (both
shipped methods do this to decide where segments end). Training can follow the stream in order (the streaming mode) or
shuffle everything like ordinary offline training (the IID mode). Runs can
stop and resume exactly where they left off, and every run prints a
per-stage timing report. Memory Storyboard and BabyCL are included as
faithful ports of the original code, each expressed as a config file. The
full walkthrough: [docs/design/architecture.md](docs/design/architecture.md).
Frame dumps can be packed into a shard store for fast random reads; see
[data loading](docs/data_loading.md).

Trained checkpoints can be evaluated independently through
`egolearn.evaluation.evaluate_checkpoint` or `egl eval`.
Full ImageNet decoding runs as a standalone benchmark, with dataset preparation
deferred until evaluation. See [the evaluation guide](docs/evaluation.md) for
Python usage, CLI configuration, and checkpoint compatibility.

The rest of the consolidation (LifelongMemory's QA/memory line, more evals)
arrives per [docs/roadmap.md](docs/roadmap.md), and **nothing is ported
without triage** — see the porting policy there.

For staged training with selected checkpoint weights, see the
[initialization guide](docs/initialization.md).

## Layout

| Path | Purpose |
| --- | --- |
| `src/egolearn/` | The installable library: `core` `config` `data` `memory` `model` `training` `evaluation`, plus `methods/<name>/` (per-paper components) |
| `src/egolearn/configs/` | Packaged hydra configs (`pkg://egolearn.configs`) — entry points never require running from the repo root |
| `projects/` | One directory per paper: thin entry points + configs composing on the library ([conventions](projects/README.md)) |
| `tests/` | Tiered tests: unit (default), `video`, `slow`, `gpu`, `llm` markers |
| `scripts/` | Committed operational scripts and cluster launchers |
| `docs/` | Roadmap and design notes |

Spatial relation categorization evaluates two-choice relation matching in
2D abstract scenes and 3D rendered scenes. The registered `probe:spatial_relations`
reports accuracy with equal component weights; see the
[task definition, data preparation, and probe configuration](docs/evaluation/spatial_relations.md).

## Lifecycle

Private now → **public at first release, with full git history** (so: public-repo hygiene
from day 1, enforced by gitleaks and pre-commit). After the flip, unpublished research
continues on branches of the private `egolearn-dev` mirror and merges here at paper
release. Details and exact workflow: [RELEASING.md](RELEASING.md).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) (branching, gates, style, tests) and
[SECURITY.md](SECURITY.md) (what must never be committed) before your first PR.

## Cluster jobs

On login nodes, use `scripts/slurm/submit [sbatch options] -- <command>` for containerized jobs,
CPU environment preparation and immutable code snapshots. See [cluster setup](docs/cluster.md)
for the scratch cache environment, pinned image and Hydra sweep arrays. Run all
Python commands, including `egl doctor` and benchmark planning, in compute allocations.
