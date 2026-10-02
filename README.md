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
uv run egl train projects/babycl/configs/saycam_default.yaml dry_run=true
# train (dataset roots come from the command line / config, never code)
uv run egl train projects/babycl/configs/saycam_default.yaml stream.root_dir=$DATA_ROOT
```

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is provisioned automatically.

## Command line

Use `egl` (also available as `egolearn`) after `uv sync`, or prefix it with `uv run`:

```bash
egl --help
egl train toy_streaming dry_run=true
egl train projects/babycl/configs/saycam_default.yaml seed=42
egl eval imagenet_linear_eval checkpoint=runs/example/final.pth data_root=/path/to/imagenet
```

See the [CLI guide](docs/cli.md) for configuration paths, overrides, and multirun.

## SAYCam dataset access

The SAYCam videos used in the BabyCL and Memory Storyboard projects are hosted on
[Databrary](https://nyu.databrary.org/volume/564/). Researchers can request access
with approval from their institution's IRB and must follow the dataset's data-use
terms.

For the VTWT and Baby Winoground evaluation benchmarks, prepare the data following the
[BabyVLM preparation notes](https://github.com/ShawnKing98/babylmms-eval#data-preparation):

1. Download `frames.txt` from the
   [Multimodal SAYCam-S data release](https://nyu.databrary.org/volume/564/session/67447).
2. Rename the downloaded file to `frames.zip` and extract it into a local image
   directory. The release is reported to contain 600,285 images (about 14 GB).
3. Configure that directory as `image_dir` for VTWT and `pos_dir` for Baby
   Winoground. The [evaluation guide](docs/evaluation.md#babyvlm-benchmark-data)
   describes the labels, synthetic negative images, and other probe inputs.

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

Trained checkpoints can be evaluated independently through
`egolearn.evaluation.evaluate_checkpoint` or `egl eval`.
Full ImageNet decoding runs as a standalone benchmark, with dataset preparation
deferred until evaluation. See [the evaluation guide](docs/evaluation.md) for
Python usage, CLI configuration, and checkpoint compatibility.

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
