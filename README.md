<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/icon-dark.svg">
  <img alt="egolearn: a robot head with a visor next to a video block" src="docs/assets/icon-light.svg" width="124">
</picture>

# egolearn

[![ci](https://github.com/agentic-learning-ai-lab/egolearn/actions/workflows/ci.yml/badge.svg)](https://github.com/agentic-learning-ai-lab/egolearn/actions/workflows/ci.yml)

EgoLearn trains and evaluates models from first-person video streams. It supports
streaming and offline self-supervised learning, bounded replay memory, temporal
segmentation, and image–text learning.
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
See the [evaluation data guide](docs/evaluation/data.md#saycam-access) for IRB approval,
access requirements, and benchmark data preparation.

<a id="whats-here"></a>

## Training and evaluation

EgoLearn trains models from video streams while controlling what the learner keeps
in memory and revisits during training. Streaming runs process observations in
chronological order and sample from bounded replay buffers. Offline runs shuffle
the selected dataset across epochs.

Memory stores frame references and metadata, while the loader decodes and augments
images when training or segmentation needs them. A configuration selects the memory
policy, batch composition, model, and losses. The
[architecture guide](docs/design/architecture.md) explains how these choices fit together.

The included [BabyCL](projects/babycl/README.md) and
[Memory Storyboard](projects/storyboard/README.md) projects provide method implementations
and experiment configs. Runs support snapshot continuation and report time spent in
each training stage. For large frame collections, the [data-loading guide](docs/data_loading.md)
covers packed storage, worker processes, and augmentation controls.

Trained checkpoints can be evaluated independently through
`egolearn.evaluation.evaluate_checkpoint` or `egl eval`.
Full ImageNet decoding runs as a standalone benchmark, with dataset preparation
deferred until evaluation. See [the evaluation guide](docs/evaluation.md) for
Python usage, CLI configuration, and checkpoint compatibility.

The [roadmap](docs/roadmap.md) tracks candidate additions, including LifelongMemory's
QA and retrieval work. Each port requires a concrete need and an agreed scope.

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
| `docs/` | Training and evaluation guides, reference, design notes, and roadmap |

Spatial relation categorization evaluates two-choice relation matching in
2D abstract scenes and 3D rendered scenes. The registered `probe:spatial_relations`
reports accuracy with equal component weights; see the
[task definition, data preparation, and probe configuration](docs/evaluation/spatial_relations.md).

IntPhys 2 evaluates intuitive physics by violation of expectation: a video model
should find each physically impossible video more surprising than its matched
possible twin. The registered `probe:intphys2` reports pairwise accuracy and AUROC from
V-JEPA prediction error; see the
[task definition, pinned data download, and probe configuration](docs/evaluation/intphys2.md).

GRASP Level 2 uses the same readout across 16 physics tests, reporting pairwise
accuracy, AUROC, and scores by test and concept. See the [GRASP guide](docs/evaluation/grasp.md)
for manifest preparation and probe configuration.

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
