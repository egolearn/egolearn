<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/icon-dark.svg">
  <img alt="egolearn: a robot head with a visor next to a video block" src="docs/assets/icon-light.svg" width="124">
</picture>

# egolearn

[![ci](https://github.com/agentic-learning-ai-lab/egolearn/actions/workflows/ci.yml/badge.svg)](https://github.com/agentic-learning-ai-lab/egolearn/actions/workflows/ci.yml)

EgoLearn is a Python library for training and evaluating models on first-person
video. It supports streaming and offline self-supervised learning, bounded replay
memory, temporal segmentation, and image–text learning.
[Agentic Learning AI Lab](https://agenticlearning.ai), New York University.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a workstation or compute node.
uv installs Python 3.12 and the training dependencies. On cluster login nodes,
use the [shell launcher](docs/cluster.md) to request a compute allocation first.

```bash
git clone https://github.com/agentic-learning-ai-lab/egolearn.git
cd egolearn
uv sync --extra train
uv run egl train toy_streaming dry_run=true
uv run egl train toy_streaming
```

The toy configuration trains a small convolutional model on synthetic colored
blocks using the CPU. It needs no dataset access, pretrained weights, or API keys.
The dry run validates the pipeline. The training run prints losses and writes its
run record under `runs/toy/records/`; the toy configuration does not save weights.

After installation, use `uv run egl` or activate `.venv` and run `egl` directly.
`egolearn` is an alias for the same command.

```bash
uv run egl --help
uv run egl data list
```

The [CLI guide](docs/cli.md) covers configuration files, data preparation,
evaluation, model export, cluster submission, and benchmark suites.

<a id="whats-here"></a>

## Training

Streaming training processes observations in chronological order and samples
previous observations from bounded replay buffers. Offline training shuffles the
selected dataset across epochs. Configurations select the model, losses, memory
policies, and batch composition. Both modes support resumable snapshots and run records.

Start with a method's configuration and data instructions:

| Method | Training guide |
| --- | --- |
| BabyCL | [Streaming and offline image–text learning on SAYCam](projects/babycl/README.md) |
| Memory Storyboard | [Streaming visual learning with temporal segmentation](projects/storyboard/README.md) |
| V-JEPA 2 | [Video representation learning on SAYCam and BabyView](projects/vjepa2/README.md) |

The SAYCam videos used by BabyCL and Memory Storyboard require
[Databrary access](docs/evaluation/data.md#saycam-access). Configure dataset roots
and prepared inputs according to the method's guide before running real-data training.
Datasets and checkpoints are not included in this repository.

See the [architecture guide](docs/design/architecture.md) for stream and memory
contracts, the [data-loading guide](docs/data_loading.md) for storage and worker
settings, and the [initialization guide](docs/initialization.md) for starting from
selected checkpoint weights.

## Model checkpoints

Published checkpoints are available from [EgoLearn on Hugging Face](https://huggingface.co/egolearn).
SAYCam and BabyView models require manual approval. You must already have access to
the corresponding training dataset on Databrary.

To request access:

1. Open the access-request form on the model's Hugging Face page.
2. Describe your **Intended use**, then enter a proof URL in
   **Link to Databrary access proof**.
   A screenshot of your Databrary permissions should identify you and the
   corresponding dataset, SAYCam or BabyView. Make sure EgoLearn reviewers can
   open the link. Redact unrelated personal information and exclude dataset
   images or videos.
3. Submit the form. No separate email is needed.

The EgoLearn team reviews the evidence before granting access to the weights.
KrishnaCam and WalkingTours checkpoints are available without an access request.

## Evaluation

Evaluate saved checkpoints independently with `egl eval` or the
`egolearn.evaluation.evaluate_checkpoint` API. The
[evaluation guide](docs/evaluation.md) lists configuration examples and checkpoint
formats; the [data guide](docs/evaluation/data.md) covers access and preparation.

Benchmarks include Labeled-S and BabyView object recognition, ImageNet decoding,
VTWT and Baby Winoground image–text matching, and shape bias. Additional guides cover
[spatial relations](docs/evaluation/spatial_relations.md),
[IntPhys 2](docs/evaluation/intphys2.md), and [GRASP](docs/evaluation/grasp.md).
Generative models can run [EgoSchema](docs/evaluation/egoschema.md) and
[Ego4D NLQ](docs/evaluation/ego4d_nlq.md) through a served vision-language model.

The [BabyCL results](projects/babycl/README.md#saycam-results) compare two seeds
with the paper. [Benchmark suites](docs/benchmark_suites.md) pin inputs and seeds,
retain run records, and generate tables with baseline comparisons.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/egolearn/` | Streams, memory, models, training, evaluation, and method implementations |
| `src/egolearn/configs/` | Packaged Hydra defaults and runnable toy configurations |
| `projects/` | Method guides and experiment configurations ([conventions](projects/README.md)) |
| `tests/` | Unit tests and constrained video, slow, GPU, and LLM tiers |
| `scripts/` | Data preparation, benchmarks, and cluster launchers |
| `docs/` | User guides, API reference, design notes, and [roadmap](docs/roadmap.md) |

## Citation and contributing

Use [CITATION.cff](CITATION.cff) to cite the software, and cite the relevant method
papers when using their implementations. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development and review requirements, [SECURITY.md](SECURITY.md) for repository
hygiene, and [RELEASING.md](RELEASING.md) for the publication procedure.
