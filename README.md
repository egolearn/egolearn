<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/icon-dark.svg">
  <img alt="egolearn: a robot head with a visor next to a video block" src="docs/assets/icon-light.svg" width="124">
</picture>

# egolearn

[![ci](https://github.com/egolearn/egolearn/actions/workflows/ci.yml/badge.svg)](https://github.com/egolearn/egolearn/actions/workflows/ci.yml)

egolearn is a Python library for training and evaluating visual and image–text
models on first-person video. Streaming runs read the video once in
chronological order and replay past frames from bounded memory, while offline
runs shuffle the whole dataset across epochs. It is developed by the
[Agentic Learning AI Lab](https://agenticlearning.ai) at New York University.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the
dependencies. The toy configuration trains a small convolutional network on
synthetic colored blocks, on the CPU, without any dataset or pretrained weights.

```bash
git clone https://github.com/egolearn/egolearn.git
cd egolearn
uv sync --extra train
uv run egl train toy_streaming dry_run=true
uv run egl train toy_streaming
```

The dry run builds and checks the whole pipeline, then exits. The training run
prints a loss line every five steps and writes a run record under
`runs/toy/records/`, but it saves no weights. To train, export and reload a
model, follow the [toy export example](docs/cli.md#toy-model-export).

Run `uv run egl --help` to list the commands, or activate `.venv` and call
`egl` directly.

## Next steps

- [First checkpoint evaluation](docs/evaluation/first_evaluation.md): score a public model on a CPU.
- [Command line](docs/cli.md): configs, overrides, export and data preparation.
- [Cluster jobs](docs/cluster.md): submit training to Slurm.
- [Status and limitations](docs/status.md): what is supported.

## Methods

| Method | What it trains |
| --- | --- |
| [BabyCL](projects/babycl/README.md) | Visual and verbal representations from SAYCam in one chronological pass |
| [Memory Storyboard](projects/storyboard/README.md) | Visual representations from a stream, with temporal segmentation |
| [MAE and StreamMAE](projects/streammae/README.md) | Image encoders by masked reconstruction, streaming or offline |
| [V-JEPA 2](projects/vjepa2/README.md) | Video encoders on SAYCam or BabyView |
| [KrishnaCam](projects/krishnacam/README.md) | Visual baselines trained directly from MP4 recordings |

Each method page lists the inputs it needs, and SAYCam and BabyView require
[Databrary access](docs/evaluation/data.md). The repository itself contains no
datasets or checkpoints.

## Evaluation

`egl eval` scores a saved model on Labeled-S, BabyView and ImageNet object
recognition, VTWT and Baby Winoground image–text matching, shape bias, spatial
relations, IntPhys 2 and GRASP, while EgoSchema and Ego4D NLQ query a served
vision-language model. The [evaluation guide](docs/evaluation.md) lists each
benchmark and the data it needs. The [model catalog](docs/evaluation/hub.md)
lists the released weights: models trained on KrishnaCam and WalkingTours are
public, and models trained on SAYCam or BabyView require approval.

## Citation and contributing

Cite the software with [CITATION.cff](CITATION.cff), and cite the method papers
for the implementations you use. [CONTRIBUTING.md](CONTRIBUTING.md) covers
development and review, and [SECURITY.md](SECURITY.md) covers checkpoint trust
and vulnerability reporting.
