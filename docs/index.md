# egolearn

egolearn trains and evaluates visual and image–text models on first-person
video. If you are new, start with the [overview](overview.md), which installs
the package and trains a toy model on the CPU.

## Get started

- [First checkpoint evaluation](evaluation/first_evaluation.md): score a public model.
- [Command line](cli.md) and [cluster jobs](cluster.md): run training locally or on Slurm.
- [Data loading](data_loading.md): packed frame stores and loader workers.
- [Start from a checkpoint](initialization.md): copy trained weights into a new run.

## Datasets and evaluation

- [Dataset access](evaluation/data.md): SAYCam, BabyView and benchmark inputs.
- [Checkpoint evaluation](evaluation.md): benchmarks and their scores.
- [Models and Hub data](evaluation/hub.md): released weights and access requests.
- [Benchmark suites](benchmark_suites.md) and [publishing results](publishing_results.md).

## Methods

[BabyCL](projects/babycl.md), [Memory Storyboard](projects/storyboard.md),
[MAE and StreamMAE](projects/streammae.md), [V-JEPA 2](projects/vjepa2.md),
[KrishnaCam](projects/krishnacam.md).

## Design and project

- [Architecture](design/architecture.md): how streams, memories, models and probes fit together.
- [Status and limitations](status.md), [changelog](changelog.md) and [contributing](contributing.md).
