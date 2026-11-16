# Status and limitations

EgoLearn supports streaming and offline training, saved-model evaluation, and
reproducible benchmark suites. Use this page to choose a supported workflow and
check its limits before preparing data or allocating compute.

## Training and data

| Workflow | Support | Starting point |
| --- | --- | --- |
| Synthetic CPU training | Runs without data access or pretrained weights | [Quickstart](../README.md#quickstart) |
| BabyCL | Streaming and offline image–text learning on prepared SAYCam inputs | [BabyCL](../projects/babycl/README.md) |
| Memory Storyboard | Streaming visual learning with temporal segmentation | [Memory Storyboard](../projects/storyboard/README.md) |
| MAE and StreamMAE | Masked image reconstruction, full-pass runs, and matched streaming/offline ablations | [MAE and StreamMAE](../projects/streammae/README.md) |
| WalkingTours | Original-video preparation and BabyCL or StreamMAE training on extracted frames | [WalkingTours](datasets/walkingtours.md) |
| V-JEPA 2 | Initial offline runs and matched full IID/streaming recipes on SAYCam and BabyView | [V-JEPA 2](../projects/vjepa2/README.md) |
| KrishnaCam | Direct-video visual training with streaming and offline recipes | [KrishnaCam](../projects/krishnacam/README.md) |
| Frame directories and packed stores | Ordered images, aligned transcripts, replay, and worker loading | [Data loading](data_loading.md) |
| Raw video | PyAV decoding on CPU or supported NVIDIA hardware | [Video training](../projects/babycl/docs/krishnacam.md) |

The training loops run in one process and do not implement distributed data-parallel
training. The shared streaming driver supports mixed precision through `driver.amp`;
the shared offline driver rejects that option. V-JEPA 2 configures precision in
its own recipes. Exact step limits can interrupt an advance, and snapshots retain
the remaining work for continuation.

PyAV requires constant-rate videos with frame-count metadata. Frame-directory
recipes require already prepared images; setting their `fps` does not extract or
resample frames. Match the timing, orientation, and crop settings documented by
the method before comparing results.

BabyCL's aligned SAYCam transcripts are obtained separately, and its mixed offline
recipe needs a precomputed storyboard. EgoLearn does not provide a raw-transcript
converter or storyboard exporter. See [training inputs](../projects/babycl/docs/training_data.md)
and [offline inputs](../projects/babycl/docs/offline.md).

## Evaluation and artifacts

The [evaluation guide](evaluation.md#benchmark-selection) lists runnable configs,
required data, model capabilities, and score definitions. Generic fixed-feature
linear and SGD probes are available, along with standalone ImageNet full-network
fine-tuning. kNN and SVM readouts are not implemented.

EgoSchema and Ego4D NLQ query a served generative model through an OpenAI-compatible
client. The `qa` extra installs the client, not the model server. Read the
[server prerequisites](evaluation/egoschema.md#serving-the-model) and each dataset's
access restrictions before running either benchmark.

Model exports use safetensors weights and JSON construction metadata. They can be
loaded for evaluation or [initialization](initialization.md); full snapshots also
retain training state for resuming a run. See [checkpoint trust](../SECURITY.md#model-export-and-checkpoint-trust).

[Results publication](publishing_results.md) uploads per-run evidence and updates
the website from a public Hugging Face feed. Published scores must retain their
training and evaluation protocols for meaningful comparisons.

Dataset catalog entries describe access requirements and preparation commands.
They do not guarantee an available download. [Checkpoint access](evaluation/hub.md#checkpoint-access)
and [benchmark data access](evaluation/data.md) are separate requirements.

## Reproduction limits

Method guides specify the supported scientific protocols and deviations from their
papers. Matching model capabilities does not make different preprocessing, splits,
training budgets, or readout procedures comparable.

The V-JEPA 2 recipes use small egocentric models rather than Meta's original
training scale. They do not provide external Meta checkpoint conversion, action
conditioning, or V-JEPA 2.1. The [method guide](../projects/vjepa2/README.md#recipe-and-compatibility)
describes the architecture and evaluation differences.
