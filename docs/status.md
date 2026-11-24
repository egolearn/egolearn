# Status and limitations

This page lists what egolearn supports and the limits that change how you use
it. Check it before you prepare data or request compute.

## Supported workflows

- Streaming and offline training for the methods in the
  [overview](../README.md#methods), from extracted frames, packed frame stores
  or video files.
- Evaluation of saved models on the benchmarks in the
  [evaluation guide](evaluation.md), with fixed-feature linear probes and
  full-network fine-tuning on ImageNet.
- Model export, initialization from a checkpoint, Slurm submission and
  benchmark suites across seeds.

## Training limits

- Training runs in one process on one device. There is no distributed
  data-parallel training.
- Mixed precision through `driver.amp` works in streaming runs only. V-JEPA 2
  sets precision in its own recipes.
- Video decoding with PyAV needs constant-frame-rate files with frame-count
  metadata.
- Frame-directory streams read frames you have already extracted. Their `fps`
  setting gives the frame times, but it does not extract or resample frames.
- BabyCL needs aligned SAYCam transcripts, which you obtain separately, and
  egolearn has no converter for raw transcripts. Its offline recipe also needs
  a storyboard file that egolearn cannot export.

## Evaluation limits

- kNN and SVM readouts are not implemented.
- EgoSchema and Ego4D NLQ query a vision-language model that you serve through
  an OpenAI-compatible API. The `qa` extra installs the client, not a server.
- Access to released weights and access to benchmark data are separate
  requests. A dataset's catalog entry tells you how to obtain it, not that a
  download is available to you.

## Reproduction

Method pages state which paper protocol they follow and how they differ from
it. Scores are comparable only when preprocessing, splits, training budget and
readout all match. The V-JEPA 2 recipes train small models on egocentric video.
They do not convert Meta's checkpoints, and they include neither action
conditioning nor V-JEPA 2.1.
