# BabyCL training inputs

Prepare SAYCam-S frames, aligned transcripts, and a vocabulary before running the
BabyCL configuration. You need [approved SAYCam access](../../../docs/evaluation/data.md#saycam-access)
for the recordings and prepared frames; the repository contains no restricted data.

## Frames

Use upright frames sampled at 5 fps, with one directory per recording. The
default configuration selects session directories matching `S_*` and orders them
lexicographically. Keep the original chronological recording names.

Frame filenames end in their timestamp in seconds relative to that recording.
For example, this synthetic recording has three frames spaced 0.2 seconds apart:

```text
SAYCAM_ROOT/
  S_example/
    S_example_0.00.jpg
    S_example_0.20.jpg
    S_example_0.40.jpg
```

The stream reads the final underscore-separated field as the timestamp. It also
accepts `.jpeg` and `.png` files. Prepare the frames at the intended sampling rate
before training; setting `stream.fps=5` does not resample a frame directory.
BabyCL's views expect upright input and do not apply SAYCam's orientation fix.
Check the preparation metadata before flipping frames that may already be upright.

Use the same prepared frames as the paper when reproducing its results. If you
extract frames yourself, retain recording-relative timestamps and record the
extraction, orientation, resizing, and encoding settings. Different preprocessing
defines a different training input even when filenames match.

## Aligned transcripts

For paper reproduction, obtain the prepared `expand_correct_seg.json` and matching
frames from the BabyCL authors under your approved data access. The placeholder
in `saycam_shared.yaml` is not a downloadable asset. EgoLearn does not bundle the
transcripts or a converter from raw SAYCam transcripts into this alignment format.

`stream.transcript_json` accepts a JSON list. This synthetic example aligns one
utterance with the three frames above:

```json
[
  {
    "video_filename": "S_example.mp4",
    "utterance": "a red ball",
    "utterance_num": 0,
    "timestamps": [0.0, 0.2, 0.4]
  }
]
```

The filename without its extension must match the session directory exactly.
`timestamps` contains recording-relative frame times in seconds, and empty lists
are skipped. Keep the full timestamp list, including gaps: BabyCL splits
noncontiguous frame runs into separate utterance chunks. `utterance_num` identifies
the utterance, and optional `clip_importance` supplies a numeric sample weight
when that ablation is enabled.

The default SAYCam tokenizer expects pre-normalized text. Preserve the text and
alignment from the prepared training release rather than substituting raw speech
transcripts. The stream surfaces each utterance only at its last timestamp.

## Vocabulary and configuration

Use the [CVCL training vocabulary](../../../docs/evaluation/data.md#training-vocabulary)
with the paper recipe, keeping its original word-to-integer mapping. Evaluation
must use the same vocabulary as training. For a custom training corpus, retain
the vocabulary used to construct its model rather than rebuilding it for evaluation.

Set `SAYCAM_ROOT` to the frame-directory root, `SAYCAM_TRANSCRIPTS` to the aligned
JSON file, and `BABYCL_VOCAB` to the vocabulary JSON. Follow the
[training command](../README.md#training) to validate the pipeline and start training.
In-training probes require their own prepared data; `probes=[]` disables them.
