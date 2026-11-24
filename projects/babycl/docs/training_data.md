# BabyCL SAYCam inputs

This page describes the inputs that BabyCL's SAYCam configs read: frames,
aligned transcripts, a vocabulary and, for the offline variant, a storyboard.
The [BabyCL page](../README.md#training) shows how to pass them to training.
You need [approved SAYCam access](../../../docs/evaluation/data.md#saycam-access)
to obtain them, and the repository contains no restricted data.

To reproduce the paper, ask [the BabyCL team](mailto:mengye@nyu.edu) for the
prepared frames and the aligned transcript file `expand_correct_seg.json` once
you have SAYCam access. EgoLearn includes neither the transcripts nor a
converter from raw SAYCam transcripts.

## Frames

Store upright frames sampled at 5 fps, with one directory per recording. The
directory names must sort chronologically, and each filename ends in the
frame's time in seconds from the start of its recording:

```text
SAYCAM_ROOT/
  S_example/
    S_example_0.00.jpg
    S_example_0.20.jpg
    S_example_0.40.jpg
```

The stream reads the timestamp from the last underscore-separated field and
accepts `.jpg`, `.jpeg` and `.png` files. The default config selects directories
matching `S_*`. Because `stream.fps` does not resample frames, extract them at
5 fps. Raw SAYCam frames are upside down and BabyCL does not flip them, so check
whether your frames are already upright before you flip them.

## Aligned transcripts

`stream.transcript_json` is a JSON list with one entry per utterance. This
example aligns one utterance with the three frames above:

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

`video_filename` without its extension must match a recording directory.
`timestamps` lists the recording-relative times of the utterance's frames. Keep
any gaps in that list, because BabyCL splits noncontiguous runs into separate
image–text pairs. The optional `clip_importance` field weights the pair in the
clip-importance ablation. The default tokenizer expects the prepared release's
normalized text, so do not substitute raw speech transcripts.

## Vocabulary

Use the [CVCL training vocabulary](../../../docs/evaluation/data.md#training-vocabulary)
for the paper recipe, keeping its word-to-index mapping, and evaluate a model
with the vocabulary it was trained with.

## Offline storyboard

The offline config `saycam_iid.yaml` also needs a storyboard that splits every
selected recording into visual segments. EgoLearn has no storyboard exporter, so
you write this JSON file from your own segmentation. This example splits the
recording above into two segments:

```json
{
  "fps": 5.0,
  "videos": [{
    "video_dir": "S_example",
    "frames": ["S_example_0.00.jpg", "S_example_0.20.jpg", "S_example_0.40.jpg"],
    "segments": [{"start": 0, "end": 2, "global_id": 7}, {"start": 2, "end": 3, "global_id": 8}]
  }]
}
```

`fps` must equal `stream.fps`, and `videos` needs one entry for each selected
recording. `frames` lists the recording's files in numeric timestamp order.
Segment bounds are zero-based frame indices with an exclusive `end`, and a
recording's segments must cover all its frames without gaps or overlap.

`global_id` is an optional nonnegative integer label. Segments with the same ID
are positives for SupCon. Give it for every segment or for none; without it,
each segment is its own class. `stream.root_dir` sets the frame location and
overrides any `root_dir` stored in the file. Resuming a run requires the same
storyboard file as the saved run.

A pairs-only run, with no segment slot, needs no storyboard.
