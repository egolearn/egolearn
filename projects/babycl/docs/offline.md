# Offline BabyCL inputs

The mixed-batch recipe requires a precomputed storyboard in addition to frames,
transcripts, and a vocabulary. Produce a JSON manifest from your segmentation
procedure using the contract below.
The pairs-only driver (`segments_per_batch=0`, with only the image/text loss)
does not require a storyboard. Omit `storyboard_json` and `segment_views` for it.

## Storyboard format

This synthetic example contains two frames and one segment:

```json
{
  "fps": 5.0,
  "videos": [{
    "video_dir": "S_0",
    "frames": ["S_0_0.00.jpg", "S_0_0.20.jpg"],
    "segments": [{"start": 0, "end": 2, "global_id": 42}]
  }]
}
```

- `fps` must equal `stream.fps`.
- `videos` contains one entry for every selected nonempty session. Session names
  are unique and correspond to directories under `stream.root_dir`.
- `frames` is the exact list of image filenames in that session, ordered by the
  timestamps parsed by `SaycamStream` (numeric timestamp order, not lexical order).
- Segment bounds are zero-based frame indices with an exclusive `end`. Segments
  must be nonempty, ordered, contiguous, and cover `[0, len(frames))` without gaps
  or overlaps. Utterance timestamps are not segment indices.
- `global_id` is a nonnegative 64-bit integer shared across sessions. Segments
  with the same ID supply positives for the supervised contrastive loss.
  Supply it for every segment or omit it throughout the manifest. When omitted,
  IDs are assigned consecutively across the entire manifest in video and segment
  order, before session filtering. This gives each segment a distinct global ID.
- `stream.root_dir` owns data paths, overriding any `root_dir` stored in the
  manifest. The file's SHA-256 is printed when its index is built.

Utterance chunks come from `stream.transcript_json` through the same AN-chunk
emitter as streaming training, including timestamp matching and deduplication.
The index requires enough chunks to fill `pairs_per_batch`.

## Run

```bash
uv run python -m egolearn.train --config-dir projects/babycl/configs --config-name saycam_iid \
  stream.root_dir="$DATA_ROOT" stream.transcript_json="$TRANSCRIPT_JSON" \
  tokenizer.vocab_json="$VOCAB_JSON" driver.batch.storyboard_json="$STORYBOARD_JSON"
```

Each update draws the configured number of storyboard segments with replacement
and retains their global IDs, including repeated draws of the same segment.
Each epoch shuffles the full utterance index once. Prefetch and resume use the
same seed-derived batch sequence. Resume requires the same storyboard bytes as
the saved run, so changed segment labels cannot silently alter the objective.
