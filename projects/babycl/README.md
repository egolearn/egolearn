# BabyCL / BabyCLIP (reproduction project)

Reproduces the BabyCLIP default run (continual visual + verbal learning on
SAYCam, single chronological pass).

**This project owns the paper's contribution, as 1:1 ports** (in
`src/egolearn/methods/babycl/`; the config's `_imports_` list makes them
register themselves on import):

* `refine.py` — **verbatim** segmentation math from mcl's
  `utils/online_ms_refine.py` (prefix sums, chunk expansion 2–12 s,
  gap cutting, representative picking)
* `admission:babycl_online_segmenter` — the OnlineSegmenter: a sliding window over
  the stream; chunks anchored on utterances and grown by visual similarity; packing
  into ~3 min segments; each segment finalized exactly once, only when the stream is
  far enough past its end that nothing can change it (the paper's watermark rule);
  global-id merge (cos ≥ 0.85) with handover across sessions
* `admission:an_chunk_emitter` — the feeder's AN-chunk admission path: each
  utterance's raw timestamps are split into runs of contiguous 5fps steps; a chunk is
  emitted in one piece only after all of its frames have streamed past; duplicates
  are dropped. **No similarity expansion** — the original feeder discards the
  segmenter's expanded chunks; this port preserves that.
* `embedder:frozen_resnext50` (the frozen ImageNet encoder the segmenter embeds
  frames with), `encoder:text_bow` (CVCL bag-of-words text encoder), `view:cvcl_aug`
  (matches an_transform using a radius blur), `view:babycl_ms_aug` (the
  augmentation segments EFFECTIVELY receive: center-crop at gather + strong SimCLR
  at train, fused into one pipeline)

Everything else is the core toolbox (saycam/babyview adapters,
ntxent/tcl/clip_infonce, simclr views, resnext50, spill-routed tables,
solver/driver) plus `benchmarks/labeled_s_4afc`.

Run:

```bash
python -m egolearn.train --config-dir projects/babycl/configs --config-name saycam_default dry_run=true
python -m egolearn.train --config-dir projects/babycl/configs --config-name saycam_default
```

## IID training

`saycam_iid` uses the full BabyCL model, losses, AdamW groups, and augmentations
with all selected sessions available before training starts. Supply the offline
storyboard JSON as well as the frame, transcript, and vocabulary paths:

```bash
python -m egolearn.train --config-dir projects/babycl/configs --config-name saycam_iid \
  stream.root_dir="$DATA_ROOT" stream.transcript_json="$TRANSCRIPT_JSON" \
  tokenizer.vocab_json="$VOCAB_JSON" driver.batch.storyboard_json="$STORYBOARD_JSON"
```

Each batch contains three independently sampled storyboard segments, 32 frames
per segment with two visual views, and 32 utterance/frame pairs. Segment sampling
uses replacement and TCL uses labels 0, 1, 2 local to each batch, as in BabyCL.
The storyboard must cover every selected session and match its exact frame order;
its embedded data root is ignored in favor of `stream.root_dir`. Utterance chunks
use the streaming AN-chunk builder's timestamp alignment and deduplication.

Choose the epoch definition explicitly:

- `driver.batch.epoch_basis=utterances` (default): visit every utterance chunk once
  in shuffled order, dropping the incomplete final batch. Visual segments are
  drawn independently for each batch. This follows the original IID baseline.
- `driver.batch.epoch_basis=frame_budget`: run
  `floor(dataset_frame_count / (segments_per_batch * frames_per_segment + pairs_per_batch))`
  updates per epoch. Each update samples from the complete pools; frames and
  utterance chunks can recur across updates. At batch size 128, this matches an
  image-only IID epoch's number of image draws, not unique frame coverage.


Published ablations as config diffs:

| Paper ablation | Diff |
|---|---|
| buffer policy (FIFO+FIFO / FIFO+random / single reservoir) | `memories.*.tiers.long.update`, or merge tiers |
| buffer size (2%/10%, 5%/25%) | tier `capacity` values |
| multimodal fraction (1/4, 1/2, 3/4) | slot `n` values (3+32 / 2+64 / 1+96) |
| clip-importance weighting | `use_sample_weights: true` (weights flow via `weight.sample`) |
| resampling k=4 vs k=8 | `driver.rate.steps_per_admit` |
| CL-CVCL baseline | drop segments slot + ntxent/tcl; pairs-only recipe |
| one-pass CVCL | additionally `rate.steps_per_admit: 1` |
| BabyView | `stream: babyview`, `tokenizer.normalize: true`, portrait pre-crop view |
 