# BabyCL / BabyCLIP (reproduction project)

Reproduces the BabyCLIP default run (continual visual + verbal learning on
SAYCam, single chronological pass).

**This project owns the paper's contribution, as 1:1 ports** (`components/`; the
config's `_imports_` list makes them register themselves on import):

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
  frames with), `trunk:text_bow` (CVCL bag-of-words text encoder), `view:cvcl_aug`
  (an_transform 1:1, incl. the PIL radius blur), `view:babycl_ms_aug` (the
  augmentation segments EFFECTIVELY receive: center-crop at gather + strong SimCLR
  at train, fused into one pipeline)

Everything else is the core toolbox (saycam/babyview adapters,
ntxent/tcl/clip_infonce, simclr views, resnext50, spill-routed tables,
solver/driver) plus `benchmarks/labeled_s_4afc`.

Run:

```bash
python tools/train.py --config projects/babycl/configs/saycam_default.yaml --dry-run
python tools/train.py --config projects/babycl/configs/saycam_default.yaml
```

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
 