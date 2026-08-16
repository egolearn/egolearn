# BabyCL / BabyCLIP (reproduction project)

Reproduces the BabyCLIP default run (continual visual + verbal learning on
SAYCam, single chronological pass).

**This project owns the paper's contribution, as 1:1 ports** (`components/`,
registered on import via the config's `_imports_`):

* `refine.py` — **verbatim** segmentation math from mcl's
  `utils/online_ms_refine.py` (prefix sums, chunk expansion 2–12 s,
  gap cutting, representative picking)
* `admission:babycl_online_segmenter` — the OnlineSegmenter: sliding window,
  utterance-anchored expansion, ~3 min packing, watermark finalize-once,
  global-id merge (cos ≥ 0.85) with cross-session handover
* `admission:an_chunk_emitter` — the feeder's AN-chunk admission path: raw timestamp
  runs split on contiguous 5fps steps, atomic complete-visibility emission,
  dedup. **No similarity expansion** — the original feeder discards the
  segmenter's expanded chunks; this port preserves that.
* `embedder:frozen_resnext50` (ImageNet-frozen segmentation embedder),
  `trunk:text_bow` (CVCL BOW text encoder), `view:cvcl_aug` (an_transform
  1:1, incl. the PIL radius blur), `view:babycl_ms_aug` (the EFFECTIVE
  segment augmentation: center-crop at gather + strong SimCLR at train,
  fused into one pipeline)

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
 