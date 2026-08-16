# Streaming, data, and modeling design notes

Contracts for code arriving via triage (see ../roadmap.md). Each is a cleanup of
something that already exists in memory-storyboard, lifelong-memory, poodle, or
midway-network — not a new invention.

> **Status (2026-08):** the streaming/replay/modeling side of these contracts is
> now implemented — see [architecture.md](architecture.md) for the shipped
> interfaces. Where the implementation refines a sketch below (e.g. labels are
> revised through a batch-time LabelStore lookup instead of shared-memory
> arrays, so parallel loader workers never need to see labels at all), the
> architecture doc is the current reference. §5 text interfaces and §6 QA evals
> remain design-only.

## Access-pattern premise

The primary regime is a temporally ordered stream: an ingest head advances once,
and replay revisits already-seen frames at random. Poodle/midway readers instead
target shuffled sampling of random pairs from mp4s — their seek ladders and
chunk-permutation plans solve a problem this regime doesn't have. Baseline storage
is pre-extracted frames, which makes sequential ingest, random replay, and
shuffled iid baselines all plain file reads. On-the-fly mp4 decoding is deferred
until a project needs shuffled training at a scale where extraction is
impractical.

## Core contracts

1. **StreamIndex** — a deterministic, temporally ordered sequence of typed
   records `{source_id, t, modality, locator}`. Frames dominate; captions,
   transcripts, and in-stream QA items interleave on the same timeline.
   Ordering is explicit and sorted, fps subsampling is a parameter, and
   cross-modal alignment is a time-window join. The index never shuffles —
   what to revisit is replay's job.
2. **StreamingDataset** — map-style over a StreamIndex, returning view tensors
   plus a mutable label. Labels live in shared memory behind `set_labels`, so
   online pseudo-labeling reaches worker processes mid-stream. Dataset quirks
   such as SAYCam's inverted orientation are dataset properties, not
   `if dataset == ...` branches. Corrupt frames: log and substitute a random
   other index.
3. **Frame extraction & IO** — an offline extraction CLI (ffmpeg: parallel,
   resumable, rotation corrected at extraction time via ffprobe), then plain
   frame-directory reads. One metadata schema: `{path, num_frames, fps,
   rotation}`.
4. **Replay & memory** — index replay is a retention policy (FIFO, reservoir,
   MinRed, balanced, ...) composed with a sampling policy (uniform, recency- or
   hardness-weighted), both on an index set, seeded, behind the `Sampler`
   surface the trainer consumes. Batch ratios are trainer config. Beyond-index
   memory — generative, latent, caption, KV-cache — is not forced under the
   index abstraction: training-time providers expose `sample(n) → batch`,
   query-time memories expose `retrieve(query, k)`.
5. **Modeling — two interfaces.** The split is by interface, not by whether the
   model learns. Any local model may be trained or continually adapted on the
   stream; API-served models are the only necessarily frozen case.
   - *Embedding*: downstream code depends only on `encode(frames) → features`.
     Methods compose backbone, projectors, and losses on top; ablation grids
     are configs, not classes.
   - *Text*: QA and memory depend only on `caption(frames) → text` and
     `answer(question, context) → text` — local, adapted, or API. API calls go
     through one client registry with retries, cost accounting, env-var keys,
     and a disk cache keyed on model version + prompt hash; caching is valid
     only while weights are fixed.
   - A jointly learned vision+language model exposes both interfaces.
   - IO convention: the data layer yields uint8 `(T, H, W, C)` frames with
     timestamps and text records as `(t, str)`; transforms produce encoder
     tensors; VLM adapters take raw frames and text.
6. **Evals** — readout evals consume the embedding interface; QA evals the text
   interface. In-stream QA is scored time-causally: an answer at time `t` uses
   only records seen before `t`. Split files are explicit and versioned;
   hyperparameters are selected on validation, never test; outputs are parsed
   as JSON, never `eval()`.
7. **Reproducibility** — everything seeded; checkpoints are state_dicts plus
   config; trainers resume.

## Deferred by default

Reversible triage defaults, revisited whenever a project shows the need:

- On-the-fly mp4 decoding. Random access is already free over extracted frames —
  iid baselines run that way. Decoding from mp4s matters only when extraction is
  impractical.
- Plugin registries — hydra `_target_` instantiation covers it.
- Streaming-pipeline frameworks — adopt only against a measured bottleneck.

## Port triage findings (2026-08-02)

Promote when needed: temporal segmentation core; replay buffer family; SSL
losses; guaranteed-overlap two-crop; extraction CLI, ffprobe rotation, metadata
cache; LLM provider switch and prompt-template contract.

Fix at port time, don't replicate: dead imports in the main storyboard trainer;
unsorted stream order; test-set model selection in kNN/SVM readout; missing
seeding; lifelong-memory's `end=3` default, swallowed caption-merge exception,
and broken rate-limit handling.

Leave behind: per-ablation model classes; seven trainer copies; apex dependency;
GPU-burning dummy ops; `sys.path` hacks; hardcoded cluster paths; shuffled-access
mp4 reader machinery.

## Licensing rules

egolearn is MIT. PooDLe/Midway are GPL-3.0 by election, not inheritance — their
vendored code is BSD/Apache/MIT — so the lab relicenses its own pieces here with
co-author OK.

- Lab-authored code: relicense to MIT, no header needed.
- Third-party-derived permissive code keeps its attribution header: mmcv ResNet
  and DINO ViT/utils are Apache-2.0; RAFT and the pytorch-examples probe
  utilities are BSD-3.
- Never vendor GPL or CC-BY-NC code. PooDLe's `misc.py`, `lars.py`, and
  `lr_sched.py` are MAE-lineage CC-BY-NC — use the DINO equivalents instead.
