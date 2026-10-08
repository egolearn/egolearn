# V-JEPA 2 on SAYCam and BabyView

Status: finalized after two independent review rounds before implementation.

## Scope and scientific contract

Issue #39 requests a trained V-JEPA model, W&B loss/wall-time/throughput,
IntPhys 2 accuracy against a random model, and reproducible run records.
The user additionally requests BabyView and explicitly selected initial runs
from scratch. Implement the V-JEPA 2 pretraining architecture/objective with
small, clearly labeled experimental settings; do not claim reproduction of
Meta's internet-scale training, V-JEPA 2-AC, or V-JEPA 2.1.

Main already has a useful V-JEPA implementation from #8. Reuse its contracts
and preserve the behavior and state-dict layout of existing default configs.
Reference: [Meta V-JEPA 2](https://github.com/facebookresearch/vjepa2), inspected
at `204698b45b3712590f06245fbfba32d3be539812`, specifically
`src/models/{vision_transformer,predictor}.py`, `src/models/utils/modules.py`,
`src/masks/multiseq_multiblock3d.py`, and the ViT-L 16-frame pretraining config.
This revision includes newer methods too; only the original V-JEPA 2 path is
in scope. Retain Meta attribution/license for any adapted reference logic.

## Reuse and pending PRs

| Existing component | Use |
| --- | --- |
| `SaycamStream`, `BabyViewStream` | Existing frame adapters, ordered sessions, locators, dataset metadata |
| `Loader`, `ViewBank`, `clip_rrc` | Seeded temporal-consistent clip augmentation, worker pool |
| `OfflineDriver` | Shuffled complete clip windows, resumable epoch order, prefetch |
| `LockstepDriver` | Existing interface remains compatible; chronological clip admission is follow-up scope |
| `ComposedModel`, model builder, registry | Named context/target encoders and predictor; portable metadata |
| `VJEPA` loss, mask infrastructure | L1 masked target prediction, frozen full-clip target, EMA |
| Solver, hooks, snapshots, evaluation API | AdamW, warmup/cosine, W&B, recovery and standalone scoring |
| #37 `3a7add8` | IntPhys 2 verified pair metadata and shared VoE scoring |
| #36 `680c6dc`, #42 `ed98423`, #49 `6669a45` | Immutable Slurm submission and automatic train/eval records; #49 contains #36/#42 |
| #40 `f2f3c9d` | Preserve live construction metadata on snapshot resume |
| #34 `bd46094` | Restricted checkpoint loading and evaluation provenance |

Use a branch from current main `a63c140`, isolated from unrelated dirty work.
Merge required dependency branch heads locally without rewriting their history;
do not merge their GitHub PRs. Resolve shared docs/checkpoint conflicts by
retaining both behaviors and run their focused tests. Avoid merging unrelated
BabyCL, MAE or slot-method branches. Reuse the isolated `saycam.py` adapter
change from `feat/mae` at `1c8c4a2` (session_glob, natural_order, worker cache
reset) rather than reimplementing its nested-directory support. Existing MAE cluster artifacts can identify
available read-only dataset mounts and a working image, without copying its
training loop. #41 safetensors is optional, not a training prerequisite.

## Architecture and objective changes

1. Register explicit `encoder:vjepa2`, `predictor:vjepa2`, `loss:vjepa2` in
   `methods/vjepa2/`, loaded through `_imports_`. Subclass existing encoder,
   predictor and loss, extending shared primitives with optional settings.
2. Reference mode uses tubelet/patch embedding, GELU MLP, biased QKV,
   LayerNorm epsilon 1e-6, truncated-normal initialization (std .02), zero
   biases, and output-projection rescaling by sqrt(2 * layer number).
3. Preserve the reference RoPE frequency expansion, including its documented
   frequency-repeat quirk, behind an explicit convention. Existing corrected
   adjacent-pair rotations remain the old default. Test against a small,
   independently expressed reference calculation, including masked positions.
   Cast rotation tables to Q/K dtype so bf16 attention is valid.
4. Predictor uses the reference two zero-initialized mask tokens. Meta selects
   the token by clip-length group (not mask kind). Our single 16-frame group
   trains token 0 for both mask kinds; evaluation also uses token 0. Store
   supported mask indices in construction config and reject an untrained index.
   Maintain token positions when concatenating context and target; absent
   causal masks, permutation of their order is mathematically immaterial.
   The raw reference predictor default of 1 is overridden by Meta's training
   wrapper; copying that default would evaluate an untrained token.
5. V-JEPA 2 masking uses int(H * W * scale) spatial area and samples a common block size per kind/batch, places 8 small
   (.15 spatial area) or 2 large (.7 area) blocks covering full clip duration,
   retries fully covered grids, then truncates sorted context/target indices
   separately to each batch's minimum. Derive randomness from seed, compose
   sequence/step, view and mask kind. Reject degenerate grids/specifications
   and bound retries with an actionable error rather than silently unmasking.
6. Reuse target initialization, normalized teacher tokens, mean L1 over kinds
   and views, and EMA. Validate matching context/target architecture before
   training using resolved encoder specs, not just tensor shapes, and synchronize
   before saving/evaluating the random baseline.
   Add optional loss-local bf16 autocast; teacher stays frozen/eval, FP32
   master weights. Defer activation checkpointing until a measured memory need.
7. Initial runs require grad_accumulation=1. This avoids existing driver EMA
   and partial-window snapshot semantics under accumulation; fail early for
   V-JEPA 2 when accumulation differs, through a loss attribute checked by
   pipeline validation. No broad driver rewrite or DDP in scope.
   Nonfinite training must fail via #49's record_fail_on_nonfinite contract.

## Configs and initial experiments

Add `projects/vjepa2/configs/` with shared model/training defaults, SAYCam,
BabyView, tiny synthetic smoke and IntPhys evaluation configs. Dataset roots,
orientations, W&B settings and site resources come from env/config only.
Use upright prepared frames; do not infer orientation from the dataset name.
Check local storage before selecting the exact input layout: existing default
BabyView adapter requires root/session/frame files, and supplied archives may
need a manifest or an adapter extension to represent nested sessions correctly.
Clips must never cross sessions or gaps in source timestamps/frame numbers.
Prefer complete 16-frame windows at the prepared 5 fps; record fps and stride.

Initial recipe: small ViT (dim 384, depth 12, heads 6, patch16, tubelet2),
predictor dim192/depth6/heads6, 16 frames at 128px, batch8, 1000 updates,
AdamW lr1e-4/weight_decay .04, step warmup100 then whole-progress cosine, EMA .998 constant (short-run adaptation),
bf16 on CUDA, FP32 CPU smoke. These are initial egocentric experiments,
not Meta's ViT-L/256px/global-batch3072 schedule. Use seed0 on both datasets.
Include a config preset for reference ViT-L dimensions for later scaling.
Use offline clip sampling from each video source for these initial runs. The
existing fixed-window component is a labeler, not a clip emitter; a streaming
recipe with frame references would silently repeat single frames. A real clip
emitter and chronological replay experiments are follow-up scope, not part
of this initial rollout. Add an optional max_frame_gap to offline clip indexing
to split at timestamp discontinuities; enable it at 1.5/fps in both presets.

Use local loss JSONL plus W&B and automatic records. Prefer a small reusable
metrics hook for synchronized end-to-end clips/sec, frames/sec, elapsed time,
learning rate and finite loss (one row per logging interval), so both training
modes report the same quantities. Avoid presenting asynchronous CUDA stage
host timings as kernel durations. Preserve raw histories outside Git.

## Evaluation, rollout and acceptance

- Save a random baseline with target copied from context and the predictor
  still randomly initialized before any optimizer update. Evaluate this and
  final checkpoints with identical IntPhys config, resolution, contexts and
  metadata checksum, using #37. Report all voe_acc_ctx<C> and pooled voe_acc.
  Empirical random scores need not equal the theoretical 50% chance level.
- Test tiny CPU training, nonzero gradients, no target gradients, EMA, token
  positions, mask edge cases, mixed precision, checkpoint loading and exact
  interrupted/resumed training using an unchanged total schedule.
- Test source clip continuity and config wiring with synthetic frame files,
  including BabyView nesting if required. No human-subject frames in tests.
- Run full CPU suite, lint/format/type/lock/build checks. Ask Claude for design
  review and revise before implementation; ask again for code review and fix
  confirmed findings before dataset training. Keep review text outside Git.
- Submit immutable code only after checks/review. First run a short real-data
  GPU preflight on both sources; check loaded counts, finite gradients/loss,
  memory and measured throughput. Then launch the 1000-step runs with
  checkpoints and W&B. Never mutate launched snapshots.
- Produce project README instructions immediately; add measured results only
  when completed records exist, with code/config identity. Distinguish submitted,
  running, failed and complete runs; never substitute synthetic scores for real
  data. Queue delays/data access must be reported explicitly if they prevent
  completion in this session.

## Final protocol clarifications

- Reuse the small adapter-only diff from feat/mae; add strict filename parsing
  opt-in for these presets. Clip indexing reports frame/session/continuous-run,
  gap and discarded-frame counts, and splits on delta <= 0 or > 1.5/fps.
- Use a thin method entry point around build/fit/record_run, not a new trainer.
  Before fit, validate single-update accumulation and CUDA availability, copy
  context to target only for a fresh run, save random.pth in the existing
  restricted-load-compatible checkpoint format and link it to the active record.
  Resume loads the snapshot without resynchronizing its EMA target. Dry-run
  uses synthetic source overrides when real mounts are unavailable.
- Wire a shared optional metrics hook before W&B. Log interval wall time,
  clips/sec and frames/sec with CUDA synchronization at measurement boundaries.
  Loss diagnostics are separate from the loss-term dictionary, because drivers
  sum every returned loss value: per-kind L1, context/target counts, mean feature
  std over target tokens and over predictions. Abort preflight on nonfinite
  values, missing model/predictor gradients, static input clips, or zero feature
  spread. No claim that finite nonzero variance alone proves useful learning.
- IntPhys uses context=video, target=video_target, predictor=pred, mask_index=0,
  size128, num_frames16, context_frames=[2,4,6,8,10,12,14], distance=l1, and the
  pinned Main metadata. It samples 16 frames uniformly over each complete video;
  this has variable effective fps, not the training 5 fps. Report that temporal
  distribution change explicitly. Compare checkpoints under exactly this same
  protocol; it is neither the original 64-frame repo default nor the paper's
  sliding-window protocol. Validate divisibility and supported mask index.
- AdamW splits bias/norm from decay. Cosine schedule, .998 EMA, reduced model,
  128px and 1000 updates are explicitly short-run adaptations. FP32 reference
  parity is tested; BF16 frequency calculation can have rounding differences.
- One exact initial baseline and one trained model per dataset are the required
  first-run comparison. Multiple random seeds/bootstrap intervals are follow-up
  statistical work, not a reason to withhold these preliminary measurements.
- No measured results or raw reviews enter Git. The algorithm is published;
  reusable implementation, documented experimental configs, and tests belong
  on feat/vjepa2. Keep a local result report outside the checkout with records and
  W&B links; a public README contains commands and status, no unpublished numbers.
  No main merge or publication is authorized by this workflow.
- The implementation branch is explicitly stacked on #37/#49/#40/#34 and records
  the dependency identities. Cluster-specific paths stay in private launch env.

Preflight also verifies median source interval within 10% of 1/fps and a nonempty
clip index. Predictor default mask_index is 0. A CHANGELOG entry accompanies the
implementation. Shared metrics guard zero-duration intervals.

Implementation-review refinements: set warmup_frac=0 with warmup_min_steps=100
to avoid a zero-lr first update in the shared scheduler. Preflight selects the
actual next shuffled batch and measures identical adjacent pairs (only wholly
identical clips fail). Resume archives abandoned metric tails and retains a
canonical curve with cumulative retained-segment time. W&B failures receive a
nonzero exit code. Index metadata labels fps_source: SAYCam uses filename
timestamps; BabyView frame-number timing depends on declared extraction fps.
Confirm BabyView fps using preparation logs/frame-count-versus-duration evidence
before training; the median of derived timestamps alone is not independent proof.
IntPhys square resizing also differs from training random resized crops.
