# V-JEPA 2: initial egocentric video runs

Implementation of the pretraining method from Assran et al.,
[V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
(2025). This project trains from scratch on SAYCam-S or BabyView. It reuses
EgoLearn's data, optimization, checkpoint, W&B and standalone evaluation interfaces.
The [design](../../docs/design/vjepa2.md) records the architecture audit and dependency stack.

## Run

Prepared frames must be upright, sampled at 5 fps, and available locally under
`SAYCAM_ROOT` or `BABYVIEW_ROOT`. SAYCam uses `root/S_<recording>/*_<seconds>.jpg`;
BabyView uses `root/<child>/<recording>/frame_00000001.jpg` (override
`stream.session_glob='*'` for an already flattened source). No transcript is used.
The strict adapters reject malformed names. Clip indexing checks the median
frame interval and cuts at discontinuities; clips never cross recordings or gaps.
`fps_source` distinguishes SAYCam filename timestamps from BabyView times derived
from frame numbers and the declared fps. BabyView extraction rate must be verified
from preparation metadata (filter settings and frame count versus source duration);
frame numbers alone cannot prove it.
Session, frame, gap, clip and discarded-frame counts are logged. Incomplete tails
are dropped. This is offline clip training, not chronological continual learning.

```bash
uv sync --extra train
uv run python -m egolearn.methods.vjepa2.train \
  --config-dir projects/vjepa2/configs --config-name smoke
uv run python -m egolearn.methods.vjepa2.train \
  --config-dir projects/vjepa2/configs --config-name saycam
uv run python -m egolearn.methods.vjepa2.train \
  --config-dir projects/vjepa2/configs --config-name babyview
```

The smoke config uses only synthetic data. Use `dry_run=true` to validate it.
For a real GPU preflight, use a new output directory, `preflight=true`,
`driver.max_steps=8`, `hooks.log_every=1`, `hooks.checkpoint_every=8` and
`hooks.snapshot_every=8`. This checks the next shuffled batch for entirely identical-frame clips, finite
nonzero context and predictor gradients, a gradient-free teacher, and nonzero
feature spread. It records the fraction of identical adjacent frame pairs and
peak allocated GPU memory; partial duplication is measured, not rejected.
Then start the full config in another directory. GPU runs fail if CUDA is absent.

Training writes `random.pth` before the first update, copying context into the
teacher in the same process; `final.pth`, periodic checkpoints, resumable snapshots,
`metrics.jsonl`, and automatic run records follow. Use `resume=auto` with the same
absolute `run_dir` and unchanged scientific config/total update budget. The runner
preserves the EMA teacher on resume and refuses to overwrite an existing baseline
on a fresh run. Accumulation greater than one is rejected for this method.

W&B defaults to online logging; set `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE`
and credentials in the environment. Offline W&B runs can be synced later.
Loss, per-mask diagnostics, representation spread, learning rate, synchronized
clips/sec and frames/sec are logged every 25 updates. The first rate includes
startup/indexing; subsequent intervals include I/O and checkpoint overhead.
`metrics.jsonl` persists the loss curve even if online logging is unavailable.
On resume it removes replayed rows after the restored step, preserving the old
file as `metrics-abandoned-<id>.jsonl`. Elapsed time accumulates across retained
segments, excluding abandoned work; throughput covers each current interval.
Failures are marked failed in W&B as well as the automatic run record.

Use the existing immutable cluster launcher, with site resources and read-only
archive mounts supplied through your private environment:

```bash
egl submit --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 -- \
  python -m egolearn.methods.vjepa2.train \
  --config-dir projects/vjepa2/configs --config-name saycam \
  run_dir="$EGOLEARN_ARTIFACTS/vjepa2-saycam-seed0"
```

See [cluster setup](../../docs/cluster.md). Commit the reviewed implementation
before submission; never change archived code or the queued environment.

## Recipe and compatibility

The initial recipe is ViT-S/16 (384 wide, 12 blocks, 6 heads), a 192-wide,
6-block/6-head predictor, 16 frames at 128px, batch 8 and 1000 updates.
It uses AdamW lr 1e-4, decay .04 (bias/norm exempt), 100-update step warmup
followed by the shared progress cosine,
constant EMA .998, and BF16 forward with FP32 loss/weights. The reduced size,
resolution, small batch, EMA and short schedule are adaptations for initial runs;
the cosine uses whole-run progress after warmup (`warmup_frac: 0` avoids a
zero-lr first update). They do not reproduce Meta's training scale or reported
accuracy. `large.yaml`
provides reference ViT-L/predictor dimensions for future scaling only.

Explicit `encoder:vjepa2`, `predictor:vjepa2`, `loss:vjepa2` registrations preserve
existing `video_vit`/`vjepa` defaults and checkpoints. Reference initialization,
LayerNorm eps 1e-6, 3D rotary frequency expansion (including Meta's documented
frequency-repeat quirk), two zero mask tokens and block-size/truncation rules
follow [Meta's source](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).
The single clip-length group trains mask token **0** for both mask types; token 1
is reserved and rejected at evaluation. This follows the reference training
wrapper, rather than its raw predictor's default argument. Masks derive from the
run seed, step, view and kind; their RNG stream differs from Meta's data workers.
External Meta checkpoint key conversion, DDP, action conditioning, V-JEPA 2.1,
and activation checkpointing are outside this initial implementation.

## IntPhys 2

Set `INTPHYS2_ROOT` to the verified `Main` directory described in the
[IntPhys guide](../../docs/evaluation/intphys2.md). Evaluate `random.pth` and
`final.pth` with the same configuration and separate JSON output paths:

```bash
uv run python -m egolearn.evaluate --config-dir projects/vjepa2/configs \
  --config-name intphys2 checkpoint="$RUN_DIR/random.pth" output_json="$RUN_DIR/random-intphys2.json"
uv run python -m egolearn.evaluate --config-dir projects/vjepa2/configs \
  --config-name intphys2 checkpoint="$RUN_DIR/final.pth" output_json="$RUN_DIR/trained-intphys2.json"
```

This scores all 506 pairs at 128px using 16 uniformly sampled frames and context
lengths 2,4,6,8,10,12,14. The predictor's effective mask index is 0. Teacher outputs
are layer-normalized, matching training. Uniform whole-video sampling implies
variable effective fps, unlike the 5-fps training clips. Evaluation resizes
the full frame to a square; training uses random resized crops. Report `voe_acc` and every
`voe_acc_ctx<C>` together with this protocol: it differs from both the repository's
64-frame default and the paper's sliding-window protocol. The empirical random
baseline need not equal 50%. One seed and one initialization are preliminary;
no uncertainty estimate or statistically significant improvement is claimed.

## Results and dependencies

No dataset-scale results are committed here. Keep loss histories, checkpoints,
records, evaluation scores and a local results report outside Git; do not infer
results from synthetic tests or job submission. Report complete/failed/pending
status, commit, resolved config, input identities, W&B URL and measured timings.

The initial branch integrates #37 (IntPhys), #49 (run records, including #36/#42
submission and CLI), #40 (resume metadata), and #34 (checkpoint/provenance).
Their commit identities are in the design. Reconcile with main when those PRs
land. The nested-directory adapter reuses the isolated change from `feat/mae`.

## Full IID and streaming runs

The `full_saycam_iid`, `full_saycam_streaming`, `full_babyview_iid`, and
`full_babyview_streaming` recipes train the validated small architecture from
scratch with batches of 32 complete clips. Streaming fills a 32-clip FIFO,
then admits two new clips per update. Clips contain 16 frames at the prepared
5 fps and never cross recordings or timestamp gaps. The FIFO spans recordings;
it provides local replay without access to future clips. Incomplete recording
and gap tails are counted and dropped. Every complete clip is admitted once.

IID shuffles the identical clip inventory each epoch. For N clips, batch B and
advance S, both modes run `ceil((N - B + S) / S)` updates, approximately B/S=16
IID passes. The inventory, discarded frames and exact update budget are saved
in the manifest. Set `run_control.expected_frames` and `expected_clips` from an
independent mount inventory to reject missing archives. Keep these dataset
counts and local output paths outside Git.

```bash
uv run python -m egolearn.methods.vjepa2.full_train \
  --config-dir projects/vjepa2/configs --config-name full_saycam_streaming
uv run python -m egolearn.methods.vjepa2.full_train \
  --config-dir projects/vjepa2/configs --config-name full_babyview_iid
```

Both modes use the same update-indexed LR: 1e-4, warm up for the larger of 100
updates and 5% of the budget, then cosine decay. AdamW decay is .04 and EMA is
constant .998, retaining the pilot's optimizer and teacher settings to isolate
data order. These remain small-model experimental settings, not a reproduction
of Meta's training recipe. Batch size or clip advance changes require matching
FIFO, recipe and warmup settings; the runner rejects inconsistencies.

Each output directory has an immutable scientific manifest, attempt configs,
JSONL metrics, status, an exact `random.pth`, rolling resumable snapshot, and
snapshots at matched IID epoch boundaries. `final.pth` is a restricted-load
compatible evaluation export written only after the budget and source counts
pass. Resume automatically restores `snapshot_latest.pth`, including optimizer,
EMA, memory, admission position and randomness. Recipe/runtime/inventory changes
require a new output directory; operational controls may change between attempts.
Intentional pauses produce attempt records with `status: failed` and
`error_type: PauseTraining` in the generic record format; `status.json` says
`paused`. They never produce a final export. W&B attempts share an experiment
group; their IDs and URLs are saved in status and attempt files.

`run_control.stop_after_steps=8 preflight=true` validates complete source spans,
reports static clips and duplicate frames, and checks gradients. Valid static
source clips remain in the training inventory. It trains eight updates under
the **full** schedule, saves a snapshot, and returns 76. Set the cap to 16 to check continuation, then to 0 to finish.
For independent validation, use a separate output directory. SIGUSR1 or
`max_seconds` (default 46 hours including startup) pauses at a completed update
and returns 75. The full schedule remains unchanged.

Slurm runs require an **absolute** output directory, for example
`run_dir="$EGOLEARN_ARTIFACTS/vjepa2-saycam-streaming-seed0"`. Use that same path
for manual resubmissions. Local runs may use relative directories.

For the immutable Slurm launcher, opt into `EGOLEARN_REQUEUE_ON_PAUSE=1`, request
`--requeue --signal=B:USR1@600 --time=48:00:00`, and run Python directly (or use
`exec python ...` in array commands so signals reach the trainer). Exit 75
requeues into a new allocation; failures and preflight exit 76 do not requeue.
Send manual USR1 only after the new `attempts/*.json` file appears for the
current attempt; a previous attempt may have left a stale `running` status.
Python has not installed its handler during interpreter/import startup. The scheduled signal
arrives near the allocation deadline. Use `--open-mode=append` to retain logs
across requeues. Do not restart inside an expiring allocation. Schedule IntPhys evaluation as an
`afterok` dependent job, using the same random/final protocol documented above.
See [the full-run design](../../docs/design/vjepa2-full.md).
