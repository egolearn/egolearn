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

Trained seed-0 ViT-S/16 models are available in the private
[EgoLearn Hub collection](https://huggingface.co/collections/agentic-learning-ai-lab/egolearn-6abc0a0d87953889058b0317).
Each pinned artifact includes its model card, scientific configuration, training
loss history, evaluation results and checksums:

| Dataset | IID | Streaming |
| --- | --- | --- |
| SAYCam-S | [Checkpoint](https://huggingface.co/agentic-learning-ai-lab/egolearn-vjepa2-vits16-saycam-iid-seed0/tree/e6d997b1140d8e70f6a1745f22412b362e4f8f53) | [Checkpoint](https://huggingface.co/agentic-learning-ai-lab/egolearn-vjepa2-vits16-saycam-streaming-seed0/tree/a414d784ac5280146646d4a6b3de3a7afd5228b9) |
| BabyView | [Checkpoint](https://huggingface.co/agentic-learning-ai-lab/egolearn-vjepa2-vits16-babyview-iid-seed0/tree/1070cec79333fbc91c563757e1eddfd4b6f08b75) | [Checkpoint](https://huggingface.co/agentic-learning-ai-lab/egolearn-vjepa2-vits16-babyview-streaming-seed0/tree/ad3a21994b21eb5074c9adfb03cb837e344376c9) |

Download `model.pth` from the chosen revision with an authorized Hub account and
pass its local path as `checkpoint` to either evaluation command. These exports
preserve every context/EMA encoder and predictor tensor, plus construction and
training provenance, while omitting optimizer state. Evaluation records retain
the original `final.pth` hash; `training.json` maps it to the exported file hash.
The model cards document the single-seed results and protocol limitations.

The `image_evals` configuration runs Labeled-S and BabyView frozen linear
decoding and novel-object shape bias. Set `LABELED_S_ROOT`, `LABELED_S_SPLIT`
(the pinned canonical metadata), `BABYVIEW_LINEAR_MANIFEST`, and
`SHAPE_BIAS_MANIFEST`, then run:

```bash
uv run python -m egolearn.evaluate --config-dir projects/vjepa2/configs \
  --config-name image_evals checkpoint="$RUN_DIR/final.pth" \
  output_json="$RUN_DIR/trained-image-evals.json"
```

All four data environment variables are required by this recipe. Prepare
BabyView manifests with `python -m egolearn.evaluation.benchmarks.babyview_4afc.prepare`
and shape-bias stimuli with `python scripts/prepare_shape_bias.py` (see each
command's `--help`). The canonical Labeled-S split source and checksum are
defined in `egolearn.evaluation.benchmarks.labeled_s_linear.probe`.

Repeat with `random.pth` and a separate output path. This protocol freezes the
EMA `video_target` encoder, repeats each normalized image for 16 frames, and
mean-pools its output tokens. Images use 128px center crops for linear decoding
and whole-image resizing for shape bias. The canonical Labeled-S split and
prepared BabyView split are unchanged; both classifiers use train-only feature
standardization and the existing L-BFGS scorer (L2=0.001, 500 maximum iterations).
Shape bias uses all 256 stimuli and 57,600 triplets. This is an explicit
static-image transfer protocol, not a temporal-understanding benchmark or a
reproduction of published V-JEPA image-evaluation scores.

No dataset-scale results are committed here. Keep loss histories, checkpoints,
records, evaluation scores and a local results report outside Git; do not infer
results from synthetic tests or job submission. Report complete/failed/pending
status, commit, resolved config, input identities, W&B URL and measured timings.

The branch depends on #37 (IntPhys), #49 (run records, including #42's CLI),
and #34 (checkpoint/provenance). Submission (#36) and resume metadata (#40)
are merged into main. The design records the initial dependency commits;
merge main again when the remaining dependencies land. The nested-directory
adapter reuses the isolated change from `feat/mae`.

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


### Loader throughput and recovery

The shared loader reuses the packed/pinned batch interface from the pending MAE
work (`1c8c4a2`). `loader.packed=true` batches worker outputs into shared tensors;
`loader.pin_memory=true` collates a homogeneous slot into pinned CPU storage for
GPU transfer. Multiple heterogeneous slots retain the ordinary list path.

For streaming replay, `loader.cache_frames=128 loader.affinity=true` caches up to
128 decoded PIL frames **per worker**, before augmentation. Retained clips stay
on the same worker; new clips go to workers with the fewest live clip keys.
For the full FIFO recipe, allow at least `ceil(batch_size/workers) * clip_len`
frames per worker; a smaller cyclic cache may have almost no hits.
Ownership metadata retains only the current batch. The cache never stores
augmented views or future frames, and it is intentionally absent from snapshots.
A cold cache after resume changes throughput only. Caching and affinity require
positive worker counts; disable both for an in-process CPU smoke run. Frame count bounds cache
memory; choose the limit with source resolution and worker count in mind.

Allocate CPUs for the workers plus the training process, and benchmark with the
same model, data and batch recipe. For example, start with 16 workers, four parent
threads (`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`), packed/pinned batches, and enable
cache/affinity for streaming. IID usually benefits from additional workers and
has little short-term cache reuse. Measure sustained GPU utilization after worker
startup, alongside update time and loader timing; stage timings can overlap. Packed
`data.load` includes final collation, while the list path times collation in
`batch.compose`. Compare full update time when evaluating these options.

Operational loader settings can change when resuming the same runtime. A source
change requires a new output directory. After reviewing the source diff for
semantic equivalence, `egolearn.methods.vjepa2.migrate.fork_run(config, source,
source_sha256=old_digest, target_sha256=reviewed_digest, reason=reviewed_reason)` performs an explicit migration.
Pass an unresolved full-run config with a fresh `run_dir`, the source manifest's
package digest, the reviewed destination's package digest, and a reason describing the change. It verifies the
scientific config, complete inventory, dependency versions, deterministic random
baseline and resumable state, and selects the snapshot with the greatest saved
eligible step (an epoch snapshot can be newer than the rolling snapshot).
Epochs above the rolling step that predate the latest attempt are excluded;
earlier milestones are retained. Completed sources are rejected. It copies the
baseline, snapshot, previous epoch checkpoints and metrics; `migration.json`
records both manifests and checksums. The original run remains intact. This is
an explicit assertion of reviewed code equivalence, not a general compatibility
converter. The normal runner's runtime guard remains strict.

On continuation, metrics beyond the restored step are archived before the
canonical history is rewritten. Previous W&B attempts remain historical: use
attempt start steps to exclude their abandoned tails. A smaller operational
`hooks.snapshot_every` interval reduces work lost on abrupt termination.


### Independent runs in one allocation

For clusters that support GPU MPS, independent full runs can share a GPU while
keeping separate models, optimizers, RNG streams and checkpoints. Benchmark the
combined throughput and memory use, including the period after shorter runs finish.
GPU utilization is device-wide in this arrangement. Enable MPS through the cluster's
supported submission option and confirm that its server and pipe are visible inside
the container; the supervisor does not start an MPS daemon.

`python -m egolearn.commands.train_group specs.json --record group-attempt.json`
reads a JSON list. Each entry contains `config_dir`, `config_name`, `overrides`
(a list of Hydra overrides), `run_dir`, `log`, and an explicit parent `threads`
count. Paths and logs must be distinct; run directories may not be nested.
The supervisor builds each `run_dir` override, sets both OMP and MKL thread counts,
and appends a header and output to each child log. Operational preflight caps and
dry-run overrides are rejected. Slurm record names automatically include the job ID
and restart count, preserving records across automatic requeue. Hydra output is
isolated inside each run directory.

Migrate and verify checkpoints in a separate short job before launching a group.
The child bootstrap ignores USR1 during import and interpreter teardown; the full
runner temporarily installs its checkpoint handler. The supervisor detects each
new attempt file before forwarding a checkpoint request to that trainer's PID.
It never sends USR1 to a loader process group. One paused run or a supervisor USR1
pauses its live peers and returns 75. Any child failure pauses peers and fails the
group without requeue. Completed runs use the full runner's verified completion
guard when a group is resubmitted. A single bounded shutdown deadline covers both
startup readiness and checkpoint completion (default 480 seconds, shorter than the
Slurm launcher's 600-second warning). Timeout kills remaining child process groups
and returns failure. Abrupt scheduler termination still relies on atomic snapshots.

For a manual group pause, use `scancel --batch --signal=USR1 JOB_ID`; signal only
the batch supervisor, which forwards requests to ready trainers. Do not use
`--full`, which also reaches workers during import.
