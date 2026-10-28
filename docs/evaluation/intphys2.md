# IntPhys 2

The IntPhys 2 probe measures violation-of-expectation on synthetic physics
videos. It reports pairwise accuracy, with higher scores better and 50% chance
performance.

Scores from this probe are not directly comparable to the paper's tables.
The probe samples each whole video into one window, while the paper's
[prediction-based protocol](https://arxiv.org/html/2506.09849v1)
evaluates overlapping windows.

A video model that has learned intuitive physics should find an impossible
video more surprising than its matched possible twin. Surprise is the model's
prediction error, without a trained classifier.

The benchmark is Bordes et al.,
[*IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic
Environments*](https://arxiv.org/abs/2506.09849) (2025). We evaluate its Main
split: 253 scenes covering object permanence, immutability, spatio-temporal
continuity, and solidity, labeled Easy, Medium, or Hard (172 videos are
unlabeled) and filmed with fixed or moving cameras. Each scene has four videos,
forming two matched (possible, impossible) pairs typed `1_Possible`/`1_Impossible`
and `2_Possible`/`2_Impossible`: 1,012 videos and 506 pairs. The Debug and
HeldOut splits are not used.

## Downloading the data

The videos are released under CC-BY-NC 4.0 for evaluation use only, and EgoLearn
never downloads them. Download the Main split from Hugging Face at a pinned
revision (no login needed):

```bash
uvx --from huggingface_hub hf download facebook/IntPhys2 --repo-type dataset \
  --revision a077a2f94e25889016fc6e5983cf21e2ddc25fb2 --include "Main/*" \
  --local-dir /datasets/intphys2
```

Or download the official archive, which contains every split (1.8 GB):

```bash
curl -LO https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip
unzip IntPhys2.zip 'Main/*' -d /datasets/intphys2
```

Both give byte-identical `Main/metadata.csv` and 1,012 `Main/Videos/*.mp4` files.
Set `data_root` to `/datasets/intphys2/Main`. The probe checks `metadata.csv`
against its pinned SHA-256 and rejects any other copy, so every run scores the
same 506 pairs.

## Scoring contract

Each video is decoded with PyAV and sampled to `num_frames` frames spread
uniformly over its whole duration, first and last frames included. In 61 of the
506 pairs the two videos differ in length (by at most 5%), so their sampled
frames are close to, but not exactly, time-aligned. Every frame goes through the
registered evaluation `view`, which must be a per-image view such as `resize`
(the default: bicubic resize to `size` and ImageNet normalization) or
`center_crop`; the `clip_*` views expect clip tensors. The clip is scored as a
single window.

For each context length C in `context_frames`, the first C frames are the context
and the remaining frames the target. The context encoder sees only the context
tokens, and the predictor fills in the target tokens. Surprise is their mean
L1 distance to the frozen target encoder's layer-normalized tokens at the same
positions, matching the V-JEPA training objective. `distance: l2` uses the
mean squared error instead. Context lengths must be multiples of the encoder's
tubelet size. All context lengths share one decode and one target-encoder pass
per video.

A pair is correct when the impossible video is strictly more surprising than
the possible one. `voe_acc_ctx<C>` scores each context length alone. `voe_acc`
first averages each video's surprise over all context lengths, then compares
once per pair.

## Registered evaluation probe

`probe:intphys2` requires `video_representation` and `prediction`. It uses named
model components as in a `loss:vjepa` model. These are the context encoder
(`context`, default `video`), frozen target encoder (`target`, default
`video_target`), and predictor (`predictor`, default `pred`). It runs during
training through `probes:` or on a checkpoint through the standalone evaluator.

Save this standalone config as `intphys2_eval.yaml` and set the dataset root:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.intphys2
benchmarks:
  - _target_: "probe:intphys2"
    name: intphys2
    data_root: /datasets/intphys2/Main
    num_frames: 64
    context_frames: [2, 4, 6, 8, 10, 12, 14, 16]
    distance: l1
    view: resize
    size: 224
```

The values shown are the defaults, and `context_frames` also accepts a single
integer. Compose the config without opening the dataset, then evaluate your
checkpoint on a compatible GPU:

```bash
uv run egl eval intphys2_eval.yaml checkpoint=run/final.pth --cfg job
uv run egl eval intphys2_eval.yaml \
  checkpoint=run/final.pth device=cuda output_json=intphys2_results.json
```

The evaluator defaults to CPU if you omit `device=cuda`.

For training, use the same entry under the training config's `probes` list and
retain its registration module in `_imports_`. Metrics are `voe_acc` and one
`voe_acc_ctx<C>` per context length, and the evaluator records the full probe
configuration, including defaults, with the results. Before scoring, the probe
checks that all 1,012 videos exist.

Each run decodes all 1,012 videos one after another, and its duration depends on
the model, device, and storage. It pauses training when run as a probe, so
schedule it sparingly there.
