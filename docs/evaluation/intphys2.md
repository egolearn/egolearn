# IntPhys 2

IntPhys 2 tests whether a video prediction model is surprised by physically
impossible events. Each scene comes as matched possible and impossible videos.
A pair counts as correct when the model's prediction error, its surprise, is
higher on the impossible video, so no classifier is trained. The headlines are
pairwise accuracy `voe_acc` and AUROC `voe_auroc`, both with chance at 0.5.
Scores are not directly comparable to the paper's tables, because the sampling
differs (see [differences](#differences-from-the-reference-protocol)).

The benchmark is Bordes et al.,
[*IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments*](https://arxiv.org/abs/2506.09849)
(2025). The probe uses its Main split, 1,012 videos in 506 pairs that cover
object permanence, immutability, spatio-temporal continuity and solidity.

## Downloading the data

The videos are released under CC BY-NC 4.0 for evaluation use only, and
egolearn does not download them. Download the Main split from Hugging Face at
the pinned revision. You do not need a login:

```bash
export EGOLEARN_DATA=/path/to/data
uvx --from huggingface_hub hf download facebook/IntPhys2 --repo-type dataset \
  --revision a077a2f94e25889016fc6e5983cf21e2ddc25fb2 --include "Main/*" \
  --local-dir "$EGOLEARN_DATA/intphys2"
```

The [official archive](https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip)
holds the same Main split. The probe checks `Main/metadata.csv` against its
pinned SHA-256 and refuses any other copy, so every run scores the same pairs.

## Running the evaluation

The probe needs a model with a video context encoder, a frozen target encoder
and a predictor, as in a V-JEPA model. It finds them by the component names
`video`, `video_target` and `pred`, which the `context`, `target` and
`predictor` options can change. Save this config as `intphys2_eval.yaml`:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.intphys2
benchmarks:
  - _target_: "probe:intphys2"
    name: intphys2
    data_root: ${oc.env:EGOLEARN_DATA,data}/intphys2/Main
    num_frames: 64
    context_frames: [2, 4, 6, 8, 10, 12, 14, 16]
```

Evaluate a checkpoint on a GPU:

```bash
uv run egl eval intphys2_eval.yaml checkpoint=run/final.pth device=cuda \
  output_json=intphys2_results.json
```

The values shown for `num_frames` and `context_frames` are the defaults. You
can also put the same entry under `probes` in a training config. There, each
run decodes all 1,012 videos and pauses training, so schedule it sparingly.

## Metrics

| Metric | Meaning |
| --- | --- |
| `voe_acc` | Share of pairs in which the impossible video is more surprising, with surprise averaged over all context lengths |
| `voe_acc_ctx<C>` | The same, for context length C alone |
| `voe_auroc` | Probability that an impossible video is more surprising than a possible one, ignoring the pairing. Ties count half |

For a context length C, the model sees the first C sampled frames and predicts
the rest. Surprise is the mean L1 distance between the predicted tokens and the
frozen target encoder's layer-normalized tokens, which is the V-JEPA training
loss.

## Differences from the reference protocol

- Each video is sampled uniformly to `num_frames` frames over its whole length
  and scored as one window. The paper's prediction-based protocol evaluates
  overlapping windows instead.
- Only the Main split is used, not the Debug or HeldOut splits.
