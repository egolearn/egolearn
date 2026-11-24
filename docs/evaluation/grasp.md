# GRASP

GRASP Level 2 tests intuitive physics with the same violation-of-expectation
scoring as [IntPhys 2](intphys2.md). A video prediction model should be more
surprised by a physically implausible video than by its plausible twin, and
no classifier is trained. The headlines are pairwise accuracy `voe_acc` and
AUROC `voe_auroc`, both with chance at 0.5. GRASP itself asks language models
yes/no questions, so these scores are not comparable to the GRASP paper's
tables (see [differences](#differences-from-the-reference-protocol)).

The benchmark is Jassim et al.,
[*GRASP: A novel benchmark for evaluating language GRounding And Situated Physics understanding in multimodal language models*](https://arxiv.org/abs/2311.09048)
(IJCAI 2024). Level 2 has 16 tests, each with 128 plausible and 128
implausible Unity videos, which gives 2,048 pairs. The tests cover eight
concepts: collision, continuity, gravity, inertia, object permanence, solidity,
support and unchangeableness. A test is named after the concepts it combines,
such as `Gravity`, `GravitySupport` or `SolidityContinuity2`.

## Downloading the data

The videos are linked from the
[upstream repository](https://github.com/i-machine-think/grasp), whose code is
MIT licensed. The videos state no license of their own, and egolearn does not
download them. Fetch `videos.zip` (about 4.6 GB, both levels) from the link in
the upstream README, then unpack Level 2 and build the pair manifest:

```bash
export EGOLEARN_DATA=/path/to/data
unzip videos.zip 'level2/*' -d "$EGOLEARN_DATA/grasp"
uv run egl data prepare grasp --data_root "$EGOLEARN_DATA/grasp/level2" \
  --output "$EGOLEARN_DATA/grasp/manifest.json"
```

Preparation needs the `train` extra. The manifest pairs each plausible video
`P_<Test>/<n>.mp4` with the implausible `IP_<Test>/<n>.mp4` and records every
video's SHA-256 and frame count. It also prints how many pairs differ on
their first frame, and a nonzero count is a reason to inspect the data.

## Running the evaluation

The model needs the same components as for IntPhys 2, a video context encoder,
a frozen target encoder and a predictor. Save this config as
`grasp_eval.yaml`:

```yaml
defaults:
  - /evaluate
  - _self_

_imports_:
  - egolearn.evaluation.benchmarks.grasp
benchmarks:
  - _target_: "probe:grasp"
    name: grasp
    manifest: ${oc.env:EGOLEARN_DATA,data}/grasp/manifest.json
    data_root: ${oc.env:EGOLEARN_DATA,data}/grasp/level2
    tests: null
    verify_videos: false
```

```bash
uv run egl eval grasp_eval.yaml checkpoint=run/final.pth device=cuda \
  output_json=grasp_results.json
```

`tests` restricts the run to a list of test names, which are the upstream
folder names without their `P_` prefix, and
`verify_videos: true` rehashes every selected video against the manifest before
scoring. Frame sampling, context lengths and the other options are those of
[IntPhys 2](intphys2.md#running-the-evaluation). A full run decodes all 4,096
videos, so restrict `tests` when you use GRASP as a training probe.

## Metrics

| Metric | Meaning |
| --- | --- |
| `voe_acc`, `voe_acc_ctx<C>`, `voe_auroc` | As for [IntPhys 2](intphys2.md#metrics), over the selected pairs |
| `voe_acc_test_<test>`, `voe_auroc_test_<test>` | Per test, with the test name in snake case, such as `gravity_inertia2` |
| `voe_acc_concept_<concept>`, `voe_auroc_concept_<concept>` | Per concept. A two-concept test counts toward both |

Check `voe_auroc` next to `voe_acc`, since a model can score well on pairs
through pair-level artifacts that the unpaired AUROC does not reward.

## Differences from the reference protocol

- The GRASP paper scores multimodal language models on single-video yes/no
  questions. This benchmark pairs the videos and compares surprise instead,
  following Garrido et al.,
  [*Intuitive physics understanding emerges from self-supervised pretraining on natural videos*](https://arxiv.org/abs/2502.11831)
  (2025).
- Garrido et al. slide 16-frame windows over each video and pick the best
  context length for each property. Here each video is scored once with fixed
  context lengths, so the scores are not comparable to theirs either.
- Level 1, language grounding, is not used.
