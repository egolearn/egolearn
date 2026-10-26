# GRASP

**Task:** violation-of-expectation (VoE) on synthetic physics videos, scored
the same way as [IntPhys 2](intphys2.md). A video model that has learned
intuitive physics should find a physically implausible video more surprising
than its plausible twin. Surprise is the model's own prediction error; no
classifier is trained. The headlines are **pairwise accuracy** (`voe_acc`) and
the pairing-free **AUROC** (`voe_auroc`); both are higher-is-better with chance
at 50%.

The benchmark is Jassim et al., [*GRASP: A novel benchmark for evaluating
language GRounding And Situated Physics understanding in multimodal language
models*](https://arxiv.org/abs/2311.09048) (IJCAI 2024). We evaluate its
Level 2, the intuitive-physics level: 16 tests over eight concepts (collision,
continuity, gravity, inertia, object permanence, solidity, support,
unchangeableness), each rendered as 128 plausible and 128 implausible 10 s
Unity videos at 50 fps. That is 4,096 videos and 2,048 pairs. Level 1 (language
grounding) is not used.

| Test | Concepts |
| --- | --- |
| `Collision` | collision |
| `Continuity` | continuity |
| `Gravity` | gravity |
| `GravityContinuity` | gravity, continuity |
| `GravityInertia`, `GravityInertia2` | gravity, inertia |
| `GravitySupport` | gravity, support |
| `Inertia`, `Inertia2` | inertia |
| `ObjectPermanence`, `ObjectPermanence2`, `ObjectPermanence3` | object permanence |
| `SolidityContinuity`, `SolidityContinuity2` | solidity, continuity |
| `Unchangeableness`, `Unchangeableness2` | unchangeableness |

## Downloading the data

GRASP is a single-video yes/no task upstream, with no pair metadata. The videos
are linked from the [upstream repository](https://github.com/i-machine-think/grasp)
(the code is MIT; the videos state no license of their own), and EgoLearn never
downloads them. Fetch `videos.zip` (4.6 GB, both levels) from the link in the
upstream README and unpack it:

```bash
unzip videos.zip 'level2/*' -d /datasets/grasp
```

This leaves one `level2/P_<Test>/<n>.mp4` folder per test for the plausible
videos and an `IP_<Test>/` twin folder for the implausible ones, numbered
`0.mp4` to `127.mp4`.

## Pairing and the manifest

The upstream generator seeds each video with its index, so `P_<Test>/<n>.mp4`
and `IP_<Test>/<n>.mp4` share the same randomized scene and differ only in the
physics violation. Garrido et al.,
[*Intuitive physics understanding emerges from self-supervised pretraining on
natural videos*](https://arxiv.org/abs/2502.11831) (2025), pair the videos this
way and report pairwise accuracy; we follow that convention. Because the pairing
is a property of the generator rather than an authored contract, `prepare`
records, for every pair, whether the two videos open on an identical first
frame, and the probe reports the count of mismatches as `first_frame_mismatches`
in its metadata. Treat a non-zero count as a reason to look at the data.

Build the pair manifest once after unpacking (needs the `video` extra for
PyAV):

```bash
egl data prepare grasp --data_root /datasets/grasp/level2 \
  --output /datasets/grasp/manifest.json
```

The manifest lists the 16 tests with their concepts, the 2,048 pairs, and every
video's SHA-256 and frame count. It holds no pixels. The command prints the
pairs per test and the number of first-frame mismatches, and it never overwrites
an existing output. An unexpected `P_<Test>` folder name, a missing `IP_` twin
folder, or differing file names between twins are errors; check the unpacked
tree against the table above.

## Scoring contract

Decoding, views, context lengths, and surprise are exactly those of
[IntPhys 2](intphys2.md#scoring-contract): each video is sampled uniformly to
`num_frames` frames, each context length C in `context_frames` scores the first C
frames as context and the rest as the V-JEPA prediction target, and all lengths
share one decode and one target-encoder pass. The manifest records every
video's frame count, so unequal twins are visible before scoring.

`voe_acc_ctx<C>` scores each context length alone; `voe_acc` averages each
video's surprise over the context lengths and then compares once per pair.
`voe_auroc` ranks every selected video by that pooled surprise with no pairing:
it is the probability that an implausible video outranks a plausible one, ties
counting one half. A model can score well on `voe_acc` through pair-level
artifacts (both videos of a pair being hard or easy together); the AUROC is the
single-video check against that.

Every pair also contributes to its test and each of its concepts, so the probe
reports `voe_acc_test_<test>`, `voe_auroc_test_<test>`,
`voe_acc_concept_<concept>` and `voe_auroc_concept_<concept>`, all from the
pooled surprise. Test labels are the snake-case folder names
(`gravity_inertia2`); a two-concept test counts toward both concepts, as in
Garrido et al.

The numbers are not comparable to the GRASP paper's tables, which score
multimodal language models on a yes/no question (chance 50%, humans about 80%),
nor to Garrido et al., who slide 16-frame windows over each video and pick the
best context length per property. We score each clip once with fixed contexts.

## Registered evaluation probe

`probe:grasp` requires `video_representation` and `prediction` and reaches the
model only by name (`context`, `target`, `predictor`), exactly like
`probe:intphys2`. It runs during training through `probes:` or on a checkpoint
through the standalone evaluator.

Merge this fragment into an evaluation config (set the dataset root):

```yaml
_imports_:
  - egolearn.evaluation.benchmarks.grasp
benchmarks:
  - _target_: "probe:grasp"
    name: grasp
    manifest: /datasets/grasp/manifest.json
    data_root: /datasets/grasp/level2
    tests: null
    verify_videos: false
    num_frames: 64
    context_frames: [2, 4, 6, 8, 10, 12, 14, 16]
    distance: l1
    view: resize
    size: 224
```

The values shown are the defaults. `manifest` and `data_root` are path inputs,
so run records pin them. `tests` restricts the run to a list of test names from
the table (unknown names fail before any video is read); the grouped metrics
then cover only those tests, while `voe_acc` and `voe_auroc` pool the selected
pairs. `verify_videos: true` re-hashes every selected video against the manifest
before scoring, which reads the whole 4.6 GB; by default the probe only checks
that the files exist. The metadata recorded with the results lists `n_pairs`,
the selected `tests`, `n_tests`, and `first_frame_mismatches`.

Each full run decodes all 4,096 videos one after another, four times as many as
IntPhys 2, so schedule it sparingly inside training or restrict `tests` there.
