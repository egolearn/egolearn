# Spatial relation categorization

**Task:** two-choice spatial relation matching in a frozen image representation.
Given an anchor scene, choose the candidate depicting the same spatial relation
in a new position. The other candidate depicts the opposite relation. Cosine
similarity between pooled image embeddings supplies the decision; no relation
classifier is trained. The headline is **spatial relation categorization accuracy**
(higher is better; chance is 50%).

This characterization follows the paper's Section 2 and Table 1: model similarity
judgments operationalize the infant familiarization/novelty paradigm, and the
authors evaluate accuracy and relative condition difficulty. Sections 3 and 5
apply the task to 2D stimuli and rendered 3D scenes respectively. Our benchmark
does not measure infant looking time or fit developmental trajectories.
[Published paper](https://www.cs.princeton.edu/~bl8144/papers/DavidsonEtAlCognition2024.pdf).

| Display name | Component key | Source provenance | Monitoring trials |
| --- | --- | --- | ---: |
| 2D abstract scenes | `abstract_2d` | Paper Experiment 1; stored as `exp1` | 768 |
| 3D rendered scenes | `rendered_3d` | Paper Experiment 3; stored as `exp3` | 512 |

Experiment numbers remain only as source provenance and existing file/manifest
identifiers. Display labels and probe metric keys use the descriptive names.
Renaming the frozen trial IDs or manifests would invalidate existing hashes and
comparisons, so integration maps them to these labels without rewriting data.

These are **labeled relation-matching tasks**, following Davidson, Orhan, and
Lake, [*Spatial relation categorization in infants and deep neural networks*
(Cognition, 2024)](https://doi.org/10.1016/j.cognition.2023.105690).
One candidate preserves the anchor's relation; the other changes it. For example,
a target below a reference should match a target in a new position below the
reference, rather than one above it. Both test candidates use the same target
object, even in trials where it differs from the anchor's target.

Report **accuracy**, not a shape-bias-style preference percentage. Shape versus
texture can be a competition between two legitimate similarity criteria; here,
the spatial relation supplies the answer key. Accuracy differences between
relations or object-change conditions can diagnose representational biases.
A correct choice alone does not establish abstract relational understanding:
appearance, geometry, architecture, and learned invariances can also influence it.

## Included datasets

| Dataset | Conditions | Prepared v1 size |
| --- | --- | --- |
| 2D abstract scenes | 3 renderers × 2 relations × 2 target conditions | 1,024 trials/cell; 12,288 trials |
| 3D rendered scenes | 2 relations × 2 target conditions | 2,048 trials/cell; 8,192 trials |

Relations are `above_below` and `between_outside`; target conditions are
`identical` and `different`. Exp. 1 renderers are `different_shapes`, `split_text`
(glyphs/shapes as visual objects; no language input), and `random_color`.
Experiment 2 (rotations) and Experiment 4 (containment) are outside this dataset.

Exp. 1 uses the external historical generator at angle zero, one familiarization
image, and same-relation target-distance ratio 1.0. Each generator is initialized
with seed 33 and called one trial at a time through `generate_single_triplet`.
This bypasses its cached batch API, which resets its RNG and reuses equal-sized
chunks. Images are exported as RGB PNGs with round-to-nearest uint8 conversion.
These are frozen regenerated stimuli, not a claim of pixel-exact paper recovery.
The manifest records renderer source hashes and dependency versions. OpenCV and
colorcet are required during preparation to avoid silent rendering fallbacks.

Exp. 3 uses all 256 configurations and eight target objects in each relation.
For every configuration, all eight identical-target trials are included. A
seeded cyclic derangement selects eight changed-target trials, with every object
appearing exactly once as anchor and once as test target. Thus each relation has
2,048 identical and 2,048 changed-target trials. The complete image inventory has
12,288 files; images are copied byte-for-byte and reused across trials.
This balanced selection is an explicitly versioned adaptation, not the original
paper's trial list. It does not use legacy model-result CSVs or their row ordering.

**Residual limitation:** above/below uses a Table reference and between/outside
uses a StorageRack. These source scenes cannot separate relation difficulty from
reference-object appearance. Report both conditions, and interpret their difference
with this limitation. Source-image hashes identify the supplied local renders;
the preparation script does not establish their original rendering provenance.

## On-disk layout and labels

```text
data/spatial_relations/v1/
  exp1/
    manifest.json
    trials.jsonl
    audit.json
    images/<renderer>/<relation>/<target_condition>/<index>_<role>.png
  exp3/
    manifest.json
    trials.jsonl
    audit.json
    images/<relation>/<original_filename>.png
```

All image and trial paths in a manifest are relative to that experiment directory.
Each trial has `id`, `group_id`, `split`, `relation`, `target_condition`, `renderer`,
`anchor`, two `candidates`, and a zero-based `correct_index`. Candidate order is
seeded and balanced within conditions. Exp. 3 also records the reference object,
anchor object, candidate object, and configuration index.

The manifest inventories image hashes, dimensions, modes, condition counts, and
the trial/audit-file hashes. Validation checks local paths, file integrity, unique trial
IDs, image coverage, all condition cells, and balanced conditions/candidate order.
Preparation writes to a temporary directory and publishes each experiment only
after validation; existing experiment directories are never overwritten.
Data and manifests live under ignored `data/`; assets are not committed or
redistributed with the library.

These are **test-only frozen-representation evaluations**. No fitted classifier
or train/validation split is needed. If a supervised readout is added later,
create a separate version with group-disjoint splits; do not randomly split
individual trials, which share images. Exp. 3 group IDs cover each
relation/reference/configuration, including all object pairings. Exp. 1 groups
share the trial index across renderers and target/relation conditions to keep
potentially shared geometry together.

The audit records repeated image triplets regardless of candidate order. Seeded
sampling can revisit a stimulus: the seed-33 Exp. 1 export contains 12,286 unique
triplets across 12,288 draws (trial indices 1 and 8 coincide in each of the two
`different_shapes`/`identical` relation conditions). These draws are retained,
not silently resampled. Exp. 3 has 8,192 unique triplets. Any future supervised
split must additionally merge groups connected by shared image hashes, including
these repeated draws; group IDs alone are not a sufficient leakage check.

## Preparing and checking data

Use an installed egolearn environment (`uv sync --extra train`). Set
`SPATIAL_GENERATOR_ROOT` to the external `simple-relational-reasoning` checkout
and `SPATIAL_RENDER_ROOT` to the existing `complex-relational-reasoning` directory.
The external renderer is imported in a subprocess whose working directory is the
supplied checkout; no source files are vendored or modified. Its old torchvision
`functional_tensor` import is mapped to the maintained internal implementation
inside the preparation process.

```bash
uv run --no-sync --with matplotlib --with scipy --with colorcet \
  --with opencv-python-headless python -m egolearn.cli data prepare spatial-relations \
  --generator-root "$SPATIAL_GENERATOR_ROOT" \
  --exp3-image-root "$SPATIAL_RENDER_ROOT" \
  --output-dir data/spatial_relations/v1

uv run egl data prepare spatial-relations \
  --output-dir data/spatial_relations/v1 --validate-only
```

Use `--experiment exp1` or `--experiment exp3` to prepare independently. Use a
separate output directory and `--n-per-condition 2` for an Exp. 1 smoke export.
The default full export has 1,024 trials per Exp. 1 condition. Preparation-only
renderer dependencies are supplied through `uv --with`; loading the prepared
PNGs does not require the historical generator or those dependencies. Recreating
identical pixels also requires matching the recorded source hashes, package
versions, and rendering environment; use the frozen assets for comparisons.

## Fixed monitoring subset

Keep `data/spatial_relations/v1/` for full evaluations and create the smaller
`data/spatial_relations/monitoring_v1/` for frequent training-time monitoring:

```bash
uv run egl data prepare spatial-monitoring \
  --full-root data/spatial_relations/v1 \
  --output-dir data/spatial_relations/monitoring_v1 --seed 33

uv run egl data prepare spatial-relations \
  --output-dir data/spatial_relations/monitoring_v1 --validate-only
```

| Component | Selection | Trials | Images |
| --- | --- | --- | --- |
| 2D abstract scenes | 64 per renderer × relation × target condition | 768 | 2,304 |
| 3D rendered scenes | 16 configuration IDs shared across both relations | 512 | 768 |
| Total | | 1,280 | 3,072 |

Exp. 1 samples 32 trials at each correct candidate index within each condition.
The source rows are sorted before seeded sampling, so input-file ordering cannot
change membership. Exp. 3 retains every trial for the selected configurations:
all eight objects, both target conditions, and both relations. This preserves its
balanced anchor/test object usage and candidate order. Full-set trial IDs,
labels, group IDs, and image bytes are preserved; no stimuli are regenerated.

Each experiment has the same portable manifest/trials/images/audit layout as the
full dataset. Its source metadata identifies the monitoring selection protocol,
seed, selected configurations, and parent manifest/trial hashes. Images are
copied, so the subset can be moved independently. Both parent datasets and all
subset files are checked; the output directory is published only after both
experiments validate. An existing output directory is never overwritten.

The monitoring subset overlaps the full evaluation set by design. It is an
evaluation view of the full data, not classifier-training data or an independent
held-out validation split. If monitoring scores guide model selection, the full
score includes those monitored examples and should not be described as entirely
untouched holdout performance. Keep this subset fixed across checkpoints.

## Scoring contract

Use the same deterministic image preprocessing for every member of a trial.
Avoid random crops, flips, or augmentations that change or remove the relation.
Encode each unique image once per evaluated checkpoint. Compare cosine similarity
between the anchor representation and each candidate; the answer key is
`correct_index`. For ties, use half-credit and report the tie rate and tolerance
explicitly. The signed cosine margin is similarity to the correct candidate minus
similarity to the foil. Report accuracy and mean margin per relation × target
condition (and renderer for 2D scenes). Keep component scores alongside the
balanced headline defined below.

Half-credit ties are a new explicit convention, not numerical equivalence to the
legacy scorer's inconsistent tie rules. For uncertainty estimates, resample
`group_id` blocks rather than treating shared-image trials as independent.
Untrained-model and simple image-similarity controls help assess whether a score
reflects more than incidental visual structure.

## Headline score and diagnostics

Our integration defines this aggregation; it is not a composite score claimed by
the paper. Let A(d, r, t, s) be mean trial accuracy within scene domain d, relation
pair r, target condition t, and rendering style s. Score each component as the
equal mean of its cells, then average the two components equally:

```text
2D accuracy = mean of 12 cells (3 styles × 2 relation pairs × 2 target conditions)
3D accuracy = mean of  4 cells (1 style  × 2 relation pairs × 2 target conditions)
Spatial relation categorization accuracy = (2D accuracy + 3D accuracy) / 2
```

Each relation pair and each target condition receives half the weight within a
component. Each 2D renderer receives one third. Thus each 2D cell has headline
weight 1/24 and each 3D cell has weight 1/8. Do not pool all trials (which gives
2D 60% weight in both current profiles) or all cells (which gives it 75%). The
same formula applies to monitoring and full evaluation. Missing domains/cells
are errors, not grounds for silently changing weights. Aggregate metrics are
fractions in [0, 1]; multiply by 100 for display. No chance correction is applied.

The probe retains component, relation-pair, target-condition, and cell accuracies.
`same_object` means the target identity stays the same from anchor to candidates;
`changed_object` means both candidates share a new target identity. Both
conditions change target position. The latter tests matching across object
identity changes within the supplied object inventory, not generalization to
held-out object categories. `object_change_gap` is same-object accuracy minus
changed-object accuracy within a component, in fractional accuracy units; it is
a diagnostic with no higher-is-better direction. Multiply by 100 for percentage
points. Overall relation/target diagnostics also weight the domains equally.
Mean cosine margins and tie rates are reported separately for each component.

High accuracy indicates that the chosen embedding/similarity rule favors spatial
relation preservation on these stimuli. It does not alone establish symbolic
reasoning, independence from appearance, or general spatial competence. A low
cosine score also does not prove that no relational information is decodable.
Neither the difference between relation pairs nor the object-change gap should
be called a standalone bias score. Retain the Table/StorageRack confound when
interpreting 3D relation differences. This version excludes rotations and
containment, and does not assess transfer by training on 2D and testing on 3D.

Our five-checkpoint subset check supports coarse overall monitoring. Individual
condition errors were substantially larger than overall errors, and one close
model ordering reversed. Use the full profile for final scores and condition
comparisons; a single headline is a convenience, not a substitute for those
diagnostics. Monitoring/full results must have distinct log namespaces.

## Registered evaluation probe

The `probe:spatial_relations` component requires `image_representation` and uses
`model.encode("image", inputs).pooled`, allowing any compatible frozen image
encoder. The evaluation hook places the model in evaluation mode. No external
DINO checkout is required by this probe. Its deterministic preprocessing matches
the subset comparison: RGB, bilinear resize to 224, ImageNet normalization,
autocast and TF32 disabled, and float64 cosine scoring. Use float32 model weights
to match the reference inference precision. It encodes each manifest image once
per evaluation, then reuses features across trials; embeddings are not reused
across checkpoints. Initialization validates all files and their hashes.

Merge this fragment into an evaluation config (set paths for your dataset root):

```yaml
_imports_:
  - egolearn.evaluation.benchmarks.spatial_relations
probes:
  - _target_: "probe:spatial_relations"
    name: spatial_relations_monitoring
    abstract_2d_manifest: data/spatial_relations/monitoring_v1/exp1/manifest.json
    rendered_3d_manifest: data/spatial_relations/monitoring_v1/exp3/manifest.json
    batch_size: 32
    tie_tolerance: 1.0e-6
    output_json: "runs/spatial_relations_monitoring_{step}.json"
```

For full evaluation, use the two manifests under `data/spatial_relations/v1/`,
the name `spatial_relations_full`, and a separate output path. Combining a full
component with a monitoring component is rejected. The main log metric is
`<name>/accuracy`; component metrics are `<name>/abstract_2d/accuracy` and
`<name>/rendered_3d/accuracy`. Artifacts include descriptive labels, component
source metadata and manifest hashes, preprocessing, tie tolerance, counts,
all diagnostics, and the versioned aggregation identifier
`equal_domains_equal_relation_target_renderer_cells_v1`.

## Comparing monitoring scores against full scores

The standalone comparison runner loads existing frozen DINO checkpoints from an
external checkout. Set `DINO_SOURCE_ROOT` and `SPATIAL_COMPARISON_CONFIG` to that
checkout and its model comparison YAML. The environment needs CUDA, the external
model dependencies, and an installed egolearn package.

```bash
uv run --no-sync python scripts/compare_spatial_subsets.py \
  --source-root "$DINO_SOURCE_ROOT" \
  --comparison-config "$SPATIAL_COMPARISON_CONFIG" \
  --full-root data/spatial_relations/v1 \
  --subset-root data/spatial_relations/monitoring_v1 \
  --output-dir outputs/spatial_monitoring_comparison
```

The YAML `models` mapping supplies `enabled`, `display_name`, `loader_type`,
`checkpoint_path`, and `checkpoint_key` (default `teacher`). DINOv2 entries use
`loader_type: dinov2` and a saved `config_file`; older DINO entries use
`loader_type: emin_vit`, `emin_root`, `arch`, and `patch_size`. Relative paths are
resolved against the external source root. Each backbone must load strictly;
training projection heads are excluded.

The runner validates both datasets, encodes full-set images once, and selects
monitoring predictions by unchanged trial IDs. It uses bilinear resize to 224,
ImageNet normalization, final normalized CLS features, float32 inference with
TF32 disabled, and float64 cosine scoring. The default tie tolerance is `1e-6`.
Outputs include per-trial NPZ scores, checkpoint/data hashes, overall and
condition-level CSV/JSON comparisons, and 2,000 alternative balanced subsets to
assess sampling error and model-order agreement. Exp. 3 sampling preserves shared
configuration blocks. Sampling ranges describe error against the finite full
dataset, not uncertainty for an unknown image population. Keep the original
subset fixed rather than selecting a seed based on model results.
