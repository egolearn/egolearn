# Spatial relation categorization

This benchmark tests whether a frozen image encoder groups scenes by their
spatial relation. Each trial shows an anchor scene and two candidates with the
target moved to a new position. One candidate keeps the anchor's relation, such
as target below the reference, and the other shows the opposite relation. The
model picks the candidate whose pooled embedding is closer to the anchor's by
cosine similarity, without training a classifier. The headline is accuracy,
with chance at 0.5.

The stimuli come from Davidson, Orhan and Lake,
[*Spatial relation categorization in infants and deep neural networks*](https://doi.org/10.1016/j.cognition.2023.105690)
(Cognition, 2024). The benchmark has two components:

| Component | Source | Full set | Monitoring subset |
| --- | --- | ---: | ---: |
| 2D abstract scenes (`abstract_2d`) | Paper Experiment 1, three renderers | 12,288 trials | 768 trials |
| 3D rendered scenes (`rendered_3d`) | Paper Experiment 3 | 8,192 trials | 512 trials |

Both components test the relations above/below and between/outside. In each,
half the trials keep the anchor's target object and half change it. The
monitoring subset is a fixed, balanced part of the full set, small enough for
frequent checks during training.

## Downloading the data

The packaged config `spatial_relations_eval` reads the monitoring subsets
directly from the Hugging Face dataset
`agentic-learning-ai-lab/egolearn-spatial-relation-categorization` (CC BY 4.0,
cite the paper). The dataset needs repository access and `HF_TOKEN`, as
described in [Models and Hub data](hub.md). To keep a verified local copy, run:

```bash
uv run egl data fetch spatial-relation-categorization --revision v1
```

Only the monitoring subsets are hosted, so for the full set you prepare the
data yourself as described in the next section.

## Preparing and checking data

You need the authors' 2D scene generator and their Experiment 3 renders. Clone
the generator at the tested commit:

```bash
git clone https://github.com/guydav/simple-relational-reasoning.git
git -C simple-relational-reasoning checkout 08385fe7e15c664a18cb4bf174fe9dda28e42d64
export SPATIAL_GENERATOR_ROOT="$PWD/simple-relational-reasoning"
```

Request the 3D renders from the authors through the contact in the
[published paper](https://www.cs.princeton.edu/~bl8144/papers/DavidsonEtAlCognition2024.pdf).
Set `SPATIAL_RENDER_ROOT` to the directory that holds `quinn_above_below/` and
`quinn_between/`. Then render, export and validate both components:

```bash
uv run --no-sync --with matplotlib --with scipy --with colorcet \
  --with opencv-python-headless python -m egolearn.cli data prepare spatial-relations \
  --generator-root "$SPATIAL_GENERATOR_ROOT" \
  --exp3-image-root "$SPATIAL_RENDER_ROOT" \
  --output-dir data/spatial_relations/v1
```

The extra packages are needed only for rendering. The output holds one
directory per component, `exp1/` for 2D and `exp3/` for 3D, each with a
`manifest.json`.

## Running the evaluation

Evaluate any model with a pooled image representation on the hosted
monitoring subsets:

```bash
uv run egl eval spatial_relations_eval checkpoint=run/final.pth
```

For the full set, point both manifests at your prepared copy and give the
benchmark its own name, so its scores stay apart from monitoring scores. Both
manifests must come from the same profile, full or monitoring.

```bash
uv run egl eval spatial_relations_eval checkpoint=run/final.pth \
  benchmarks.0.name=spatial_relations_full \
  benchmarks.0.abstract_2d_manifest=data/spatial_relations/v1/exp1/manifest.json \
  benchmarks.0.rendered_3d_manifest=data/spatial_relations/v1/exp3/manifest.json \
  output_json=spatial_relations_full.json
```

To track the score during training, add the probe to the `probes` list of
your training config:

```yaml
_imports_:
  - egolearn.evaluation.benchmarks.spatial_relations
probes:
  - _target_: "probe:spatial_relations"
    name: spatial_relations_monitoring
    abstract_2d_manifest: "hf://datasets/agentic-learning-ai-lab/egolearn-spatial-relation-categorization@v1/monitoring/2d-stimuli/manifest.json"
    rendered_3d_manifest: "hf://datasets/agentic-learning-ai-lab/egolearn-spatial-relation-categorization@v1/monitoring/3d-stimuli/manifest.json"
```

## Metrics

Metric names start with the benchmark's `name`, and all values are fractions.

| Metric | Meaning |
| --- | --- |
| `<name>/accuracy` | Headline: the mean of the 2D and 3D component accuracies |
| `<name>/abstract_2d/accuracy`, `<name>/rendered_3d/accuracy` | Component accuracy, an equal-weight mean over its condition cells |
| `<name>/above_below/accuracy`, `<name>/between_outside/accuracy` | Accuracy for each relation, averaged over the components |
| `<name>/same_object/accuracy`, `<name>/changed_object/accuracy` | Accuracy when the candidates keep or change the anchor's target object |
| `<name>/<component>/object_change_gap`, `mean_cosine_margin`, `tie_rate` | Diagnostics: same-object minus changed-object accuracy, the cosine margin of the correct candidate over the foil, and the share of ties |

A condition cell is one combination of renderer, relation and target
condition, which gives 12 cells in the 2D component and 4 in the 3D component.
Weighting cells and components equally keeps the larger 2D trial count from
dominating the headline. A tie within `tie_tolerance` (default `1e-6`) counts
as half correct.

## Differences from the reference protocol

- Only paper Experiments 1 and 3 are included. Rotations (Experiment 2) and
  containment (Experiment 4) are not.
- The 2D stimuli are regenerated with the authors' generator, and the 3D
  trials are a balanced selection of the authors' renders. Neither matches the
  paper's trial lists, and the equal-weight headline is not a score from the
  paper, so scores are not directly comparable to the paper.
- Regenerated 2D pixels depend on the rendering environment. Compare scores
  only on the same prepared copy, which each result identifies by its manifest
  hashes.
- The monitoring subset overlaps the full set. Report full-set scores for final
  results and condition comparisons.
- In the 3D scenes, above/below uses a table as the reference object and
  between/outside uses a storage rack. A difference between the two relations
  may reflect the reference object rather than the relation.
