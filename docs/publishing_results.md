# Publishing benchmark results

This page is for maintainers who add evaluated checkpoints to the results
table on the egolearn website. To evaluate or compare your own models, you
need only [checkpoint evaluation](evaluation.md) and
[benchmark suites](benchmark_suites.md).

`egl publish` turns local checkpoints and their evaluation outputs into
per-run records in the Hugging Face dataset `egolearn/egolearn-results`. The
website reads its table from that dataset, so a new publication appears without
a website rebuild. Weights either stay local or go to private model
repositories.

Before you start, install the `train` extra and collect, for each checkpoint:

- the checkpoint file and the saved Hydra config of its training run
- the JSON files that `egl eval` wrote for it (`output_json`)
- the run records of those evaluations, which hold their input hashes

## Publication manifest

Write a JSON manifest that describes one training condition and its seeds.
Keep it under the ignored `exports/` directory, since its paths are local and
are never uploaded:

```json
{
  "schema_version": 1,
  "condition": {
    "id": "babycl-saycam",
    "label": "BabyCL",
    "dataset": "SAYCam",
    "regime": "continual",
    "architecture": "ResNeXt-50",
    "description": "Final checkpoint of the default BabyCL SAYCam configuration."
  },
  "expected_seeds": [1337, 1338, 1339],
  "runs": [{
    "training_seed": 1337,
    "checkpoint": "runs/babycl-seed1337/final.pth",
    "training_config": "outputs/<date>/<time>/.hydra/config.yaml",
    "repo_id": "your-org/egolearn-babycl-seed1337",
    "input_hashes": "exports/babycl-seed1337-input-hashes.json",
    "evaluations": ["runs/babycl-seed1337/evaluation.json"]
  }]
}
```

The `condition` fields are the text the website shows, and they cannot change
after the first publication. To add seeds later, reuse the same `condition.id`
and `expected_seeds`. The table shows a mean only once every expected seed is
published. Each run's `training_seed` must match the seed in its
`training_config`, the `.hydra/config.yaml` that Hydra saved for the training
run. `repo_id` names the model repository that receives the weights if you
upload them.

A score fills a website column when its benchmark name and metric match the
column, so name the benchmarks in your evaluation configs after the default
columns: `labeled_s_4afc`, `babyview_4afc`, `vtwt`, `baby_winoground`,
`shape_bias`, `labeled_s_linear`, `babyview_linear` and `imagenet_linear`.

### Input hashes

The input-hash file lists the SHA-256 of each file input, such as a manifest or
vocabulary, that an evaluation read:

```json
[
  {"benchmark": "labeled_s_4afc", "kind": "manifest", "sha256": "<recorded sha256>"},
  {"benchmark": "labeled_s_4afc", "kind": "vocab_json", "sha256": "<recorded sha256>"}
]
```

`kind` is the benchmark's config key. Copy each digest from the
`input_provenance` of the evaluation's run record, where it appears under
`benchmarks.<index>.<kind>`. If the record is missing, rerun the evaluation
rather than hashing today's files.

### Optional fields

- `"reference": true` with `"expected_seeds": [null]` publishes an imported
  checkpoint whose training seed and config are unknown. Give its single run
  `"training_seed": null` and `"training_config": null`, and use a separate
  condition ID for each such checkpoint.
- `not_applicable` maps a column ID to a reason, such as
  `{"vtwt": "No trained image-text head."}`. The website shows N/A with that
  reason. Leave a column out to show a dash for an evaluation that has not run.
- `columns` replaces the default columns. Each entry needs `id`, `benchmark`,
  `metric`, `label`, `description`, `unit` (`fraction` or `number`) and
  `direction` (`min` or `max`). A column ID must keep its definition across
  publications.

## Preparation

Export the checkpoints and check the evaluations into a new directory:

```bash
uv run egl publish prepare exports/manifest.json --output exports/publication
```

Preparation writes `model.safetensors` and `model.json` for each run, reloads
each export, checks that every evaluation scored that exact checkpoint, and
writes `bundle.json`. It does not contact the Hub, so you can read
`bundle.json` before anything is uploaded. Evaluation files from before
evaluations recorded a status need `--allow-legacy-evaluations`.

## Upload

Set `HF_TOKEN` in your environment to a token with write access to the
destination organization, or pass `--prompt-token` to type it. Then publish
the results and keep the weights local:

```bash
uv run egl publish push exports/publication \
  --results-repo egolearn/egolearn-results \
  --public-results --results-only
```

The website needs `--public-results`, which makes the results dataset public
while weights stay private. Without `--results-only`, the command also uploads
each model to a private repository named by `repo_id`. To attach weights after
a results-only upload, rerun `push` on the same bundle rather than preparing a
new one, because a new export does not match the published records. Read the
[checkpoint publication rules](evaluation/hub.md) before you upload
SAYCam-trained weights.

`--collection SLUG` also adds the results and any uploaded models to a Hugging
Face collection. The command saves its receipt as `publication.json` in the
bundle, and you can rerun it safely, since identical models and records are
reused.
