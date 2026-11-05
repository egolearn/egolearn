# Publishing benchmark results

Use `egl publish` to publish evaluation results and update the website's benchmark
table, with optional checkpoint uploads. The publisher stores results in a dataset
repository and uploaded weights in private model repositories. Your original files remain local.

Install the `train` extra before preparing exports. On the cluster, run these commands
inside a compute allocation, following the [cluster guide](cluster.md).

## Publication manifest

Create a local JSON manifest with the training condition, expected seeds, and source
files. Paths in this manifest are local inputs and are not uploaded. Keep it under
the ignored `exports/` directory.

```json
{
  "schema_version": 1,
  "condition": {
    "id": "example-condition",
    "label": "Example model",
    "dataset": "Example dataset",
    "regime": "continual",
    "architecture": "ResNeXt-50",
    "description": "Final checkpoint from the declared training configuration."
  },
  "expected_seeds": [1337],
  "runs": [{
    "training_seed": 1337,
    "checkpoint": "runs/example/final.pth",
    "training_config": "runs/example/.hydra/config.yaml",
    "repo_id": "your-org/egolearn-example-seed1337",
    "input_hashes": "runs/example/input_hashes.json",
    "evaluations": ["runs/example/evaluation.json"]
  }]
}
```

Use the same condition ID and expected seed set when publishing additional seeds,
and give different configurations distinct condition IDs. The publisher checks the
saved training seed, but you must identify which experiments share a training condition.

Evaluation files use the native `EvaluationResult` format. Each file must identify
the checkpoint digest and step, benchmark version, evaluation seed, configuration,
and completed metrics. Failed evaluations cannot enter a publication bundle.

The input-hash file records the identities consumed by the evaluation:

```json
[
  {
    "benchmark": "labeled_s_4afc",
    "kind": "manifest",
    "sha256": "<SHA-256 recorded when the evaluation ran>"
  },
  {
    "benchmark": "labeled_s_4afc",
    "kind": "vocab_json",
    "sha256": "<SHA-256 recorded when the evaluation ran>"
  }
]
```

Provide recorded hashes for file inputs such as manifests, split files, CSVs and
vocabularies. Do not substitute a hash of today's data for missing historical
evidence. Copy hashes from the evaluation's run record when available. Preparation
checks their format and required keys, but does not independently verify the supplied
hashes against a run record or read the original inputs.
Directory-based evaluations can supply recorded class ordering and counts
through their native metadata. Neither source establishes a digest of the underlying
image collection; the published protocol states that limit.

The default columns cover Labeled-S and BabyView 4AFC, VTWT, Baby Winoground group
accuracy, shape preference, and Labeled-S, BabyView and ImageNet linear decoding.
To publish another metric, provide a `columns` array in the manifest. Each column
needs `id`, `benchmark`, `metric`, `label`, `description`, `unit` and `direction`.
Supported units are `fraction` and `number`, with directions of `min` and `max`.
Column IDs must retain the same definition, including display text, across publications.

Condition metadata is also immutable. Keep the original manifest when adding seeds;
pushes that change a condition's label or description are refused.

### Imported reference checkpoints

For a reference whose training seed and configuration are unavailable, set
`reference: true` and `expected_seeds: [null]` in the manifest. Include one run with
`training_seed: null` and `training_config: null`. Keep the checkpoint, evaluation,
input-hash, and destination fields from the example above.

Use a separate condition ID for each reference checkpoint. The table reports one
checkpoint with an unknown training seed and no standard deviation. Preparation
rejects a null seed when the checkpoint records a known seed. If you have the
original training configuration and seed, use the ordinary manifest instead.

Describe the reference's known training data, architecture, and checkpoint selection
in the condition fields. A filename alone does not establish initialization history.
Model construction metadata remains part of the verified export even when the
training configuration is unavailable.

### Inapplicable benchmarks

For a model without a trained image-text head, add this top-level manifest field:

```json
{
  "not_applicable": {
    "labeled_s_4afc": "No trained image-text head.",
    "babyview_4afc": "No trained image-text head.",
    "vtwt": "No trained image-text head.",
    "baby_winoground": "No trained image-text head."
  }
}
```

The keys are column IDs, and each reason must be a nonempty string. These cells
display N/A with the supplied explanation. Leave unfinished evaluations undeclared,
which displays a dash in existing table columns. The publisher rejects scores for
a metric declared N/A and requires the same declarations across seeds of a condition.

Keep different evaluation resolutions and protocols explicit in each condition's
description. The table also includes recorded resolutions in metric tooltips.
Comparisons across architectures or training budgets do not isolate a method's effect.

## Preparation

Export and validate the batch into a new directory:

```bash
egl publish prepare exports/manifest.json --output exports/publication
```

Preparation exports `model.safetensors` and `model.json`, checks the evaluation's
checkpoint identity, and reconstructs each exported model with strict weight loading.
It writes a sanitized `bundle.json` beside the model directories. Existing destinations
are refused, and preparation does not contact the Hub.

Historical schema-1 evaluations have no completion field. After checking their exit
records, pass `--allow-legacy-evaluations` to import them. Their records retain the
legacy designation. Missing training and evaluation code commits remain null rather
than inheriting the exporter's current revision.
Imported references and N/A declarations retain the same input-hash requirements.
Recover missing historical records or rerun the affected evaluations before publishing.

Inspect the bundle and exported metadata before uploading. The publisher omits local
paths and rejects unexpected publication fields, nonfinite metrics, and incomplete
model exports. It does not upload optimizer state, replay buffers, frames or transcripts.

## Upload

Set `HF_TOKEN` through your environment, or use `--prompt-token` for a hidden prompt.
The token needs write access to the destination organization.

```bash
egl publish push exports/publication \
  --results-repo your-org/egolearn-results \
  --collection your-org/egolearn-COLLECTION_ID --results-only
```

Copy the collection slug from its Hugging Face URL, or omit the optional collection
argument. With `--results-only`, the command uploads evaluation records and their generated
table while keeping weights local. Records retain the verified checkpoint and export hashes,
with null repository, revision and visibility fields. The website labels these checkpoints
as not uploaded and links to their evaluation evidence.

Omit `--results-only` to upload private model repositories before publishing the results.
Both modes commit per-run records, the index, and `benchmarks.json` together. The returned
receipt names the results revision and any uploaded models, and is saved as `publication.json`.

Both models and results default to private. `--public-results` selects a public results
destination without making weights public. Existing visibility is preserved, and a mismatch
fails. Follow the [checkpoint publication rules](evaluation.md#hub-artifacts-and-access-tiers) before
redistributing SAYCam-trained weights. Private model cards grant no redistribution license.

Keep the prepared bundle, including `model.json` and `model.safetensors`, after a
results-only upload. Rerun `push` on that same bundle without `--results-only` to attach
weights later. Preparing again can change the export metadata hash because it records
the exporter's code revision. Such a bundle cannot replace the published evidence.

Retries reuse identical models and records. The publisher checks existing results for
conflicts before uploading weights, then checks again when committing the results.
A concurrent conflicting publication can still leave uploaded weights without a results
link. Concurrent updates to distinct runs are merged. Adding a seed extends the existing
condition; repeated evaluations of one checkpoint do not count as independent seeds.

## Website deployment

The Cloudflare Pages function at `/api/benchmarks` reads the results feed. Configure these
settings for the Pages environment that serves the website:

| Setting | Value |
| --- | --- |
| `EGOLEARN_RESULTS_REPO` | The results dataset's `org/repository` identifier |
| `EGOLEARN_RESULTS_TOKEN` | A secret with read access to that results repository |

Use a token scoped to the results repository. Add the settings to both Preview and
Production when both environments need results, then redeploy. Keep the existing
Cloudflare Access policy on the staging site, including its `/api/benchmarks` route.
The function serves sanitized metrics to anyone allowed to visit that site; it does
not add a second login or enforce Hugging Face membership for each visitor.

The build remains `bash scripts/docs/build.sh`, with `dist` as the output directory.
Pages discovers `functions/` at the repository root. Only `/api/benchmarks` invokes a
function, while the documentation and other assets remain static. Tokens are read at runtime
and never included in JavaScript, Git, build artifacts or browser responses.

An open, visible page checks for new results every minute. Each successful publication
updates the feed without rebuilding the website. Failed refreshes retain the last
successful table and show its timestamp. A first visit without a working source shows
an unavailable message instead of placeholder scores. Responses disable shared caching.

Aggregate values require all declared seeds and identical protocols. Missing results
and protocol mismatches display a dash. The table uses sample standard deviation,
and a single-checkpoint result has no standard deviation. Method details link to the model
revision and per-run evidence. Those Hub links still require the visitor's Hub access.

Reference checkpoints and N/A declarations produce schema-2 run records and feeds.
The publisher and website also read schema-1 records without rewriting their evidence.
Deploy the schema-2 website reader before publishing these new records.

For this publication path, upload through `egl publish push`. Uploading checkpoint
weights alone does not run evaluations or update the results index.
