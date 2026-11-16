# Hub artifacts

Fetch prepared benchmark collections from Hugging Face, or reference them
directly in a configuration. Check the [access tiers](#access-tiers) before you
publish data or model weights.

For details on obtaining released model weights and requesting access to gated
checkpoints, see the [checkpoint access instructions](../../README.md#model-checkpoints).

## Verified downloads

The catalog's hosted collections require repository access and `HF_TOKEN`.
The checkpoint access form grants model access only; it does not grant access to
these datasets. If you do not already have dataset access, use the
[local shape-bias preparation](../evaluation.md#prepare-local-stimuli) or
[spatial-relations preparation](spatial_relations.md#preparing-and-checking-data) instructions.

With repository access, use `egl data fetch` to check a local collection against
its published checksums:

```bash
uv run egl data fetch shape-bias --revision v1
```

This command fetches the `agentic-learning-ai-lab/egolearn-shape-bias` collection
into `$EGOLEARN_DATA/shape-bias`, or `data/shape-bias` when the variable is unset.
If a collection already exists at the destination, add `--replace` to replace
its contents. To choose another storage root,
place `--root` before `fetch`:

```bash
uv run egl data --root /path/to/data fetch shape-bias --revision v1
```

The download is staged and verified before installation, with its requested
revision, resolved commit, and verification status recorded in
`.egolearn-fetch.json`. Missing `checksums.sha256` requires `--allow-unverified`;
incorrect checksums always fail. Use `--repo org/name` to override the dataset
repository, and use a commit ID when the revision must be immutable.
The catalog keeps explicit repository IDs for existing lab-hosted collections,
so these commands retain their configured repository locations.

## Packaged collections

The packaged evaluation configs reference the repositories below at their
`v1` revisions. These lab-hosted collections require repository access and
authentication through `HF_TOKEN`; they are not anonymous public downloads. Keep
authentication tokens in the environment rather than configurations or command arguments.

| Repository | Contents | Configuration |
| --- | --- | --- |
| [egolearn-shape-bias](https://huggingface.co/datasets/agentic-learning-ai-lab/egolearn-shape-bias/tree/v1) | Prepared shape-bias stimuli from Tartaglini, Vong, and Lake (2022) | `shape_bias_eval` |
| [egolearn-spatial-relation-categorization](https://huggingface.co/datasets/agentic-learning-ai-lab/egolearn-spatial-relation-categorization/tree/v1) | Spatial monitoring subsets from Davidson, Orhan, and Lake (2024) | `spatial_relations_eval` |

Use `egl data fetch spatial-relation-categorization --revision v1` for the
second collection. Consult the dataset cards and upstream asset terms before
redistributing prepared stimuli, and cite the respective papers. A local rendering
step does not grant permission to redistribute its source assets.
The dataset catalog also lists possible collections, but a catalog entry alone
does not establish that its repository is available.

After a [verified download](#verified-downloads), override the packaged
shape-bias manifest to use your local copy:

```bash
uv run egl eval shape_bias_eval checkpoint=run/final.pth \
  "benchmarks.0.manifest=${EGOLEARN_DATA:-data}/shape-bias/stimulus_manifest.json"
```

Use your chosen storage root in the override if you fetched with `--root`.

## Released models

The following model repositories contain `model.safetensors` and `model.json`.
SAYCam and BabyView weights require the [access-request steps](../../README.md#model-checkpoints)
using evidence for the corresponding training dataset.

| Model | Training data | Access |
| --- | --- | --- |
| [BabyCL full-stream, seed 1337](https://huggingface.co/egolearn/babycl-resnext50-saycam-fullstream-seed1337), [1338](https://huggingface.co/egolearn/babycl-resnext50-saycam-fullstream-seed1338), [1339](https://huggingface.co/egolearn/babycl-resnext50-saycam-fullstream-seed1339) | SAYCam | Manual approval |
| [Visual BabyCL, seed 1337](https://huggingface.co/egolearn/babycl-resnext50-babyview-visual-seed1337) | BabyView | Manual approval |
| [Visual BabyCL, seed 0](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed0), [1](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed1), [2](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed2) | KrishnaCam | Public |
| [Visual BabyCL, seed 1337](https://huggingface.co/egolearn/babycl-resnext50-walkingtours-seed1337) | WalkingTours | Public |
| [V-JEPA 2 IID](https://huggingface.co/egolearn/vjepa2-vits16-saycam-iid-seed0), [streaming](https://huggingface.co/egolearn/vjepa2-vits16-saycam-streaming-seed0) | SAYCam | Manual approval |
| [V-JEPA 2 IID](https://huggingface.co/egolearn/vjepa2-vits16-babyview-iid-seed0), [streaming](https://huggingface.co/egolearn/vjepa2-vits16-babyview-streaming-seed0) | BabyView | Manual approval |

All V-JEPA 2 entries use seed 0. Read the model card for the training protocol
before choosing weights. Match the training budget and input protocol before
comparing a release with the two runs in the
[BabyCL results table](../../projects/babycl/README.md#saycam-results).

### Authentication and evaluation

For gated weights, set `HF_TOKEN` in your environment to a read token belonging
to the approved Hugging Face account. Create tokens in your
[Hugging Face settings](https://huggingface.co/docs/hub/security-tokens). Keep the
token out of configs, command arguments, and Git. If access is denied, check that
the approval and token belong to the same account and that the token permits the
model repository.

Benchmark data needs its own access. For example, prepare shape-bias stimuli using
[these instructions](../evaluation.md#prepare-local-stimuli) or obtain the hosted collection
above, then set `SHAPE_BIAS_MANIFEST` to your local `stimulus_manifest.json`.
This example evaluates an ungated KrishnaCam export at an immutable revision:

```bash
uv run egl eval shape_bias_eval \
  checkpoint=hf://models/egolearn/babycl-resnext50-krishnacam-seed0@311ec535c5a6bcefded0e01f449ad56eaa1b9dc1/model.safetensors \
  "benchmarks.0.manifest=$SHAPE_BIAS_MANIFEST" output_json=shape_bias_results.json
```

The result file contains the shape-bias score and model/input provenance. Replace
the checkpoint reference to use another compatible model; gated models require
approval before this command can download their weights. Model metadata is fetched
from the same revision as the weights.

## Configuration paths

A component data path can use
`hf://<org>/<repo>@<revision>[/path/in/repo]`, where the revision is a tag or
commit ID without slashes. For example, the shape-bias manifest is:

```text
hf://datasets/agentic-learning-ai-lab/egolearn-shape-bias@v1/stimulus_manifest.json
```

Use `hf://datasets/` to select a dataset repository and `hf://models/` to select
a model repository. Without a prefix, component inputs default to datasets;
checkpoint, initialization, and model-config inputs default to models.
An explicit `hf://models/org/repo@revision/vocab.json` lets a tokenizer read its
vocabulary from a model repository.

Model references must include a filename, such as
`hf://models/org/repo@revision/model.safetensors`, which the resolver downloads
without fetching other model files. Loading an export also retrieves its sibling
`model.json` at the same immutable commit, which evaluation records alongside
the original reference and content digest.

Direct Python callers can use `egolearn.hub.resolve(path)` before constructing a
dataset, while leaving local paths unchanged. Config-driven resolution uses the
standard Hugging Face cache without verifying collection checksums, so use
[verified downloads](#verified-downloads) when that check is required.

Artifact resolution can download data even with training `dry_run=true`,
because that option skips stream iteration but still constructs the pipeline
and resolves its inputs.

## Cache storage

Dataset references download the full repository snapshot so manifests can read
sibling files. When the snapshot uses symlinks, EgoLearn creates a copy with real
files to preserve manifest containment checks, which requires extra disk space.

`HF_HOME` controls the Hugging Face cache, with materialized copies stored under
`<Hub cache>/datasets--<org>--<repo>/egolearn-materialized/<generation>/<commit>`.
These copies are not tracked by `hf cache` cleanup tools, so remove the
`egolearn-materialized` directory manually when no job uses it.

An existing complete copy can be reused from a read-only shared cache, while
first-time materialization needs a writable cache. Alternatively, use
`egl data --root DIR fetch` to install real files on writable storage.
If an offline snapshot was partial, newly cached files create a new
materialization instead of reusing that subset.

## Access tiers

EgoLearn distinguishes public metadata from images that require separate
access, as described in the table below. These rules cover permitted artifact
contents and do not establish which Hub repositories are available.

| Dataset or artifact | Access tier | Permitted Hub contents |
| --- | --- | --- |
| Prepared shape-bias stimuli | Upstream terms | Source-derived stimuli only with redistribution permission; otherwise prepare locally |
| Spatial-relations trials and monitoring subsets | Public | Licensed stimuli, fixed trials, and manifests |
| BabyView 4AFC and linear evaluation manifests | Public | Split and trial metadata without pixels or restricted captions |
| Labeled-S split metadata | Public | Split identifiers without source frames |
| ImageNet | Upstream-licensed | Manifests without images |
| IntPhys 2 Main split | Upstream-licensed | No hosted copy; obtain videos from the upstream release |
| GRASP Level 2 | Upstream-licensed | Pair manifest with hashes and frame counts, without pixels; obtain videos from the upstream release |
| SAYCam, Labeled-S frames, restricted BabyView and BabyVLM images | IRB | No uploads, including private repositories and manifests with embedded pixels |

Keep full training snapshots in local run directories. Resume accepts legacy
Hub snapshot references and uses unrestricted pickle, so only resume snapshots
whose producer and contents you trust. Checksums establish file integrity
without establishing trust in Python pickle contents. See
[checkpoint trust](../../SECURITY.md#model-export-and-checkpoint-trust).

## Publication

Publish a prepared collection or exported model only when you hold the required
distribution rights. Dataset uploads include non-hidden payload files, so
inspect the directory first and exclude credentials, restricted data, embedded
frames, and personal run metadata.

The lower-level `egolearn.hub` fetch and push commands default to
`egolearn/<name>`, without adding a repository-name prefix. Use `--repo org/name`
to select another repository. Unlike `egl data fetch`, the lower-level fetch
command does not consult the dataset catalog.

To upload a prepared dataset directory:

```bash
uv run python -m egolearn.hub push prepared/ \
  --name shape-bias --repo YOUR_NAMESPACE/shape-bias --license cc-by-4.0
```

To export and upload model weights to a new repository:

```bash
uv run egl export runs/example/final.pth exports/babycl-saycam-seed0
uv run python -m egolearn.hub push exports/babycl-saycam-seed0 \
  --name babycl-saycam-seed0 --repo YOUR_NAMESPACE/babycl-saycam-seed0 \
  --repo-type model --license mit
```

Replace `YOUR_NAMESPACE` with your account or organization, where you have write
permission. Choose the paths, names, and license for your artifact.
Model uploads require a validated `model.safetensors` and `model.json` pair,
and reject other payloads, including `.pth` files. Model repository
names can include the method, training data, and seed.

Omitting `--repo` targets the project namespace `egolearn`, which requires permission
from that organization. Revisioned `hf://` references always use their explicit
organization and repository, including existing lab-hosted artifacts.

`push` creates a staging copy, adds checksums and a minimal card, and leaves
the local source unchanged. It ignores the source `README.md` and overwrites
the remote card, including custom access fields and instructions. Do not use
`push` on a repository whose model card or access form you need to preserve.
Update individual files with the Hugging Face client or web interface instead.
New repositories are private by default.
`--public` is required to create or upload to a public repository; it does not
change an existing repository's visibility.

An upload replaces the remote payload and removes stale files, preserving
`.gitattributes`. Unsupported license identifiers fail before repository
creation and report the accepted list.

## Local verification

Generate a checksum inventory after preparing a collection, then verify it
without changing the payload:

```bash
uv run python -m egolearn.hub checksum prepared/
uv run egl data verify prepared/
```

`checksum` writes the `checksums.sha256` inventory, and `verify` checks file
contents and collection completeness against it. A mismatch returns an
unsuccessful exit status without modifying the collection.
