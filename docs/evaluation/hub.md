# Models and Hub data

This page lists the released EgoLearn models and how to get access to the gated
ones. It also covers the hosted benchmark data, `hf://` paths in configs, and
the rules for publishing your own models and data.

## Released models

Each repository holds an exported `model.safetensors` and `model.json`, and its
model card describes the training data and limitations.

| Model and training data | Weights | Training guide |
| --- | --- | --- |
| BabyCL full-stream on SAYCam, seeds [1337](https://huggingface.co/egolearn/babycl-resnext50-saycam-fullstream-seed1337), [1338](https://huggingface.co/egolearn/babycl-resnext50-saycam-fullstream-seed1338), [1339](https://huggingface.co/egolearn/babycl-resnext50-saycam-fullstream-seed1339) | Approval | [BabyCL](../../projects/babycl/README.md) |
| Visual BabyCL on BabyView, seed [1337](https://huggingface.co/egolearn/babycl-resnext50-babyview-visual-seed1337) | Approval | None packaged |
| Visual BabyCL on KrishnaCam, seeds [0](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed0), [1](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed1), [2](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed2) | Public | [BabyCL on KrishnaCam](../../projects/babycl/docs/krishnacam.md) |
| Visual BabyCL on WalkingTours, seed [1337](https://huggingface.co/egolearn/babycl-resnext50-walkingtours-seed1337) | Public | [BabyCL on WalkingTours](../../projects/babycl/docs/walkingtours.md) |
| V-JEPA 2 on SAYCam, seed 0, [IID](https://huggingface.co/egolearn/vjepa2-vits16-saycam-iid-seed0) and [streaming](https://huggingface.co/egolearn/vjepa2-vits16-saycam-streaming-seed0) | Approval | [V-JEPA 2](../../projects/vjepa2/README.md) |
| V-JEPA 2 on BabyView, seed 0, [IID](https://huggingface.co/egolearn/vjepa2-vits16-babyview-iid-seed0) and [streaming](https://huggingface.co/egolearn/vjepa2-vits16-babyview-streaming-seed0) | Approval | [V-JEPA 2](../../projects/vjepa2/README.md) |

Pass a model to `egl eval` as
`checkpoint=hf://models/egolearn/<repo>@<commit>/model.safetensors`, with a
commit from the model's history, as the [evaluation example](../evaluation.md#example)
does. The families support different [benchmarks](../evaluation.md#benchmark-selection):

- **BabyCL on SAYCam** runs image and image–text benchmarks. These full-stream
  runs are longer than the `saycam_default` runs in the BabyCL results table.
- **Visual BabyCL** has no text encoder, so it cannot run 4AFC, VTWT or Baby
  Winoground.
- **V-JEPA 2** needs the static-image adapter config from the
  [V-JEPA 2 guide](../../projects/vjepa2/README.md) for image benchmarks.

## Checkpoint access

KrishnaCam and WalkingTours weights are public. SAYCam and BabyView weights need
approval, which requires your Databrary access to the training data. Get
[SAYCam](data.md#saycam-access) or [BabyView](data.md#babyview-training-data)
access first, then request the weights:

1. Open the access form on the model's Hugging Face page.
2. Describe your intended use. Under **Link to Databrary access proof**, link to
   a screenshot of your Databrary permissions that shows your name and the
   dataset. Make the link readable by the EgoLearn team, and leave out unrelated
   personal details and any dataset images or video.
3. Submit the form. The EgoLearn team reviews it, and no email is needed.

After approval, set `HF_TOKEN` in your environment to a read
[access token](https://huggingface.co/docs/hub/security-tokens) from the same
Hugging Face account, and keep it out of configs, command lines and Git. Model
approval does not cover the hosted benchmark data.

## Packaged collections

`shape_bias_eval` and `spatial_relations_eval` read their inputs from two
lab-hosted dataset repositories at revision `v1`, which need access approval
and `HF_TOKEN`.

| Repository | Contents | Config |
| --- | --- | --- |
| [egolearn-shape-bias](https://huggingface.co/datasets/agentic-learning-ai-lab/egolearn-shape-bias/tree/v1) | Shape-bias stimuli from Tartaglini, Vong and Lake (2022) | `shape_bias_eval` |
| [egolearn-spatial-relation-categorization](https://huggingface.co/datasets/agentic-learning-ai-lab/egolearn-spatial-relation-categorization/tree/v1) | Spatial monitoring subsets from Davidson, Orhan and Lake (2024) | `spatial_relations_eval` |

Without access, prepare the [shape-bias](data.md#shape-bias) or
[spatial-relations](spatial_relations.md) inputs yourself. With access,
`egl data fetch` downloads a collection and checks its published checksums:

```bash
uv run egl data fetch shape-bias --revision v1
uv run egl data fetch spatial-relation-categorization --revision v1
```

Each collection lands in `data/<name>`, or under `$EGOLEARN_DATA` when it is
set. Put `--root DIR` before `fetch` to choose another location, and add
`--replace` to overwrite an existing copy. `egl data list` shows the whole
catalog, and its `FETCH` column means only that a collection can be fetched, not
that your account has access.

## Hub paths in configs

Any input path in a config can name a file in a Hub repository at a tag or
commit:

```text
hf://datasets/agentic-learning-ai-lab/egolearn-shape-bias@v1/stimulus_manifest.json
```

Without the `datasets/` or `models/` prefix, data inputs read dataset
repositories, while `checkpoint` and `model_config` read model repositories. A
model reference names one file, and evaluation fetches the matching
`model.json` from the same commit. Dataset references download the whole
repository into the Hugging Face cache under `HF_HOME`, without checking the
collection checksums.

## Access tiers

A dataset's access tier decides what you may upload to the Hub. It does not say
whether a hosted copy exists or who can download it.

| Dataset or artifact | Access tier | Permitted Hub contents |
| --- | --- | --- |
| Prepared shape-bias stimuli | Upstream terms | Source-derived stimuli only with redistribution permission; otherwise prepare locally |
| Spatial-relations trials and monitoring subsets | Public | Licensed stimuli, fixed trials and manifests |
| BabyView 4AFC and linear evaluation manifests | Public | Split and trial metadata without pixels or restricted captions |
| Labeled-S split metadata | Public | Split identifiers without source frames |
| ImageNet | Upstream-licensed | Manifests without images |
| IntPhys 2 Main split | Upstream-licensed | No hosted copy; obtain videos from the upstream release |
| GRASP Level 2 | Upstream-licensed | Pair manifest with hashes and frame counts, without pixels |
| SAYCam, Labeled-S frames, restricted BabyView and BabyVLM images | IRB | No uploads, not even to private repositories, and no manifests with embedded pixels |

Keep full training snapshots in local run directories. They load with
unrestricted pickle, so resume only from snapshots you trust (see
[checkpoint trust](../../SECURITY.md#model-export-and-checkpoint-trust)).

## Publication

Publish a prepared collection or exported model only if you hold the
distribution rights and its [access tier](#access-tiers) allows the contents. A
dataset upload includes every non-hidden file, so first remove credentials,
restricted data, embedded frames and personal run metadata from the directory.

```bash
# A prepared dataset directory
uv run python -m egolearn.hub push prepared/ \
  --name my-stimuli --repo YOUR_NAMESPACE/my-stimuli --license cc-by-4.0

# An exported model
uv run egl export runs/example/final.pth exports/babycl-saycam-seed0
uv run python -m egolearn.hub push exports/babycl-saycam-seed0 \
  --name babycl-saycam-seed0 --repo YOUR_NAMESPACE/babycl-saycam-seed0 \
  --repo-type model --license mit
```

Replace `YOUR_NAMESPACE` with an account or organization where you can write.
Without `--repo`, `push` uploads to the `egolearn` organization. A model upload
must contain only a validated `model.safetensors` and `model.json` pair. On the
remote repository, `push` does the following:

- It creates new repositories as private. `--public` allows a public
  destination, but it does not change an existing repository's visibility.
- It replaces the remote files and removes stale ones.
- It replaces the model card, including any access form, so edit a gated
  repository on the Hugging Face website instead.
- It adds a `checksums.sha256` inventory, which `egl data verify DIR` checks.
