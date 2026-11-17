# First checkpoint evaluation

Evaluate a public KrishnaCam model on locally prepared shape-bias stimuli and
inspect its score. You can run this example on CPU without a Hugging Face account, restricted data,
or model training, with internet access for the initial stimulus and weight downloads.

## Environment and data

Run these commands from your EgoLearn checkout after following the
[installation instructions](../../README.md#quickstart):

```bash
uv sync --extra train
uv run egl data prepare shape-bias
```

Preparation downloads checksum-verified source masks and textures, renders 256
images, and writes `data/shape-bias/stimulus_manifest.json` while printing image
and triplet counts. If you set `EGOLEARN_DATA`, the collection
is written under that directory instead of `data`.

If you already prepared these stimuli, reuse the manifest and continue with
evaluation, since preparation refuses an existing collection. The
[preparation guide](../evaluation.md#prepare-local-stimuli) describes the source,
preprocessing, and upstream terms.

## Evaluation

Run the shape-bias benchmark with this public ResNeXt-50 checkpoint:

```bash
uv run egl eval shape_bias_eval \
  checkpoint=hf://models/egolearn/babycl-resnext50-krishnacam-seed0@311ec535c5a6bcefded0e01f449ad56eaa1b9dc1/model.safetensors \
  "benchmarks.0.manifest=${EGOLEARN_DATA:-data}/shape-bias/stimulus_manifest.json" \
  device=cpu output_json=shape_bias_results.json
```

EgoLearn downloads the weights and their construction metadata from the same
immutable revision. The local manifest overrides the config's hosted dataset,
so this command needs no access to the lab's prepared Hub collection.
The [model card](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed0)
describes its training inputs and limitations.

The command writes `shape_bias_results.json` in the current directory and a
separate run record under `runs/eval/records/`. Wait for a successful exit before
reading the result, since setup failures can leave an older output file unchanged.

## Result interpretation

Print the completion status and shape-preference fraction:

```bash
uv run python - <<'PYTHON'
import json

with open("shape_bias_results.json") as handle:
    result = json.load(handle)
print(result["status"])
print(result["results"]["shape_bias"]["shape_bias"])
PYTHON
```

A successful evaluation prints `complete` followed by a fraction between 0 and 1,
measured over 57,600 comparisons. A value above 0.5 means the representation
prefers matching shape over matching texture. It is not classification accuracy.
The JSON includes [additional metrics](../evaluation.md#shape-bias), along with
the checkpoint, benchmark settings, and input provenance.

For another checkpoint, use the [model catalog](hub.md#released-models) to select
a compatible evaluation. Image-only models cannot run image–text benchmarks,
and V-JEPA 2 uses its own [static-image evaluation configuration](../../projects/vjepa2/README.md#released-checkpoints).
