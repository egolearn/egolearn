# First checkpoint evaluation

In this tutorial you evaluate a public model, trained on KrishnaCam video, on
shape-bias stimuli and read its score. Everything runs on a CPU and needs
internet access for the first downloads, but no Hugging Face account,
restricted data or training.

## Stimulus preparation

Run these commands from your egolearn checkout, after the
[quickstart](../../README.md#quickstart):

```bash
uv sync --extra train
uv run egl data prepare shape-bias
```

Preparation downloads the source masks and textures, checks them against
pinned checksums and renders 256 stimulus images, then ends with a line like
this:

```text
Prepared 256 stimuli and 57,600 triplets in data/shape-bias (... assets downloaded).
```

If you set `EGOLEARN_DATA`, the stimuli go under that directory instead of
`data`. Preparation refuses to overwrite stimuli you prepared before, so in
that case go straight to the next step. The
[shape-bias data section](data.md#shape-bias) describes the source stimuli and
their license.

## Evaluation

Evaluate a public ResNeXt-50 model on the stimuli you prepared:

```bash
uv run egl eval shape_bias_eval \
  checkpoint=hf://models/egolearn/babycl-resnext50-krishnacam-seed0@311ec535c5a6bcefded0e01f449ad56eaa1b9dc1/model.safetensors \
  "benchmarks.0.manifest=${EGOLEARN_DATA:-data}/shape-bias/stimulus_manifest.json" \
  device=cpu output_json=shape_bias_results.json
```

egolearn downloads the weights and their metadata from that fixed revision.
The `manifest` override points the benchmark at your local stimuli instead of
the hosted copy, which needs access. The model's training data and limitations
are on its [model card](https://huggingface.co/egolearn/babycl-resnext50-krishnacam-seed0).

The command writes `shape_bias_results.json` in the current directory and a
run record under `runs/eval/records/`. Wait until it exits successfully before
you read the results, since a failed run can leave an older results file in
place.

## Result

Print the run's status and its shape bias:

```bash
uv run python - <<'PYTHON'
import json

with open("shape_bias_results.json") as handle:
    result = json.load(handle)
print(result["status"])
print(result["results"]["shape_bias"]["shape_bias"])
PYTHON
```

The first line is `complete`, and the second is a fraction between 0 and 1:
the share of the 57,600 comparisons in which the model judged the image with
the same shape more similar than the image with the same texture. A value
above 0.5 means the model prefers shape, and the score is not a classification
accuracy. The results file also holds the other shape-bias metrics and the
provenance of the checkpoint and inputs.

To evaluate another model, pick one from the [model catalog](hub.md#released-models),
which lists the evaluations each model supports.
