# EgoSchema

EgoSchema asks five-way multiple-choice questions about 3-minute egocentric
videos, and most questions need the whole clip, such as "What is the overall
goal of C's actions?". In egolearn, a generative vision-language model
watches the video and names the answer's letter, without any training. The
headline is accuracy on the 500 questions with public answers, reported with
a 95% bootstrap interval, and chance is 0.2.

The benchmark is Mangalam, Akshulakov and Malik,
[*EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding*](https://arxiv.org/abs/2308.09126)
(NeurIPS 2023 Datasets and Benchmarks). It has 5,031 questions about videos cut
from Ego4D, and the authors' servers score the full set.

## Downloading the data

Download the questions and public answers from the
[EgoSchema repository](https://github.com/egoschema/EgoSchema) at the pinned
commit. The benchmark checks both files against their SHA-256.

```bash
export EGOLEARN_DATA=/path/to/data
mkdir -p "$EGOLEARN_DATA/egoschema"
for f in questions.json subset_answers.json; do
  curl -L -o "$EGOLEARN_DATA/egoschema/$f" \
    "https://raw.githubusercontent.com/egoschema/EgoSchema/505c787376b5e066d0ae406d0e0d41245cebba15/$f"
done
```

The authors distribute the videos, about 100 GB, through Kaggle. Accept the
competition rules, set up the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)
and run `kaggle competitions download -c egoschema-public`, then place the
videos as `$EGOLEARN_DATA/egoschema/videos/<q_uid>.mp4`.

## Serving the model

The packaged config sends frames to an OpenAI-compatible server. Its default
model is Qwen2.5-VL-7B-Instruct served by vLLM. The egolearn `qa` extra
installs only the client, so install vLLM in its own environment by following
its [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html),
then start the server:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 0, "video": 1}'
```

The Ego4D license does not allow sending its videos to third-party services,
so use only a server that you operate under the dataset's terms. The client
reads the server's API key from `EGOLEARN_LLM_API_KEY`. If you started the
server without `--api-key`, add `model_config.model.generative.vlm.api_key_env=null`
to the evaluation command.

## Running the evaluation

Install the `train` and `qa` extras, then run the packaged config against your
server:

```bash
uv sync --extra train --extra qa
uv run egl eval egoschema_eval vlm_url=http://<host>:8000/v1 output_json=egoschema.json
```

No checkpoint is needed, since the model is built from `model_config`. Set
`vlm_model` if the server uses another model name, and add these overrides for
other kinds of run:

| Override | Effect |
| --- | --- |
| `benchmarks.0.subset_only=false benchmarks.0.submission_dir=<dir>` | Answers all 5,031 questions and writes `submission.csv` for Kaggle, `submission.json` for the validation server and `predictions.jsonl` with every reply |
| `benchmarks.0.max_questions=<n>` | Scores only the first n questions, for smoke tests |
| `model_config.model.generative.vlm.video_input=frames` | Sends frames as separate images, for servers without vLLM's video input. Raise the server's image limit to at least `max_frames` |

Replies are cached under `reply_cache`, keyed by the full request, so a rerun
reads the cache and a change to the frames, prompt or model name misses it.
The key holds the model name but not its weights, so set a new `reply_cache`
when the server serves other weights under the same name.

## Metrics

| Metric | Meaning |
| --- | --- |
| `acc` | Accuracy on the questions with a public answer |
| `acc_ci_low`, `acc_ci_high` | 95% percentile bootstrap interval of `acc` |
| `unparsed` | Share of replies that name no option. They count as wrong |
| `pred_frac_A` … `pred_frac_E` | Share of predictions for each letter, which exposes a letter bias |
| `n_scored`, `n_answered` | Questions scored, and those with a public answer |

## Differences from the reference protocol

- The headline uses the 500-question public subset. Full-set accuracy comes
  from the authors' servers, through the submission files.
- Each video is sampled at 1 frame per second, up to 64 frames, with the
  longer side at most 448 pixels (`fps`, `max_frames` and `max_side`).
  Published results for the same model use their own sampling and prompt, so
  scores can differ from them.
- A reply that names no option letter or option text is wrong, and the
  submission files record it as option A.
