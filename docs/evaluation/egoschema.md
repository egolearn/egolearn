# EgoSchema

EgoSchema asks five-way multiple-choice questions about 3-minute egocentric
videos, and most of them need the whole clip ("What is the overall goal of C's
actions?"). EgoLearn scores a generative model that watches the video and names
the answer's letter, without training anything. The headline is accuracy on
the 500 questions with public answers, reported with a 95% bootstrap interval.
Chance is 0.2.

The benchmark is Mangalam, Akshulakov and Malik,
[*EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language
Understanding*](https://arxiv.org/abs/2308.09126) (NeurIPS 2023 Datasets and
Benchmarks). It has 5,031 questions, each about one video cut from Ego4D.
Answers are public for 500 of them, and the authors' servers score the full set.

## Downloading the data

Questions and public answers come from the
[EgoSchema repository](https://github.com/egoschema/EgoSchema) at commit
`505c787376b5e066d0ae406d0e0d41245cebba15`. The loader checks both files against
their pinned SHA-256:

```bash
mkdir -p "$EGOLEARN_DATA/egoschema"
for f in questions.json subset_answers.json; do
  curl -L -o "$EGOLEARN_DATA/egoschema/$f" \
    "https://raw.githubusercontent.com/egoschema/EgoSchema/505c787376b5e066d0ae406d0e0d41245cebba15/$f"
done
```

The authors distribute the videos (about 100 GB) through Kaggle. Accept the
competition rules, then run `kaggle competitions download -c egoschema-public`.
Place the videos as `$EGOLEARN_DATA/egoschema/videos/<q_uid>.mp4`.

## Serving the model

The packaged config talks to an OpenAI-compatible server through
`generative:openai_chat`. Its default model is Qwen2.5-VL-7B-Instruct served by
vLLM on one GPU:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 0, "video": 1}'
```

EgoSchema videos are cut from Ego4D, whose license does not allow sending them
to third-party services. Point the evaluation only at servers the lab runs
itself. If the server was started with `--api-key`, put the key in an
environment variable and set `model_config.model.generative.vlm.api_key_env` to
its name.

## Running the evaluation

Install the `train` and `qa` extras with `uv sync --extra train --extra qa`, then
give the server's address:

```bash
uv run egl eval egoschema_eval vlm_url=http://<host>:8000/v1 \
  output_json=egoschema.json
```

No checkpoint is needed: the model is built from `model_config`. Set `vlm_model`
when the server serves the model under another name. To score all 5,031
questions and write the files the authors' servers accept, add
`benchmarks.0.subset_only=false benchmarks.0.submission_dir=<dir>`. The directory
receives `submission.csv` for Kaggle, `submission.json` for the validation
server and `predictions.jsonl` with every reply. A reply that names no option is
submitted as option 0.

## Scoring contract

Each video is sampled at `fps` frames per second (default 1), at most
`max_frames` frames (default 64), so a 3-minute video gives 64 frames about
2.8 s apart. Frames start at the first frame of the video, are evenly spaced,
and are resized so their longer side is at most `max_side` pixels (default 448,
aspect ratio kept). These settings are recorded with the result. Each prediction
row also records the frame count and sampling rate the model saw, and the result
metadata gives their means (`video_frames_mean`, `video_fps_mean`).

With `video_input: video` (the default) the frames reach the model as one video
together with their true sampling rate, so Qwen2.5-VL's absolute time encoding
is correct. This form uses vLLM's `video/jpeg` data URLs. With
`video_input: frames` every frame is sent as an image labelled with its time,
which any server that accepts images understands. Start vLLM with an image limit of
at least `max_frames` for this form, such as `--limit-mm-per-prompt '{"image": 64, "video": 1}'`.

The prompt shows the question, the five options as `A.` to `E.`, and the
instruction "Answer with the option's letter from the given choices directly."
Decoding is greedy. A reply is read as the letter it starts with, else the
letter after "answer is", else the option whose full text it repeats.

| Metric | Meaning |
| --- | --- |
| `acc` | Accuracy on questions with a public answer |
| `acc_ci_low`, `acc_ci_high` | 95% percentile bootstrap interval of `acc` |
| `unparsed` | Share of replies that name no option; they count as wrong |
| `pred_frac_A` … `pred_frac_E` | Share of predictions per letter, which exposes a letter bias |
| `n_scored`, `n_answered` | Questions scored, and those with a public answer |

## Reply cache

Replies are cached under `reply_cache` (default `runs/vlm_replies`, or
`$EGOLEARN_ARTIFACTS/vlm_replies` when that variable is set), one JSON file per
request named by the SHA-256 of the full request. The request includes the model
name, the prompt, every frame and the decoding settings. A rerun of the same
evaluation therefore reads the cache, and any change to the frames, the prompt or
the model name misses it. The cache cannot see the weights behind a name: give
`reply_cache` a new directory when the server serves other weights under the same
name. Empty replies are not cached.
