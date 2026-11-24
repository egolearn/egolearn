# Ego4D NLQ

Natural language queries (NLQ) come from the Ego4D episodic-memory benchmark.
Given an egocentric clip of about 8 minutes and a query such as "Where did I
put the scissors?", a model names the time window in which the answer can be
seen. In egolearn, a generative vision-language model watches the clip and
replies with the window's start and end in seconds, without any training.
The headlines are recall@1 at IoU 0.3 and 0.5 on the validation split.

The benchmark is Grauman et al.,
[*Ego4D: Around the World in 3,000 Hours of Egocentric Video*](https://arxiv.org/abs/2110.07058)
(CVPR 2022). Its v2 annotations have 4,552 validation queries over 415 clips,
and 333 test clips whose answers are hidden.

## Downloading the data

Ego4D requires a signed license agreement through
[ego4d-data.org](https://ego4d-data.org/), which also issues the download
credentials. Install the [Ego4D CLI](https://ego4d-data.org/docs/start-here/),
then download the NLQ annotations and clips:

```bash
export EGOLEARN_DATA=/path/to/data
ego4d --output_directory "$EGOLEARN_DATA/ego4d" --version v2 \
  --datasets annotations clips --benchmarks nlq
```

This gives `$EGOLEARN_DATA/ego4d/v2/annotations/nlq_{train,val,test_unannotated}.json`
and one `$EGOLEARN_DATA/ego4d/v2/clips/<clip_uid>.mp4` per clip. The benchmark
checks the annotation file against the official v2 release, and checks that
every clip it scores is present. Set `benchmarks.0.verify=false` if you score
a subset of the annotations.

## Running the evaluation

Start a server as described for [EgoSchema](egoschema.md#serving-the-model).
Ego4D frames go only to a server that you operate under the dataset's terms.
Install the `train` and `qa` extras, then run the packaged config:

```bash
uv sync --extra train --extra qa
uv run egl eval ego4d_nlq_eval vlm_url=http://<host>:8000/v1 output_json=nlq.json
```

| Override | Effect |
| --- | --- |
| `benchmarks.0.split=test benchmarks.0.annotations_json=<path to nlq_test_unannotated.json> benchmarks.0.predictions_dir=<dir>` | Writes `predictions.json` in the challenge format and `replies.jsonl` with every reply, for the hidden test split |
| `benchmarks.0.max_clips=<n>` | Scores only the first n clips, for smoke tests |

The model name, frame input and reply cache work as for
[EgoSchema](egoschema.md#running-the-evaluation).

## Metrics

The metrics follow the official `evaluate_ego4d_nlq.py`. A query counts as
recalled at k and threshold t when one of its first k predicted windows
overlaps the ground truth with IoU above t.

| Metric | Meaning |
| --- | --- |
| `r1_iou0.3`, `r1_iou0.5` | Recall@1 at IoU 0.3 and 0.5 |
| `r5_iou0.3`, `r5_iou0.5` | Recall@5 at IoU 0.3 and 0.5 |
| `miou` | Mean IoU of the first window |
| `unparsed` | Share of replies without a window. They count as misses |
| `n_queries`, `n_scored` | Queries asked, and those with a ground-truth window |

## Differences from the reference protocol

- The model is a zero-shot vision-language model, not a trained grounding
  model. It sees frames sampled at 0.5 frames per second, up to 128 frames,
  with the longer side at most 448 pixels, and is told the sampling rate.
- The default prompt asks for one window, so recall@5 equals recall@1 unless
  the model offers several. The challenge file repeats a reply's last window up
  to five, and records a reply without one as `[0, 0]`.
