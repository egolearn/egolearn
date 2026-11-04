# Ego4D NLQ

Natural language queries come from the Ego4D episodic-memory benchmark. Given an
egocentric clip of about 8 minutes and a question such as "Where did I put the
scissors?", a model names the time window in which the answer can be seen.
EgoLearn scores a generative model that watches the clip and replies with the
window's start and end in seconds, without training anything. The headlines are
recall@1 at IoU 0.3 and 0.5 on the official validation split, with the mean IoU
of the predicted window.

The benchmark is Grauman et al.,
[*Ego4D: Around the World in 3,000 Hours of Egocentric Video*](https://arxiv.org/abs/2110.07058)
(CVPR 2022). The v2 annotations have 415 validation clips with 4,552 queries,
and 333 test clips whose answers are hidden.

## Downloading the data

Ego4D requires a signed license agreement (see [ego4d-data.org](https://ego4d-data.org/)),
and the download credentials it issues expire after 14 days. With the Ego4D CLI:

```bash
ego4d --output_directory "$EGOLEARN_DATA/ego4d" --version v2 \
  --datasets annotations clips --benchmarks nlq
```

This gives `$EGOLEARN_DATA/ego4d/v2/annotations/nlq_{train,val,test_unannotated}.json`
and one file per clip, `$EGOLEARN_DATA/ego4d/v2/clips/<clip_uid>.mp4`. The
benchmark checks that every clip it scores is present before asking the model. It
also checks the annotation file against the SHA-256 of the official v2 release.
Set `benchmarks.0.verify=false` to score a subset.

## Running the evaluation

Start a server as described for [EgoSchema](egoschema.md#serving-the-model).
Ego4D frames go only to servers the lab runs itself. Install the `train` and
`qa` extras (`uv sync --extra train --extra qa`), then give the server's address:

```bash
uv run egl eval ego4d_nlq_eval vlm_url=http://<host>:8000/v1 \
  output_json=nlq.json
```

For the challenge's test split, set `benchmarks.0.split=test`, point
`benchmarks.0.annotations_json` at `nlq_test_unannotated.json` and set
`benchmarks.0.predictions_dir`. The directory receives `predictions.json` in the
challenge format and `replies.jsonl` with every reply. `benchmarks.0.max_clips`
limits a run to the first clips of the file, for smoke tests.

## Scoring contract

Each clip is sampled at `fps` frames per second (default 0.5), at most
`max_frames` frames (default 128), so an 8-minute clip gives 128 frames about
3.75 s apart. Frames start at the clip's first frame, are evenly spaced and are
resized so their longer side is at most `max_side` pixels (default 448). Each
prediction row records the frame count and rate the model saw. The
model is told the true sampling rate, as for [EgoSchema](egoschema.md#scoring-contract),
so its times count from the start of the clip like the annotations.

The default prompt, `start_end_seconds`, gives the clip's duration and the query,
and asks for the start and end time in the form "start - end". Decoding is
greedy. The default parser, `time_pairs`, reads every pair of times joined by "-", "to", "and", "," or "~", in seconds or as
clock times such as `1:05`. Each window is clipped to the clip and put in
order, and the first five are kept. A reply without a pair counts as a miss.
Other prompts and parsers are chosen with `benchmarks.0.prompt` and
`benchmarks.0.parser`, as described for [EgoSchema](egoschema.md#scoring-contract).

The metrics follow the official `evaluate_ego4d_nlq.py`. A query is recalled at
(k, t) when one of its first k windows overlaps the ground truth with IoU
strictly above t. The prompt asks for one window, so recall@5 equals recall@1
unless the model offers several. In the prediction file, a reply with fewer
than five windows repeats its last one, and a reply with none is submitted as
`[0, 0]`.

| Metric | Meaning |
| --- | --- |
| `r1_iou0.3`, `r1_iou0.5` | Recall@1 at IoU 0.3 and 0.5 |
| `r5_iou0.3`, `r5_iou0.5` | Recall@5 at IoU 0.3 and 0.5 |
| `miou` | Mean IoU of the first window |
| `unparsed` | Share of replies without a window |
| `n_queries`, `n_scored` | Queries asked, and those with a ground-truth window |

Replies are cached as described for [EgoSchema](egoschema.md#reply-cache).
