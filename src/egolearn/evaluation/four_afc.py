"""4AFC probing: one category name, four candidate images, pick the match.

A generic zero-shot readout (protocol from Vong et al.'s CVCL evaluation),
dataset-agnostic like linear_probe: benchmark packages own the trial
manifests (labeled_s_4afc, babyview_4afc), this module owns the scoring.

Protocol decisions, made once here so every dataset scores identically:

* the category name is encoded as ONE bare token -- no <sos>/<eos>.
  Wrapping it dilutes the word embedding with special tokens and costs
  points (the canonical protocol, kept by BabyCL's eval).
* image preprocessing (`view`/`size`) is explicit config, because mixing
  preprocessing across model families was a documented pitfall -- and on
  non-square crops (BabyView) the choice changes the number.
* a trial scores 1 when the target image alone has the highest cosine
  similarity to the text, and 1/k when k images share the highest
  similarity and the target is one of them: the expected score of breaking
  ties at random. Chance is 1/4, and a model that cannot tell the images
  apart scores exactly that, wherever the manifest puts the target.
  Non-finite similarities raise instead of producing a score.

Manifest json format::

    {"trials": [{"text": "ball",
                 "images": ["/path0.jpg", ...4 paths...],
                 "answer": 0}, ...],
     "chance_trials": 0}

``answer`` is the target's index in ``images`` and defaults to 0.
``chance_trials`` is optional. It counts trials that cannot be shown to the
model, such as words outside its vocabulary, and each adds 0.25 of a correct
answer to ``four_afc_acc``. With it set, ``four_afc_acc_scored`` reports the
accuracy on the listed trials alone.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from PIL import Image

from egolearn.config.registry import PROBES, VIEWS
from egolearn.core import keys as K
from egolearn.data.text import VocabTokenizer
from egolearn.evaluation.probe import Probe


@PROBES.register("four_afc")
class FourAFCProbe(Probe):
    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION, K.CAP_TEXT_REPRESENTATION)
    # 2: tied images share the credit; 1 gave every tie to the first image.
    version = "2"

    def __init__(
        self,
        name: str,
        manifest: str,
        vocab_json: str,
        vision_head: str,
        size: int = 224,
        view: str = "center_crop",
        normalize_text: bool = False,
    ):
        self.name = name
        with open(manifest) as f:
            loaded = json.load(f)
        self.trials: list[dict] = loaded["trials"]
        for trial in self.trials:
            answer = trial.get("answer", 0)
            if type(answer) is not int or not 0 <= answer < len(trial["images"]):
                raise ValueError(f"answer must index an image of the {trial['text']!r} trial")
        chance_trials = loaded.get("chance_trials", 0)
        if type(chance_trials) is not int or chance_trials < 0:
            raise ValueError("chance_trials must be a non-negative integer")
        self.chance_trials = chance_trials
        self.tokenizer = VocabTokenizer.from_json(
            vocab_json, add_sos_eos=False, normalize=normalize_text
        )
        self.vision_head = vision_head
        self.transform = VIEWS.get(view)(size=size)

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        correct = 0.0
        for trial in self.trials:
            imgs = []
            for path in trial["images"]:
                with Image.open(path) as f:
                    imgs.append(self.transform(f.convert("RGB")))
            x = torch.stack(imgs, dim=0).to(ctx.device)
            img_feats = F.normalize(
                model.apply_head(self.vision_head, model.encode("image", x).pooled), dim=1
            )

            ids, lens = self.tokenizer.batch([trial["text"]])
            txt = F.normalize(
                model.encode("text", ids.to(ctx.device), lens.to(ctx.device)).pooled, dim=1
            )
            scores = (img_feats @ txt.t()).squeeze(1)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"non-finite 4AFC similarity for {trial['text']!r}")
            # argmax would give every tie (and every NaN) to the first image,
            # which is where the preparers put the target.
            tied = scores == scores.max()
            correct += float(tied[trial.get("answer", 0)]) / int(tied.sum())
        if not self.chance_trials:
            return {"four_afc_acc": correct / max(1, len(self.trials))}
        total = len(self.trials) + self.chance_trials
        return {
            "four_afc_acc": (correct + 0.25 * self.chance_trials) / total,
            "four_afc_acc_scored": correct / max(1, len(self.trials)),
        }
