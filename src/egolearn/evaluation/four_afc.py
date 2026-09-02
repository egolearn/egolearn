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
* a trial is correct iff the target image has the highest cosine
  similarity to the text; chance = 1/4.

Manifest json format::

    {"trials": [{"text": "ball",
                 "images": ["/path0.jpg", ...4 paths...],
                 "answer": 0}, ...]}
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
            self.trials: list[dict] = json.load(f)["trials"]
        self.tokenizer = VocabTokenizer.from_json(
            vocab_json, add_sos_eos=False, normalize=normalize_text
        )
        self.vision_head = vision_head
        self.transform = VIEWS.get(view)(size=size)

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        correct = 0
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
            pred = int(torch.argmax(img_feats @ txt.t()))
            correct += int(pred == int(trial.get("answer", 0)))
        return {"four_afc_acc": correct / max(1, len(self.trials))}
