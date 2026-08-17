"""Labeled-S 4AFC benchmark probe (protocol from Vong et al.).

4AFC = four-alternative forced choice: the model is given one category name
and four candidate images and must pick the matching image.  Registered as
probe:four_afc; configs load it via
`_imports_: [egolearn.evaluation.benchmarks.labeled_s_4afc]`.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from PIL import Image

from egolearn.config.registry import PROBES, VIEWS
from egolearn.data.text import VocabTokenizer
from egolearn.evaluation.probe import Probe


@PROBES.register("four_afc")
class FourAFCProbe(Probe):
    """4AFC in image mode: one target category as text, four candidate
    images; a trial counts as correct iff the target image has the highest
    cosine similarity to the text.

    Manifest json format::

        {"trials": [{"text": "ball",
                     "images": ["/path0.jpg", ...4 paths...],
                     "answer": 0}, ...]}

    Eval preprocessing is an explicit config field (`view`/`size`) because
    mixing preprocessing across model families was a documented pitfall.
    """

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
        self.tokenizer = VocabTokenizer.from_json(vocab_json, normalize=normalize_text)
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
            img_feats = F.normalize(model.apply_head(self.vision_head, model.vision_tap(x)), dim=1)

            ids, lens = self.tokenizer.batch([trial["text"]])
            txt = F.normalize(model.text_tap(ids.to(ctx.device), lens.to(ctx.device)), dim=1)
            pred = int(torch.argmax(img_feats @ txt.t()))
            correct += int(pred == int(trial.get("answer", 0)))
        return {"four_afc_acc": correct / max(1, len(self.trials))}
