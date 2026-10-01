"""Baby Winoground benchmark probe.

Introduced by BabyVLM: https://shawnking98.github.io/BabyVLM/
Labels: https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/baby_winoground
Synthetic negatives: https://nyu.databrary.org/volume/564/folder/77120
Task attribution: BabyVLM / babylmms-eval, Apache-2.0:
https://github.com/ShawnKing98/babylmms-eval/blob/main/LICENSE
Data access and preparation: docs/evaluation.md#babyvlm-benchmark-data

1:1 port of BabyCL's original evaluation script (protocol from Wang et al.,
BabyVLM).  Each instance pairs (pos_image, positive_phrase) with a
synthetically perturbed (neg_image, negative_phrase).  All four image-phrase
cosine similarities are computed, and the instance is GROUP-correct only
when both of these hold (strict >):

    sim(pos_img, pos_phrase) > sim(pos_img, neg_phrase)     # cond1 / Positive Ctx
    sim(neg_img, neg_phrase) > sim(neg_img, pos_phrase)     # cond2 / Negative Ctx

CSV format (rows grouped by instance_id)::

    instance_id,positive_phrase,negative_phrase,image,image_tag
    0,see person,touch pillow,S_..._00.jpg,positive
    0,see person,touch pillow,S_..._00_neg.jpeg,negative

Positive images are looked up in `pos_dir` (SAYCam frames); negative images
in `neg_dir` (the synthetic negatives directory).  Tokenization and
preprocessing are shared verbatim with the VTWT probe (out-of-vocabulary
words map to padding index 0, no <sos>/<eos>, normalization chosen per model
family); instances with a missing row or image are skipped and counted, as
in the original.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

from egolearn.config.registry import PROBES
from egolearn.core import keys as K
from egolearn.evaluation.probe import Probe

from ..vtwt import encode_phrase, eval_transform


@PROBES.register("baby_winoground")
class BabyWinogroundProbe(Probe):
    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION, K.CAP_TEXT_REPRESENTATION)

    def __init__(
        self,
        name: str,
        csv_path: str,
        pos_dir: str,
        neg_dir: str,
        vocab_json: str,
        vision_head: str,
        image_norm: str = "imagenet",
        size: int = 224,
        output_json: str | None = None,
    ):
        self.name = name
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        with open(vocab_json) as f:
            self.word2idx = json.load(f)
        self.vision_head = vision_head
        self.transform = eval_transform(size, image_norm)
        self.output_json = output_json

        instances: defaultdict[int, dict] = defaultdict(dict)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                iid = int(row["instance_id"])
                tag = row["image_tag"].strip()  # "positive" | "negative"
                instances[iid]["positive_phrase"] = row["positive_phrase"].strip()
                instances[iid]["negative_phrase"] = row["negative_phrase"].strip()
                instances[iid][f"img_{tag}"] = row["image"].strip()
        self.instances = dict(instances)

    def _image_feat(self, model, ctx, path: str) -> torch.Tensor:
        with Image.open(path) as f:
            x = self.transform(f).unsqueeze(0).to(ctx.device)
        return F.normalize(
            model.apply_head(self.vision_head, model.encode("image", x).pooled), dim=-1
        )

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        n_correct = n_cond1 = n_cond2 = n_total = n_skipped = 0
        per_instance = []
        for iid in sorted(self.instances):
            inst = self.instances[iid]
            if "img_positive" not in inst or "img_negative" not in inst:
                n_skipped += 1
                continue
            pos_path = os.path.join(self.pos_dir, inst["img_positive"])
            neg_path = os.path.join(self.neg_dir, inst["img_negative"])
            if not (os.path.isfile(pos_path) and os.path.isfile(neg_path)):
                n_skipped += 1
                continue

            pos_img = self._image_feat(model, ctx, pos_path)
            neg_img = self._image_feat(model, ctx, neg_path)
            txts = []
            for phrase in (inst["positive_phrase"], inst["negative_phrase"]):
                ids, lens = encode_phrase(phrase, self.word2idx, ctx.device)
                txts.append(F.normalize(model.encode("text", ids, lens).pooled, dim=-1))
            pos_txt, neg_txt = txts

            s_i1_c1 = float((pos_img * pos_txt).sum())
            s_i1_c2 = float((pos_img * neg_txt).sum())
            s_i2_c1 = float((neg_img * pos_txt).sum())
            s_i2_c2 = float((neg_img * neg_txt).sum())

            cond1 = s_i1_c1 > s_i1_c2  # pos image prefers pos phrase
            cond2 = s_i2_c2 > s_i2_c1  # neg image prefers neg phrase
            correct = cond1 and cond2

            n_correct += int(correct)
            n_cond1 += int(cond1)
            n_cond2 += int(cond2)
            n_total += 1
            per_instance.append(
                {
                    "instance_id": iid,
                    "cond1_correct": cond1,
                    "cond2_correct": cond2,
                    "group_correct": correct,
                    "s_posimg_posphrase": round(s_i1_c1, 6),
                    "s_posimg_negphrase": round(s_i1_c2, 6),
                    "s_negimg_posphrase": round(s_i2_c1, 6),
                    "s_negimg_negphrase": round(s_i2_c2, 6),
                }
            )

        group = n_correct / n_total if n_total else 0.0
        cond1_acc = n_cond1 / n_total if n_total else 0.0
        cond2_acc = n_cond2 / n_total if n_total else 0.0
        if self.output_json:
            os.makedirs(os.path.dirname(self.output_json) or ".", exist_ok=True)
            with open(self.output_json, "w") as f:
                json.dump(
                    {
                        "group_acc": group,
                        "cond1_acc": cond1_acc,
                        "cond2_acc": cond2_acc,
                        "n_correct": n_correct,
                        "n_total": n_total,
                        "n_skipped": n_skipped,
                        "per_instance": per_instance,
                    },
                    f,
                    indent=2,
                )
        return {
            "bw_group_acc": group,
            "bw_pos_ctx_acc": cond1_acc,  # paper: Positive Ctx
            "bw_neg_ctx_acc": cond2_acc,  # paper: Negative Ctx
            "bw_n": float(n_total),
            "bw_skipped": float(n_skipped),
        }
