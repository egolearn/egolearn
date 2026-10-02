"""Visual Two-Word Test (VTWT) benchmark probe.

Introduced by BabyVLM V1: https://shawnking98.github.io/BabyVLM/
Labels: https://github.com/ShawnKing98/babylmms-eval/tree/main/dataset/vtwt
Task attribution: BabyVLM / babylmms-eval, Apache-2.0:
https://github.com/ShawnKing98/babylmms-eval/blob/main/LICENSE
Data access and preparation: docs/evaluation.md#babyvlm-benchmark-data

Each sample contains one image and two two-word phrases. A sample is correct
when the positive phrase has strictly higher cosine similarity to the image.

CSV format::

    image,positive_phrase,negative_phrase
    S_20140420_1803_01_082_00.jpg,see person,touch pillow

Protocol:
  * Tokenization uses phrase.strip().lower().split()[:32]. Out-of-vocabulary
    words map to padding index 0, a fixed zero embedding, and count toward the
    mean-pool denominator. No <sos>/<eos> markers are added.
  * Images use RGB, bicubic Resize(224), CenterCrop(224), and model-specific
    normalization: `imagenet` for EgoLearn-trained BabyCL models, `clip` for
    the CVCL family.
  * Missing images are skipped and counted.
"""

from __future__ import annotations

import csv
import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from egolearn.config.registry import PROBES
from egolearn.core import keys as K
from egolearn.evaluation.probe import Probe

NORMS = {
    "clip": dict(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    ),
    "imagenet": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


def encode_phrase(
    phrase: str, word2idx: dict, device, max_len: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lowercased token IDs and pooling length, including OOV tokens."""
    tokens = phrase.strip().lower().split()[:max_len]
    ids = [word2idx.get(tok, 0) for tok in tokens]  # 0 = padding_idx / OOV
    length = max(len(ids), 1)
    ids_t = torch.tensor([ids] if ids else [[0]], dtype=torch.long, device=device)
    len_t = torch.tensor([length], dtype=torch.float, device=device)
    return ids_t, len_t


def eval_transform(size: int, image_norm: str) -> T.Compose:
    norm = NORMS[image_norm]
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


@PROBES.register("vtwt")
class VTWTProbe(Probe):
    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION, K.CAP_TEXT_REPRESENTATION)

    def __init__(
        self,
        name: str,
        csv_path: str,
        image_dir: str,
        vocab_json: str,
        vision_head: str,
        image_norm: str = "imagenet",
        size: int = 224,
        output_json: str | None = None,
    ):
        self.name = name
        self.csv_path = csv_path
        self.image_dir = image_dir
        with open(vocab_json) as f:
            self.word2idx = json.load(f)
        self.vision_head = vision_head
        self.transform = eval_transform(size, image_norm)
        self.output_json = output_json
        with open(csv_path, newline="") as f:
            self.rows: list[dict] = list(csv.DictReader(f))

    @torch.no_grad()
    def run(self, model, ctx) -> dict[str, float]:
        n_correct = n_total = n_skipped = 0
        per_sample = []
        for row in self.rows:
            img_path = os.path.join(self.image_dir, row["image"].strip())
            if not os.path.isfile(img_path):
                n_skipped += 1
                continue
            with Image.open(img_path) as f:
                x = self.transform(f).unsqueeze(0).to(ctx.device)

            img_feat = F.normalize(
                model.apply_head(self.vision_head, model.encode("image", x).pooled), dim=-1
            )

            sims = []
            for phrase in (row["positive_phrase"].strip(), row["negative_phrase"].strip()):
                ids, lens = encode_phrase(phrase, self.word2idx, ctx.device)
                txt = F.normalize(model.encode("text", ids, lens).pooled, dim=-1)
                sims.append(float((img_feat * txt).sum()))
            correct = sims[0] > sims[1]  # strict >
            n_correct += int(correct)
            n_total += 1
            per_sample.append(
                {
                    "image": row["image"].strip(),
                    "sim_pos": round(sims[0], 6),
                    "sim_neg": round(sims[1], 6),
                    "correct": bool(correct),
                }
            )

        acc = n_correct / n_total if n_total else 0.0
        if self.output_json:
            os.makedirs(os.path.dirname(self.output_json) or ".", exist_ok=True)
            with open(self.output_json, "w") as f:
                json.dump(
                    {
                        "accuracy": acc,
                        "n_correct": n_correct,
                        "n_total": n_total,
                        "n_skipped": n_skipped,
                        "per_sample": per_sample,
                    },
                    f,
                    indent=2,
                )
        return {"vtwt_acc": acc, "vtwt_n": float(n_total), "vtwt_skipped": float(n_skipped)}
