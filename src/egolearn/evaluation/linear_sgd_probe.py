"""Training-hook adapter for the shared stochastic-view SGD linear-decoding protocol."""

from __future__ import annotations

from egolearn.config.registry import PROBES
from egolearn.core import keys as K
from egolearn.evaluation.linear_probe import ClassificationSplit
from egolearn.evaluation.linear_sgd import LinearSGDEvaluator
from egolearn.evaluation.probe import Probe


@PROBES.register("linear_sgd")
class LinearSGDProbe(LinearSGDEvaluator, Probe):
    requires_capabilities = (K.CAP_IMAGE_REPRESENTATION,)

    def __init__(
        self,
        name: str,
        manifest: str | None = None,
        encoder: str = "image",
        head: str | None = None,
        size: int = 112,
        batch_size: int = 1024,
        workers: int = 16,
        epochs: int = 10,
        lr: float = 3.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        larc_trust_coefficient: float | None = 0.001,
        l2_normalize: bool = True,
        seed: int = 0,
        split: ClassificationSplit | None = None,
        view: str = "imagenet_center_crop",
    ):
        self.name = name
        if (manifest is None) == (split is None):
            raise ValueError("linear SGD probe needs exactly one of manifest or split")
        if split is None:
            import json

            assert manifest is not None
            with open(manifest) as file:
                split = ClassificationSplit.from_mapping(json.load(file))
        super().__init__(
            name=name,
            encoder=encoder,
            head=head,
            size=size,
            batch_size=batch_size,
            workers=workers,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            larc_trust_coefficient=larc_trust_coefficient,
            l2_normalize=l2_normalize,
            seed=seed,
            view=view,
        )
        self.split = split
        self.classes = list(split.classes)

    def run(self, model, ctx) -> dict[str, float]:
        return self.evaluate(model, ctx, self.split)
