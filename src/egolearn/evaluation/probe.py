"""Probe protocol: evaluations that can run at any point during training.

A probe (an evaluation routine) only touches the model through named
encoders and heads, never through internals.  Every checkpoint records
those names, so any checkpoint -- from any method -- can be evaluated the
same way.

Probes also declare what a model must be ABLE TO DO for the evaluation to
make sense (`requires_capabilities`), and the pipeline checks that against
`model.capabilities()` when it is built.  So an image-text retrieval probe
wired against a vision-only model fails at startup with a clear message,
instead of every model x evaluation pair needing its own adapter.  Current
capability names: image_representation, video_representation,
text_representation, multimodal, prediction, generation.

Benchmark-specific probes live with their benchmark or project and register
into the PROBES registry (e.g. probe:four_afc in
benchmarks/labeled_s_4afc/__init__.py).  Core ships only this protocol.
"""

from __future__ import annotations


class Probe:
    name = "probe"
    # capabilities the model must have for this evaluation to apply
    requires_capabilities: tuple[str, ...] = ()

    def run(self, model, ctx) -> dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError
