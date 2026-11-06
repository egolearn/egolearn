"""Fixed-feature linear decoding on mini-ImageNet (100 classes).

The BabyCL paper fits a linear classifier on frozen 512-d embeddings of
mini-ImageNet's ``train/`` split and reports top-1 on ``val/``: features are
scaled to unit length and the classifier is scikit-learn's
``LogisticRegression(C=0.316, max_iter=1000)``.  ``probe:mini_imagenet_linear``
states the same objective for the shared deterministic L-BFGS scorer.  This is
a different protocol from ``benchmark:imagenet_linear_sgd`` (full ImageNet,
stochastic views, SGD).

The caller supplies an ImageFolder-style root containing ``train/<class>/...``
and ``val/<class>/...``; the images are never downloaded by EgoLearn.

Config::

    _imports_: [egolearn.evaluation.benchmarks.mini_imagenet_linear]
    probes:
      - {_target_: "probe:mini_imagenet_linear", name: mini_imagenet_linear,
         data_root: <mini-imagenet-root>, head: embed512}
"""

from .probe import MiniImageNetLinearProbe

__all__ = ["MiniImageNetLinearProbe"]

# Literal metadata for the generated benchmark reference; keep data terms separate from code.
BENCHMARK_METADATA = {
    "name": "mini-ImageNet linear",
    "source": "[ImageNet](https://www.image-net.org/)",
    "license": "ImageNet access and use terms",
    "access": "upstream-licensed",
    "prepare": (
        "No preparer; supply the 100-class mini-ImageNet release in `train/<class>/` "
        "and `val/<class>/` folders."
    ),
}
