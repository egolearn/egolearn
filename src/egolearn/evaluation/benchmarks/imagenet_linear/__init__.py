"""ImageNet gradient-based linear decoding.

This ports Memory Storyboard's full-ImageNet ``linear_decoding.py`` protocol,
not its separate 100-class mini-ImageNet SVM/kNN monitor.  The caller supplies
an ImageFolder-style root containing ``train/<class>/...`` and
``val/<class>/...``; ImageNet itself is never downloaded by EgoLearn.

Config::

    _imports_: [egolearn.evaluation.benchmarks.imagenet_linear]
    probes:
      - {_target_: "probe:imagenet_linear_sgd", name: imagenet_linear,
         data_root: <imagenet-root>, encoder: vision,
         size: 112, batch_size: 1024, workers: 16, epochs: 10}
"""

from .probe import ImageNetLinearProbe

__all__ = ["ImageNetLinearProbe"]
