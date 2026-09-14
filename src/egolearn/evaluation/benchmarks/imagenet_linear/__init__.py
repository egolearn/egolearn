"""ImageNet gradient-based linear decoding.

The probe freezes the image encoder and trains a randomly initialized linear
classifier on the full ImageNet training split.  Each epoch recomputes
L2-normalized features from random resized crops, then updates the classifier
with cross-entropy, SGD, LARC, and cosine learning-rate decay.  Top-1 accuracy
is measured using the registered evaluation ``view``. The default
``imagenet_center_crop`` uses bilinear resize to ``256 * size // 224`` followed
by a center crop to ``size``, preserving the original gradient protocol.

The caller supplies an ImageFolder-style root containing ``train/<class>/...``
and ``val/<class>/...``; ImageNet itself is never downloaded by EgoLearn.
The adapter defaults to 224-pixel crops, matching EgoLearn's BabyCL training
resolution.  Set ``size: 112`` to reproduce Memory Storyboard's released
evaluation command.

Set ``view: center_crop`` or ``view: resize`` to use the same evaluation
preprocessing as other probes with that view and size. ``view`` controls only
validation images; training continues to use random resized crops and flips.

Config::

    _imports_: [egolearn.evaluation.benchmarks.imagenet_linear]
    probes:
      - {_target_: "probe:imagenet_linear_sgd", name: imagenet_linear,
         data_root: <imagenet-root>, encoder: vision,
         view: imagenet_center_crop, size: 224,
         batch_size: 1024, workers: 16, epochs: 10}
"""

from .probe import ImageNetLinearProbe

__all__ = ["ImageNetLinearProbe"]
