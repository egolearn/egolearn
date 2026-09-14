"""benchmarks/: home for the evaluation benchmarks (EgoLearn aim 1).

Core owns the Benchmark and Probe protocols and generic scorers; each
benchmark directory owns its dataset format and evaluation implementation.
The standalone evaluator runs benchmarks and existing probes on checkpoints.
Benchmarks sit apart from both core and the method projects: they
are concrete datasets rather than shared infrastructure, and they belong to
no single method -- every method is scored on them.

Available: labeled_s_4afc/ and babyview_4afc/ (trial manifests for the
generic probe:four_afc; babyview also ships a probe:linear split),
labeled_s_linear/ (fixed-feature L-BFGS linear decoding), vtwt/
(probe:vtwt), baby_winoground/ (probe:baby_winoground), shape_bias/
(probe:shape_bias), and imagenet_linear/ (benchmark:imagenet_linear_sgd).
Planned: rvs_ego/, egolife/, intphys/ ...
"""
