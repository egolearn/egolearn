"""benchmarks/: home for the evaluation benchmarks (EgoLearn aim 1).

Core owns the Probe protocol and generic probes; each benchmark directory
owns its dataset format, probe implementation, and (later) leaderboard
scripts.  Benchmarks sit apart from both core and the method projects: they
are concrete datasets rather than shared infrastructure, and they belong to
no single method -- every method is scored on them.

Available: labeled_s_4afc/ (probe:four_afc), vtwt/ (probe:vtwt),
baby_winoground/ (probe:baby_winoground), shape_bias/ (probe:shape_bias).
Planned: linear_probe_imagenet/, rvs_ego/, egolife/, intphys/ ...
"""
