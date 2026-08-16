"""benchmarks/: first-class home for evaluation benchmarks (EgoLearn aim 1).

Core owns the Probe protocol and generic probes; each benchmark directory
owns its dataset protocol, probe implementation, and (later) leaderboard
scripts.  Benchmarks are neither core (they are concrete datasets/protocols)
nor projects (they belong to no method -- every method is scored on them).

Available: labeled_s_4afc/ (probe:four_afc), vtwt/ (probe:vtwt),
baby_winoground/ (probe:baby_winoground).
Roadmap homes: linear_probe_imagenet/, rvs_ego/, egolife/, intphys/ ...
"""
