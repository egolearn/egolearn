# Shared cluster image

`IMAGE` pins the lab's public CUDA 13.0 base image, including FFmpeg, jemalloc
and uv. No EgoLearn-specific Dockerfile is needed. The digest is taken from
[jepa-agent PR #94](https://github.com/agentic-learning-ai-lab/jepa-agent/pull/94)
at `66ac9417a8bd808b7047b46706b70fe4c767678c`.

See [cluster setup](../docs/cluster.md) for pulling through a CPU prep job,
cache placement, driver requirements and site-provided images.
