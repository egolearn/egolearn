# Cluster image

`IMAGE` pins the CUDA 13.0 base image used by the Slurm launcher, including
FFmpeg, jemalloc, and uv. You can use this image without building an
EgoLearn-specific container.

See [cluster setup](../docs/cluster.md) for pulling the image through a CPU job,
configuring storage, and checking the runtime and driver on your site.
