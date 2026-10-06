# scripts/slurm/

Cluster-independent Slurm launchers. Anything cluster-specific — account,
partition, GPU type, data locations — is supplied at submit time, never
committed:

```bash
sbatch --account=<account> --gres=gpu:<type>:1 scripts/slurm/train.sbatch \
    projects/babycl/configs/saycam_default.yaml \
    stream.root_dir=$DATA_ROOT stream.transcript_json=$TRANSCRIPT_JSON \
    tokenizer.vocab_json=$VOCAB_JSON
```

## Minimal cluster setup

On a login node, once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
git clone git@github.com:agentic-learning-ai-lab/egolearn.git && cd egolearn
uv sync --extra train
uv run pytest            # sanity: unit tests pass on CPU
```

The venv lives inside the repo on the shared filesystem, so compute nodes
use it directly (`uv run --no-sync` in the templates skips re-resolving on
nodes without internet). PyPI's Linux torch wheels bundle the CUDA runtime;
the compute node only needs a recent NVIDIA driver.

Home directories often have small quotas — if the torch download blows it,
point the cache at scratch first: `export UV_CACHE_DIR=$SCRATCH/.cache/uv`.

W&B on air-gapped compute nodes: `export WANDB_MODE=offline` before sbatch
(or set `wandb.mode=offline` in the config) and `wandb sync` afterwards.

## Apptainer route (fallback)

If the bare-venv route fails on your cluster (old glibc, driver/toolkit
mismatch), run the same commands inside any recent CUDA container:

```bash
apptainer exec --nv <cuda-image.sif> \
    uv run --no-sync egl train projects/babycl/configs/saycam_default.yaml
```

Mount data with `--overlay <data.sqf>:ro` or `--bind` as your site requires.
Keep site-specific image/overlay paths in your submit command or a private
wrapper — not in this repo.
