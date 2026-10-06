# Slurm launchers

See [cluster setup and submission](../../docs/cluster.md). The front door is:

```bash
source env.sh
egl submit [sbatch options] -- egl train CONFIG [Hydra options]
```

`submit` snapshots a clean commit, prepares the pinned container/environment in a
CPU job when needed, and submits the requested command with a dependency.
`--sweep` and `--commands` create one array. `--dry-run` prints the planned calls.
`train.sbatch` and `bench.sbatch` are compatibility wrappers to run on the login node without sbatch (one GPU by default);
`prep.sbatch`, `run.sbatch` and `run_array.sbatch` are internal entry points.

The scripts are adapted from the MIT-licensed jepa-agent cluster stack at commit
`a7124e81efc81eae88efdb889da0a52c33f562a8`; the shared lab copyright and MIT license
are retained in the repository's LICENSE. EgoLearn adds immutable per-run code
archives, commit-keyed wheel environments and per-task working directories.
