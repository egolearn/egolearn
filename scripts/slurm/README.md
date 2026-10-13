# Slurm launchers

See [cluster setup and submission](../../docs/cluster.md). On the login node, use:

```bash
source env.sh
scripts/slurm/submit [sbatch options] -- egl train CONFIG [Hydra options]
```

`submit` snapshots a clean commit, prepares the pinned container/environment in a
CPU job when needed, and submits the requested command with a dependency.
The command after `--` runs on a compute node. Keep Python off login nodes:
`egl submit`, `egl doctor` and `egl bench` require a workstation or compute allocation.
See [CPU benchmark planning](../../docs/benchmark_suites.md#cluster-planning-in-a-cpu-job).
`--sweep` and `--commands` create one array. `--dry-run` prints the planned calls.
`train.sbatch` and `bench.sbatch` are compatibility wrappers to run on the login node without sbatch (one GPU by default);
`prep.sbatch`, `run.sbatch` and `run_array.sbatch` are internal entry points.

The scripts are adapted from the MIT-licensed jepa-agent cluster stack at commit
`a7124e81efc81eae88efdb889da0a52c33f562a8`; the shared lab copyright and MIT license
are retained in the repository's LICENSE. EgoLearn adds immutable per-run code
archives, commit-keyed wheel environments and per-task working directories.
