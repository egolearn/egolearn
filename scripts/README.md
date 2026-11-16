# scripts/

Operational scripts for this repo. **Launch scripts are committed, never gitignored** —
cluster launchers (SLURM sbatch templates, workstation runners) live here so they are
versioned with the code they launch.

| Script | Purpose |
| --- | --- |
| `check_pr_message.py` | Check a draft PR's title and body against CONTRIBUTING.md; stdlib-only local helper, not a CI gate. `--files` enables the experiment Results warning; `--pr N` reads an existing PR with `gh`. |
| `slurm/submit` | Snapshot, prepare and submit container jobs and sweep arrays ([setup](../docs/cluster.md)) |
| `slurm/train.sbatch` | Compatibility wrapper for the submit command |
| `slurm/bench.sbatch` | Data-plane benchmark wrapper using the same snapshot/prep path |
| `bench/bench_dataloading.py` | Data-plane-only benchmark: ingest rate, batch-composition throughput, decode-vs-augment micro breakdown |
| `bench/bench_full_pipeline.py` | N real training steps of any config + the per-stage latency ledger |

The Slurm launcher archives a clean commit for each submission and runs jobs from
that snapshot. [Cluster setup](../docs/cluster.md) covers environment preparation,
storage, and arrays.

- `fetch_data.py`: wrapper for `python -m egolearn.hub` (fetch, push, verify, checksum); see [Hub artifacts](../docs/evaluation.md#hub-artifacts-and-access-tiers).

Use `egl data prepare NAME` and `egl data fetch NAME` on workstations or compute
nodes. On login nodes, use `scripts/slurm/submit` to queue commands; run Python,
including `egl doctor` and benchmark planning, inside compute allocations.
See [the CLI guide](../docs/cli.md) and [benchmark suites](../docs/benchmark_suites.md).
