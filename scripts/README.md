# scripts/

Operational scripts for this repo. **Launch scripts are committed, never gitignored** —
cluster launchers (SLURM sbatch templates, workstation runners) live here so they are
versioned with the code they launch.

| Script | Purpose |
| --- | --- |
| `check_pr_message.py` | Check a draft PR's title and body against CONTRIBUTING.md; stdlib-only local helper, not a CI gate. `--files` enables the experiment Results warning; `--pr N` reads an existing PR with `gh`. |
| `setup_branch_protection.sh` | Apply/refresh branch protection on `main` (see header for usage). Gate: 1 approving review from a code owner + 5 green checks. `enforce_admins=false` is a setup-phase-only override — re-run with defaults to restore admin enforcement. |
| `slurm/submit` | Snapshot, prepare and submit container jobs and sweep arrays ([setup](../docs/cluster.md)) |
| `slurm/train.sbatch` | Compatibility wrapper for the submit command |
| `slurm/bench.sbatch` | Data-plane benchmark wrapper using the same snapshot/prep path |
| `bench/bench_dataloading.py` | Data-plane-only benchmark: ingest rate, batch-composition throughput, decode-vs-augment micro breakdown |
| `bench/bench_full_pipeline.py` | N real training steps of any config + the per-stage latency ledger |

Planned (see `docs/roadmap.md` phase 6): hydra+submitit launching and per-run
code snapshotting.

- `fetch_data.py`: wrapper for `python -m egolearn.hub` (fetch, push, verify, checksum); see [Hub artifacts](../docs/evaluation.md#hub-artifacts-and-access-tiers).
