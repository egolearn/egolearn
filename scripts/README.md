# scripts/

Operational scripts for this repo. **Launch scripts are committed, never gitignored** —
cluster launchers (SLURM sbatch templates, workstation runners) live here so they are
versioned with the code they launch.

| Script | Purpose |
| --- | --- |
| `setup_branch_protection.sh` | Apply/refresh branch protection on `main` (see header for usage). Gate: 1 approving review from a code owner + 5 green checks. `enforce_admins=false` is a setup-phase-only override — re-run with defaults to restore admin enforcement. |

Planned (see `docs/roadmap.md` phase 6): `torch/` SLURM launchers for the NYU Torch
cluster accounts and the lab workstation, with per-run code snapshotting.
