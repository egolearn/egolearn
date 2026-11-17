# Security and checkpoint trust

Check the source of a checkpoint before loading it, and keep credentials and
restricted data outside the repository. Report vulnerabilities privately so the
maintainers can investigate without exposing users or sensitive information.

## Reporting a vulnerability

Email [mengye@nyu.edu](mailto:mengye@nyu.edu) with a description and reproduction
steps. Exclude credentials and restricted dataset contents from the report.

## Model export and checkpoint trust

Publish model weights with `egl export <checkpoint> <out_dir>`. The export contains
`model.safetensors` and `model.json`, without optimizer, RNG, replay state, or pickle
objects. Model Hub uploads accept this exported pair only; `.pth` checkpoints and
other payload files are refused. Safetensors data is never passed to `torch.load`.

The pickle trust rule applies to legacy `.pth` checkpoints and our own training
snapshots. Evaluation, initialization, and export use PyTorch's restricted
`weights_only=True` reader by default and never retry with unrestricted pickle.
Snapshots contain Python objects: only resume or export snapshots you created or
whose producer and contents you trust. Explicitly opt in with `--trusted-snapshot`
for export, `trusted_snapshot=True` for evaluation, or
`+init_from.trusted_snapshot=true` for initialization. The API opt-in must be a
boolean; strings such as `"false"` are rejected. This enables arbitrary pickle
execution; a hash or Hub download alone does not establish trust.

Model construction metadata and registration imports can execute model code even
when weights use safetensors. Inspect external configurations before loading a
model. Export provenance records the source filename and digest plus the code
revision performing the export; it does not claim to recover the training revision.
Inspect model metadata for restricted information before publishing. Full training
snapshots remain local run artifacts and are never published to model repos.

## Never commit

- Credentials, including API tokens, SSH keys, `.netrc`, and `.env` files.
  Supply keys through environment variables rather than command arguments or configs.
- Personal cluster paths or usernames. Use configuration values and environment variables.
- Access-controlled SAYCam or BabyView frames, audio, transcripts, or identifiable
  stills, including in tests and docs. Tests use synthetic inputs. The exception
  in `landing/media/` is credited material already public under an open license
  or on a dataset's project page, never content extracted from restricted recordings.
- Confidential research material or broad-permission links to restricted files.
- Binaries or files over 500 KB. Store artifacts outside Git, following the
  [Hub access tiers](docs/evaluation/hub.md#access-tiers) before publishing anything.

Gitleaks and the file-size hook check contributions, but removed files remain
in Git history. Review the staged diff before committing new material.

## If a secret lands in a commit

Rotate the credential immediately and notify [mengye@nyu.edu](mailto:mengye@nyu.edu)
privately. Coordinate cleanup with the maintainers. Deleting a file or rewriting
history does not revoke the credential or remove copies already fetched by others.
