# Security and repo hygiene

This repo's full git history becomes public at first release and **can never be
rewritten afterward** (history is shared with the private mirror). Prevention is the
only defense — hence gitleaks in pre-commit and CI, and the 500 KB file blocker.

## Never commit

- Credentials of any kind: API keys (OpenAI, Anthropic, W&B, HF), SSH keys, tokens,
  `.netrc`, `.env` files. **LLM keys are environment variables only — never CLI
  flags** (they leak into shell history and `ps` output), never config values.
- Personal cluster paths or usernames (`/scratch/<netid>`, `/home/<netid>`,
  `/vast/<netid>`) — use config values or environment variables.
- Egocentric video data or frames — SAYCam and BabyView are **access-controlled
  human-subject datasets** (IRB / data-use agreements); no frames, captions, or
  identifiable stills ever enter the repo, including in tests and docs. Tests use
  synthetic clips only.
- Reviewer identities, review text, or rebuttal drafts.
- Unpublished paper text or figures (that belongs in Overleaf).
- Binaries or files >500 KB — checkpoints, datasets, videos go to the Hub
  (docs/roadmap.md). The pre-commit hook blocks these.
- Broad-permission share links (e.g. anyone-with-link Drive folders).

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

## If a secret lands in a commit

1. **Rotate it immediately** — assume it is compromised the moment it is pushed.
2. Tell the PI.
3. While the repo is still private, a coordinated history rewrite is possible.
   After the public flip it is not; rotation is the only remedy.

## Reporting a vulnerability

Email the PI: mengye@nyu.edu.
