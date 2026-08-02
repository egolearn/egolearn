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

## If a secret lands in a commit

1. **Rotate it immediately** — assume it is compromised the moment it is pushed.
2. Tell the PI.
3. While the repo is still private, a coordinated history rewrite is possible.
   After the public flip it is not; rotation is the only remedy.

## Reporting a vulnerability

Email the PI: mengye@nyu.edu.
