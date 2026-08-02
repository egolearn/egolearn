# Releasing

## Versioning

- SemVer, 0.x while APIs are settling (minor = features, patch = fixes).
- Single version source: `__version__` in `src/egolearn/__init__.py` (hatch reads it).
- `CHANGELOG.md` follows Keep a Changelog; PRs append to `[Unreleased]`.

## Cutting a release

1. PR: bump `__version__`, move `[Unreleased]` entries under the new version heading.
2. After merge: `git tag vX.Y.Z && git push origin vX.Y.Z`, then
   `gh release create vX.Y.Z --notes-from-tag` (or paste the changelog section).

## Go-public flip checklist

Run when the first paper is released from this repo. **After the flip, history is
immutable** — it is shared with the `-dev` mirror, and a rewrite would desynchronize
the two repos forever. Prevention (gitleaks in pre-commit + CI) is the real defense;
this checklist is the final audit.

1. **Freeze merges** (announce in the lab).
2. **Move unreleased work off this repo**: create the `-dev` mirror (below), push every
   branch there, then delete all non-releasable branches from this repo — **the flip
   publishes every branch and tag, not just main**.
3. **Full-history audit on a fresh clone**:
   ```bash
   git clone --mirror git@github.com:agentic-learning-ai-lab/egolearn.git audit && cd audit
   gitleaks git --no-banner --redact .
   # large objects that slipped past the hook:
   git rev-list --objects --all \
     | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
     | awk '$1=="blob" && $3>500000 {print $3, $4}' | sort -rn
   # manual greps: tokens, netids, home paths, reviewer names
   ```
4. **Editorial pass over all docs**: remove the confidential banner, check for
   unpublished project codenames, internal links, broad-permission share links.
5. **License audit**: every third-party-derived file carries its attribution header
   (see docs/design/streaming.md); no CC-BY-NC or GPL-derived code present.
6. **Release prep**: add `CITATION.cff`, swap README to the public version, cut
   `v0.1.0` per the steps above.
7. **Flip**: `gh repo edit agentic-learning-ai-lab/egolearn --visibility public --accept-visibility-change-consequences`.
8. **Post-flip**: enable secret scanning + push protection (free on public repos);
   re-run `scripts/setup_branch_protection.sh` to confirm protection carried over.

## The dual-repo mirror workflow (post-flip)

Public `egolearn` is canonical; private `egolearn-dev` hosts unpublished work.
Same history, two remotes — GitHub cannot make a private fork of a public repo.

```bash
# One-time setup (PI), at/before the flip
gh repo create agentic-learning-ai-lab/egolearn-dev --private
git remote add dev git@github.com:agentic-learning-ai-lab/egolearn-dev.git
git push dev main
bash scripts/setup_branch_protection.sh agentic-learning-ai-lab/egolearn-dev

# Everyone: clone public, add the dev remote
git clone git@github.com:agentic-learning-ai-lab/egolearn.git && cd egolearn
git remote add dev git@github.com:agentic-learning-ai-lab/egolearn-dev.git

# Unpublished project: branch exists ONLY on dev
git switch -c proj/<name>
git push -u dev proj/<name>          # never push proj/* to origin
# collaborate via PRs on egolearn-dev targeting proj/<name>

# Keep fresh with public main (merge, never rebase)
git fetch origin && git merge origin/main

# PI, periodically: keep the mirror's main synced
git fetch origin && git push dev origin/main:main

# At paper release: the branch graduates to the public repo
git push origin proj/<name>
gh pr create --repo agentic-learning-ai-lab/egolearn --base main --head proj/<name>
# merge with a merge commit; delete proj/<name> from both remotes; sync dev main
```

Because nothing is ever rebased or squashed, every commit hash exists identically in
both repos, and moving a branch between them is a plain push.
