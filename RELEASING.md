# Releasing

This guide covers the first public release and later releases from the public
repository. Release owners review the selected history, configure the destination,
and approve publication before changing repository visibility.

## Repositories and versions

The public destination is `egolearn/egolearn`. The existing
`agentic-learning-ai-lab/egolearn` repository stays private for development.
Publish only the audited `main` branch and individually approved release tags.
Do not change the existing repository's visibility or use `git push --mirror`.
GitHub retains pull-request refs, including refs for closed and unmerged work.

After publication, public `main` is canonical. The private repository follows
that history, while unpublished work stays on private branches. Preserve shared
commit hashes by merging branches without rebasing, squashing, or force-pushing.

Versions follow SemVer. During 0.x development, minor releases add features and
patch releases fix bugs. Hatch reads the version from `src/egolearn/__init__.py`.
`CHANGELOG.md` groups changes under `[Unreleased]` until a release is prepared.

## Release preparation

1. Confirm the required reproduction results and protocols in the method guides.
   The first release requires BabyCL's SAYCam results on Labeled-S 4AFC and linear
   decoding. Memory Storyboard reproduction is optional for v0.1.
2. Review the documentation, citation metadata, dataset-access instructions, and
   third-party attribution. Check links and image credits as well as prose.
3. Open a release PR that sets the version to `0.1.0`, moves the changelog entries
   under a dated `[0.1.0]` heading, and leaves an empty `[Unreleased]` section.
   Set `version` and `date-released` in `CITATION.cff` to the approved release.
4. For the first release, update the README clone URL and badge, package URLs,
   citation repository, MkDocs repository links, `scripts/docs/hooks.py`, and
   `scripts/docs/reference.py` to `egolearn/egolearn`. Update the URL assertions
   in `tests/test_docs_hooks.py` and run the docs tests and strict build.
   Set the documentation URL to the deployed public site.
   Confirm the Hugging Face namespace and update artifact references before the
   release if they move. Existing private artifact URLs do not establish public access.
5. Require the five CI checks, the strict docs build, and code-owner approval.
   Announce a merge freeze before the final audit, then record the exact candidate
   commit. Any later commit requires another audit of the candidate's history.

The release owners must also approve the publication status of method results,
project notes, and images throughout the selected history. Removing text from the
current tree does not remove it from earlier commits.

## History audit

Audit a fresh bare clone containing the refs you intend to publish. For the first
release, start with `main` and no tags:

```bash
git clone --bare --single-branch --branch main --no-tags \
  git@github.com:agentic-learning-ai-lab/egolearn.git egolearn-audit.git
cd egolearn-audit.git
git rev-parse main
git for-each-ref --format='%(refname) %(objectname)'
```

Compare `main` with the approved release commit before proceeding. Fetch each
approved existing tag individually, then include it in the audit. Never fetch
all branches or tags into this publication clone.

Run Gitleaks and inventory every reachable blob, including deleted files:

```bash
gitleaks git --no-banner --redact --log-opts="--all -m" .
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1 == "blob" && $3 > 500000 { print $3, $2, $4 }'
```

Investigate each blob over 500,000 bytes before publication. Inspect the reachable
history for personal paths, credentials, restricted data, internal links, broad
share links, unpublished material, and missing license notices. A clean secret
scan does not settle the editorial or licensing review.

The early `PILRadiusBlur` history requires a specific attribution review. Commits
before its lab-authored rewrite describe a verbatim MoCo-style port. Record the
release owners' decision about that historical implementation before publishing
its ancestors. The rewrite alone does not settle the historical review.

Retain the candidate SHA, complete ref list, tool versions, scan outputs, and
owner sign-offs in the private release record. Keep sensitive findings outside
Git. Recheck the refs immediately before pushing from this clone.

## Destination setup

Create a new, empty `egolearn/egolearn` repository. It may stay private while the
release owners configure it, but it must receive only the audited release refs.
Do not import the private development repository's pull requests or other branches.

Configure the destination before accepting contributions:

- Enable merge commits, disable squash/rebase merges, and delete merged branches.
- Give the code owners write access and configure the required CI/reviewer secrets.
- Apply `scripts/setup_branch_protection.sh egolearn/egolearn` after `main` exists.
- Verify the advisory workflow and its reviewer checkout use the same immutable
  revision. Fork PRs must remain excluded from the secret-bearing review job.
- Enable secret scanning and push protection when available, disable the wiki,
  and decide whether to enable Discussions.

For site hosting, create a separate Cloudflare Pages project from the new
repository. Validate its `pages.dev` deployment before attaching `egolearn.org`
and `www.egolearn.org`, with `www` redirecting to the apex. Use
`bash scripts/docs/build.sh` and publish `dist`. Set
`EGOLEARN_SITE_STAGING=false` only for the public production environment.
Keep the private staging project behind Access, or remove it after migration.

## Publication

Proceed after the release owners approve the candidate, audits, and destination.
In the audited clone, verify `main` still names the approved commit and create
its annotated tag:

```bash
git rev-parse main
git tag -a v0.1.0 main -m "EgoLearn v0.1.0"
git rev-parse 'v0.1.0^{commit}'
```

Audit the tag message and include the tag's object ID in the approved ref list.
From this clone, push only the explicit refs:

```bash
git remote add public git@github.com:egolearn/egolearn.git
git push public refs/heads/main:refs/heads/main
git push public refs/tags/v0.1.0:refs/tags/v0.1.0
```

Compare the destination's branch and tag SHAs with the approved ref list. If the
new destination is private, change only `egolearn/egolearn` to public after this
verification. Leave `agentic-learning-ai-lab/egolearn` private.

Create the GitHub release from the approved tag, using its changelog section as
the release notes. Confirm the CI checks, source archive, citation metadata, docs,
and advertised artifact links are accessible without lab credentials. v0.1 is a
GitHub release; it does not require a PyPI upload.

## Development after publication

Clone public `main` and add the private repository as a second remote if you have
access:

```bash
git clone git@github.com:egolearn/egolearn.git
cd egolearn
git remote add dev git@github.com:agentic-learning-ai-lab/egolearn.git
```

Start unpublished work from public `main` and push only its new branch to `dev`:

```bash
git fetch origin
git switch -c proj/example origin/main
git push -u dev proj/example
```

Keep long-lived branches current by merging `origin/main`. Before transferring a
project branch to the public repository, audit everything newly reachable from
that branch. Transfer only the approved branch, open its public PR, and merge it
with a merge commit after review.

## Private main synchronization

Private `main` follows public `main` through fast-forward pushes by a designated
synchronization actor. A merge-commit PR in the private repository would create
private-only history, so do not merge synchronization PRs.

Before the first synchronization, configure the private repository's PR rule to
allow that actor to bypass the PR requirement. For rulesets, put the PR rule in
its own ruleset with the designated actor's bypass. Keep deletion and
non-fast-forward prohibitions in a separate ruleset without a bypass. For classic
branch protection, use
[specified actors allowed to bypass required pull requests](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule).
Keep required checks enabled and do not add this synchronization exception to
public `main`. The protection setup script does not configure this exception.

After public merges and successful checks, run from a checkout with `origin`
pointing to the public repository and `dev` pointing to the private repository:

```bash
git fetch origin
git fetch dev
git merge-base --is-ancestor dev/main origin/main &&
  git push dev refs/remotes/origin/main:refs/heads/main
```

If the private repository lacks check results for that commit, push it to a
temporary private branch and open a validation PR to run CI. After checks pass,
the designated actor performs the fast-forward push and closes that PR without
merging it. If the ancestry check fails, stop and have the release owners resolve
the divergence. Never force-push either `main`; keep unpublished commits on
private project branches so private `main` can follow public history.

## Later releases

Prepare the version, changelog, and citation metadata in a reviewed PR. Audit the
candidate and its ancestors, create the approved tag, and publish its GitHub
release. Reuse the existing public repository and site; no visibility change or
repository migration is needed.
