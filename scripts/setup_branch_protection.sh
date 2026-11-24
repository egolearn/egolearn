#!/usr/bin/env bash
# Apply branch protection to main. Requires GitHub Team (or public repo) and admin rights.
#
# Usage:
#   bash scripts/setup_branch_protection.sh owner/repo [approvals] [enforce_admins]
#   bash scripts/setup_branch_protection.sh egolearn/egolearn             # defaults: 1 approval, admins enforced
#   bash scripts/setup_branch_protection.sh agentic-learning-ai-lab/egolearn 1 false
#                                                                        # setup phase: PI can bypass
#
# Review gate: approvals must come from a CODE OWNER (.github/CODEOWNERS), so only
# code-owner approvals count. enforce_admins=false lets repo admins (org owners)
# bypass the gate — use only during repo setup, then restore true on that repository.
# Never apply it to the private development repository: rulesets there let only
# the sync deploy key update main (.github/workflows/sync-private-main.yml).
#
# Notes:
# - required_linear_history stays FALSE: this repo merges with merge commits only
#   (squash/rebase are disabled at the repo level) so hashes remain shared with the
#   private development repository. Linear history would forbid that.
# - The checks list must match the job names in .github/workflows/ci.yml.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 owner/repo [approvals] [enforce_admins]" >&2
  exit 2
fi

REPO="$1"
APPROVALS="${2:-1}"
ENFORCE_ADMINS="${3:-true}"

gh api --method PUT -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/main/protection" --input - <<EOF
{
  "required_status_checks": {
    "strict": false,
    "checks": [
      {"context": "lint"},
      {"context": "types"},
      {"context": "test"},
      {"context": "lock"},
      {"context": "gitleaks"}
    ]
  },
  "enforce_admins": ${ENFORCE_ADMINS},
  "required_pull_request_reviews": {
    "required_approving_review_count": ${APPROVALS},
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
EOF

echo "Branch protection applied to ${REPO}:main (approvals: ${APPROVALS}, code-owner review required, enforce_admins: ${ENFORCE_ADMINS})."
