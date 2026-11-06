#!/usr/bin/env bash
# Builds dist/ for Cloudflare Pages: the MkDocs site under docs/ and the
# landing page at the root. Staging (the default) keeps crawlers out.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Normalize before MkDocs reads the flag through YAML: an empty value would
# otherwise become null. Unset means staging; 0/false/no/off or empty disable it.
staging="$(printf '%s' "${EGOLEARN_SITE_STAGING-true}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
case "$staging" in
  0 | false | no | off | "") export EGOLEARN_SITE_STAGING=false ;;
  *) export EGOLEARN_SITE_STAGING=true ;;
esac
# Preview deployments stay unindexed even if they inherit production's flag.
if [[ "${CF_PAGES:-}" == 1 && "${CF_PAGES_BRANCH:-}" != main ]]; then
  export EGOLEARN_SITE_STAGING=true
fi

# Use the same uv release locally, in CI, and on Pages. Bootstrap in a temporary
# environment so an existing uv installation is never replaced.
docs_uv_version="$(cat scripts/docs/uv-version)"
docs_uv=(uv)
if [[ "$(uv --version 2>/dev/null || true)" != "uv $docs_uv_version" ]]; then
  docs_bootstrap="$(mktemp -d)"
  trap 'rm -rf "$docs_bootstrap"' EXIT
  python3 -m venv "$docs_bootstrap"
  "$docs_bootstrap/bin/python" -m pip install --disable-pip-version-check "uv==$docs_uv_version"
  docs_uv=("$docs_bootstrap/bin/python" -m uv)
fi

# Keep the docs-only sync separate from the contributor's training environment.
export UV_PROJECT_ENVIRONMENT=build/docs-venv
"${docs_uv[@]}" sync --locked --only-group docs --python 3.12

rm -rf dist
build/docs-venv/bin/mkdocs build --strict -d dist/docs
cp -R landing/. dist/
# Without a root 404 page, Cloudflare treats unknown routes as a single-page app.
cp dist/docs/404.html dist/404.html

if [[ "$EGOLEARN_SITE_STAGING" == false ]]; then
  exit 0
fi

# Staging: block crawlers at the root (the docs pages also get a noindex
# meta tag from overrides/main.html and /docs/robots.txt from the hook).
printf 'User-agent: *\nDisallow: /\n' > dist/robots.txt
build/docs-venv/bin/python - <<'EOF'
from pathlib import Path

page = Path("dist/index.html")
noindex = '<meta name="robots" content="noindex, nofollow">'
html = page.read_text()
if noindex not in html:
    page.write_text(html.replace("<head>", f"<head>\n    {noindex}", 1))
EOF
