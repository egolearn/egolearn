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

# The Pages build image ships Python but not uv, and a pip-installed uv is
# not on its PATH: run that one through python3 -m.
uv=(uv)
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install uv
  uv=(python3 -m uv)
fi

# Only the docs group: neither the package nor torch.
"${uv[@]}" sync --locked --only-group docs

rm -rf dist
.venv/bin/mkdocs build --strict -d dist/docs
cp -R landing/. dist/
# Without a root 404 page, Cloudflare treats unknown routes as a single-page app.
cp dist/docs/404.html dist/404.html

if [[ "$EGOLEARN_SITE_STAGING" == false ]]; then
  exit 0
fi

# Staging: block crawlers at the root (the docs pages also get a noindex
# meta tag from overrides/main.html and /docs/robots.txt from the hook).
printf 'User-agent: *\nDisallow: /\n' > dist/robots.txt
python3 - <<'EOF'
from pathlib import Path

page = Path("dist/index.html")
noindex = '<meta name="robots" content="noindex, nofollow">'
html = page.read_text()
if noindex not in html:
    page.write_text(html.replace("<head>", f"<head>\n    {noindex}", 1))
EOF
