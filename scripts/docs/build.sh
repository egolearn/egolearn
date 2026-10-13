#!/usr/bin/env bash
# Builds dist/ for Cloudflare Pages: the MkDocs site under docs/ and the
# landing page at the root. Staging (the default) keeps crawlers out.
set -euo pipefail

cd "$(dirname "$0")/../.."

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

# Read the flag the way scripts/docs/hooks.py does: 0/false/no/off or empty
# turn staging off, in any case; unset means staging.
staging="$(printf '%s' "${EGOLEARN_SITE_STAGING-true}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
case "$staging" in
  0 | false | no | off | "") exit 0 ;;
esac

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
