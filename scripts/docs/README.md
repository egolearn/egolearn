# Site build

Use this directory's build script to assemble the landing page and documentation
for Cloudflare Pages. You need Python 3.12 with `venv` support to install the build tools.

## Local build

From the repository root, run:

```bash
bash scripts/docs/build.sh
python3 -m http.server --directory dist 8000
```

Open `http://localhost:8000/` for the landing page or `/docs/` for the documentation.
The script uses the uv version in `uv-version`, bootstrapping it into a temporary
environment when necessary. It installs only the locked `docs` dependency group
in `build/docs-venv/`, leaving your existing `.venv/` untouched and excluding EgoLearn
and torch from the build environment.

MkDocs builds with `--strict`, so missing internal pages and anchors fail the build,
while external URLs remain unchecked. The build copies the landing page to `dist/` and
places MkDocs output under `dist/docs/`, so publish the complete `dist/` directory.

For live documentation editing without assembling the landing page:

```bash
uv run --group docs mkdocs serve
```

## Cloudflare Pages

Configure these settings for both production and preview deployments:

| Setting | Value |
| --- | --- |
| Production branch | `main` |
| Build command | `bash scripts/docs/build.sh` |
| Build output directory | `dist` |
| `PYTHON_VERSION` | `3.12` |
| `SKIP_DEPENDENCY_INSTALL` | `1` |

[`SKIP_DEPENDENCY_INSTALL`](https://developers.cloudflare.com/pages/configuration/build-image/#skip-dependency-install)
disables Cloudflare's automatic installation so the script controls dependencies.
The `docs` CI job runs the same build on every pull request with a 20-minute timeout,
without becoming a required branch-protection check.

Staging is enabled by default, adding a `noindex, nofollow` meta tag to the landing
page and documentation. Both `/robots.txt` and `/docs/robots.txt` disallow crawling
until you explicitly disable staging for the production build. For the public release,
set `EGOLEARN_SITE_STAGING=false` in Cloudflare's production environment only, leaving
it unset in the preview environment.
The script also forces staging when Cloudflare's `CF_PAGES_BRANCH` differs from
`main`, even if a preview inherits the production setting.

To inspect production output locally:

```bash
EGOLEARN_SITE_STAGING=false bash scripts/docs/build.sh
```

This rebuild removes the staging tags and generated robots files from `dist/`.

## Cold-build timing

A cold build took **9.18 seconds** on Linux x86_64 with Python 3.12 and local
temporary storage, below the 20-minute build limit. This includes uv bootstrap,
downloads into empty pip and uv caches, dependency installation, strict rendering,
and assembly of `dist/`. The locked versions were MkDocs 1.6.1 and Material 9.7.7.
This is a local measurement; Cloudflare's own runner and network determine its
deployment time.

To repeat the measurement, use a fresh checkout without `build/docs-venv/` and
with uv absent from `PATH`, then run:

```bash
docs_cache=$(mktemp -d)
time env UV_CACHE_DIR="$docs_cache/uv" PIP_CACHE_DIR="$docs_cache/pip" \
  bash scripts/docs/build.sh
```

## Content and styling

Edit documentation in `docs/`, root guidance in its original Markdown files, and
method pages in `projects/*/README.md`. The hook imports those source pages before
validation, rewrites their links and images, and retains their source edit URLs.

`mkdocs.yml` defines navigation and loads the lab's shared `lab-tokens.css` and
`lab-mkdocs.css`. Material templates live in `overrides/`, while the landing page and its
assets live in `landing/`.
