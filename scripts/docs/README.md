# Site development

Use this directory's build script to assemble the landing page and documentation
for local preview. You need Python 3.12 with `venv` support to install the build tools.

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
places MkDocs output under `dist/docs/`, so serve the complete `dist/` directory.

For live documentation editing without assembling the landing page:

```bash
uv run --group docs mkdocs serve
```

## Content and styling

Edit documentation in `docs/`, root guidance in its original Markdown files, and
method pages in `projects/*/README.md`. The hook imports those source pages before
validation, rewrites their links and images, and retains their source edit URLs.

`mkdocs.yml` defines navigation and loads the lab's shared `lab-tokens.css` and
`lab-mkdocs.css`. Material templates live in `overrides/`, while the landing page and its
assets live in `landing/`.
