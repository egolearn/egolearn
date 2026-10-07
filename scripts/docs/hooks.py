"""MkDocs hooks for the egolearn site.

Adds repository files outside docs/ as pages, rewrites relative links so each
one points at a site page or at the file on GitHub, and blocks crawlers while
the site is in staging mode.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/agentic-learning-ai-lab/egolearn"

# Repository file -> page path on the site.
EXTRA_PAGES = {
    "README.md": "overview.md",
    "CHANGELOG.md": "changelog.md",
    "CONTRIBUTING.md": "contributing.md",
    "SECURITY.md": "security.md",
    "projects/babycl/README.md": "projects/babycl.md",
    "projects/storyboard/README.md": "projects/storyboard.md",
}
REPO_OF_PAGE = {site: repo for repo, site in EXTRA_PAGES.items()}

MD_LINK = re.compile(r"(\]\()([^)\s]+)(\))")
HTML_SRC = re.compile(r"""(\b(?:src|srcset|href)=")([^"]+)(")""")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def on_config(config):
    staging = config.extra.get("staging", True)
    config.extra["staging"] = str(staging).strip().lower() not in ("0", "false", "no", "off", "")
    return config


def on_files(files, config):
    for repo_path, site_path in EXTRA_PAGES.items():
        content = (ROOT / repo_path).read_text(encoding="utf-8")
        files.append(File.generated(config, site_path, content=content))
    if config.extra["staging"]:
        files.append(File.generated(config, "robots.txt", content="User-agent: *\nDisallow: /\n"))
    return files


def _site_path(repo_path: str, files) -> str | None:
    if repo_path in EXTRA_PAGES:
        return EXTRA_PAGES[repo_path]
    if repo_path.startswith("docs/") and files.get_file_from_path(repo_path[5:]):
        return repo_path[5:]
    return None


def on_page_markdown(markdown, page, config, files):
    page_site = page.file.src_uri
    page_repo = REPO_OF_PAGE.get(page_site, "docs/" + page_site)
    page_dir = posixpath.dirname(page_site)
    if page_site in REPO_OF_PAGE:
        # Generated pages edit their source file, not a docs/ copy.
        page.edit_url = f"{GITHUB}/edit/main/{page_repo}"

    def rewrite(match: re.Match, html: bool = False) -> str:
        target = match.group(2)
        if SCHEME.match(target) or target.startswith(("#", "/")):
            return match.group(0)
        path, sep, fragment = target.partition("#")
        repo_target = posixpath.normpath(posixpath.join(posixpath.dirname(page_repo), path))
        site_target = _site_path(repo_target, files)
        if site_target is not None and html:
            # Raw HTML is not resolved by MkDocs: link from the page's output URL.
            target_url = files.get_file_from_path(site_target).url
            new = posixpath.relpath(target_url, page.url or ".")
        elif site_target is not None:
            new = posixpath.relpath(site_target, page_dir or ".")
        else:
            kind = "tree" if (ROOT / repo_target).is_dir() else "blob"
            new = f"{GITHUB}/{kind}/main/{repo_target}"
        return f"{match.group(1)}{new}{sep}{fragment}{match.group(3)}"

    markdown = MD_LINK.sub(rewrite, markdown)
    return HTML_SRC.sub(lambda match: rewrite(match, html=True), markdown)
