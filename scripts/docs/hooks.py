"""MkDocs hooks for the egolearn site.

Adds repository files outside docs/ as pages, rewrites relative links so each
one points at a site page or at the file on GitHub, and blocks crawlers while
the site is in staging mode. Code blocks and inline code are never rewritten,
and links to missing files are left as written so `mkdocs build --strict`
reports them.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable
from pathlib import Path

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
# Fenced blocks (``` or ~~~, closed by the same fence) and inline code spans.
CODE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=indent)(?P=fence)[ \t]*$"
    r"|(?P<ticks>`+)(?!`).+?(?<!`)(?P=ticks)(?!`)",
    re.MULTILINE | re.DOTALL,
)

# repo path -> (site source path, site output URL), or None when not on the site
SiteLookup = Callable[[str], "tuple[str, str] | None"]


def rewrite_links(
    markdown: str, page_repo: str, page_site: str, page_url: str, lookup: SiteLookup
) -> str:
    """Rewrite relative links outside code.

    Targets on the site become relative site links (Markdown links from the
    page's source path, raw HTML from its output URL), other existing files
    become GitHub URLs, and missing targets are left unchanged so MkDocs
    reports them.
    """
    page_dir = posixpath.dirname(page_site)

    def rewrite(match: re.Match, html: bool) -> str:
        target = match.group(2)
        if SCHEME.match(target) or target.startswith(("#", "/")):
            return match.group(0)
        path, sep, fragment = target.partition("#")
        repo_target = posixpath.normpath(posixpath.join(posixpath.dirname(page_repo), path))
        on_site = lookup(repo_target)
        if on_site is not None:
            source, url = on_site
            new = (
                posixpath.relpath(url, page_url or ".")
                if html
                else posixpath.relpath(source, page_dir or ".")
            )
        elif (ROOT / repo_target).exists():
            kind = "tree" if (ROOT / repo_target).is_dir() else "blob"
            new = f"{GITHUB}/{kind}/main/{repo_target}"
        else:
            return match.group(0)
        return f"{match.group(1)}{new}{sep}{fragment}{match.group(3)}"

    def prose(text: str) -> str:
        text = MD_LINK.sub(lambda m: rewrite(m, html=False), text)
        return HTML_SRC.sub(lambda m: rewrite(m, html=True), text)

    out, last = [], 0
    for code in CODE.finditer(markdown):
        out.append(prose(markdown[last : code.start()]))
        out.append(code.group(0))
        last = code.end()
    out.append(prose(markdown[last:]))
    return "".join(out)


def on_config(config):
    staging = config.extra.get("staging", True)
    config.extra["staging"] = str(staging).strip().lower() not in ("0", "false", "no", "off", "")
    return config


def on_files(files, config):
    from mkdocs.structure.files import File

    for repo_path, site_path in EXTRA_PAGES.items():
        content = (ROOT / repo_path).read_text(encoding="utf-8")
        files.append(File.generated(config, site_path, content=content))
    if config.extra["staging"]:
        files.append(File.generated(config, "robots.txt", content="User-agent: *\nDisallow: /\n"))
    return files


def on_page_markdown(markdown, page, config, files):
    page_site = page.file.src_uri
    page_repo = REPO_OF_PAGE.get(page_site, "docs/" + page_site)
    if page_site in REPO_OF_PAGE:
        # Generated pages edit their source file, not a docs/ copy.
        page.edit_url = f"{GITHUB}/edit/main/{page_repo}"

    def lookup(repo_path: str) -> tuple[str, str] | None:
        source = EXTRA_PAGES.get(repo_path)
        if source is None and repo_path.startswith("docs/"):
            source = repo_path[5:]
        found = files.get_file_from_path(source) if source else None
        return (source, found.url) if found else None

    return rewrite_links(markdown, page_repo, page_site, page.url, lookup)
