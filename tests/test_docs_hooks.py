"""Link rewriting in the docs-site hook (scripts/docs/hooks.py)."""

import importlib.util
import logging
from pathlib import Path

import pytest

# The docs group provides Markdown; CI's test job installs it.
markdown = pytest.importorskip("markdown")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_hooks", ROOT / "scripts/docs/hooks.py")
assert spec is not None and spec.loader is not None
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)

GITHUB = "https://github.com/agentic-learning-ai-lab/egolearn/blob/main/"
SITE = {
    "docs/cli.md": ("cli.md", "cli/"),
    "docs/assets/icon-light.svg": ("assets/icon-light.svg", "assets/icon-light.svg"),
}
PAGE = hooks.Page("README.md", "overview.md", "overview/", SITE.get)


def render(text: str) -> str:
    extensions = [hooks.link_extension(lambda: PAGE), "pymdownx.superfences"]
    return markdown.markdown(text, extensions=extensions)


def test_links_to_site_pages_and_repo_files_are_rewritten():
    out = render("[CLI](docs/cli.md) and [config](pyproject.toml#L1)")
    assert 'href="cli.md"' in out
    assert f'href="{GITHUB}pyproject.toml#L1"' in out


def test_code_is_left_alone():
    # pyproject.toml exists, so a rewrite would show up as a GitHub URL.
    text = (
        "`f[x](pyproject.toml)`\n\n"
        "```\nf[x](pyproject.toml)\n```\n\n"
        "> ~~~\n> f[x](pyproject.toml)\n> ~~~\n\n"
        "    f[x](pyproject.toml)\n"
    )
    out = render(text)
    assert out.count("f[x](pyproject.toml)") == 4
    assert GITHUB not in out


def test_missing_markdown_targets_stay_for_mkdocs_to_report():
    assert 'href="does-not-exist.md"' in render("[broken](does-not-exist.md)")


def test_raw_html_targets_are_rewritten_and_missing_ones_reported(caplog):
    html = "<img src=\"docs/assets/icon-light.svg\">\n\n<img src='docs/assets/nope.svg'>\n"
    with caplog.at_level(logging.WARNING, logger="mkdocs.plugins.egolearn"):
        out = render(html)
    assert 'src="../assets/icon-light.svg"' in out
    assert "docs/assets/nope.svg" in caplog.text
