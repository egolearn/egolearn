"""Link rewriting in the docs-site hook (scripts/docs/hooks.py)."""

import importlib.util
import logging
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pytest

# The docs group provides Markdown; CI's test job installs it.
markdown = pytest.importorskip("markdown")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_hooks", ROOT / "scripts/docs/hooks.py")
assert spec is not None and spec.loader is not None
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)

GITHUB = "https://github.com/egolearn/egolearn/blob/main/"
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


@pytest.mark.parametrize("target", ["missing.md", "#missing-anchor"])
def test_strict_site_build_rejects_broken_internal_links(tmp_path, target, caplog):
    pytest.importorskip("mkdocs")
    from mkdocs.commands.build import build
    from mkdocs.config import load_config
    from mkdocs.exceptions import Abort

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text(f"# Home\n\n[Broken]({target})\n")
    # Retain the repository's validation settings while rendering one bad page.
    config = load_config(
        str(ROOT / "mkdocs.yml"),
        strict=True,
        docs_dir=str(docs),
        site_dir=str(tmp_path / "site"),
        nav=[{"Home": "index.md"}],
        hooks=[],
    )
    with caplog.at_level(logging.WARNING, logger="mkdocs"), pytest.raises(Abort):
        build(config)
    assert target in caplog.text


@pytest.mark.parametrize("staging", [True, False], ids=["staging", "production"])
def test_site_crawlers_can_read_staging_noindex_tags(tmp_path, monkeypatch, staging):
    pytest.importorskip("mkdocs")
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    monkeypatch.setenv("EGOLEARN_SITE_STAGING", str(staging).lower())
    config = load_config(str(ROOT / "mkdocs.yml"), strict=True, site_dir=str(tmp_path / "site"))
    build(config)

    site = Path(config.site_dir)
    for name in ("index.html", "overview/index.html", "404.html"):
        assert ('content="noindex, nofollow"' in (site / name).read_text()) is staging
    assert (site / "robots.txt").exists() is staging
    if staging:
        robots = RobotFileParser()
        robots.parse((site / "robots.txt").read_text().splitlines())
        assert robots.can_fetch("Googlebot", "https://preview.example/")
        assert robots.can_fetch("Googlebot", "https://preview.example/overview/")
