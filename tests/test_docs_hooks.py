"""Link rewriting in the docs-site hook (scripts/docs/hooks.py)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_hooks", ROOT / "scripts/docs/hooks.py")
assert spec is not None and spec.loader is not None
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)

SITE = {"docs/cli.md": ("cli.md", "cli/")}


def rewrite(markdown: str) -> str:
    return hooks.rewrite_links(markdown, "README.md", "overview.md", "overview/", SITE.get)


def test_links_to_site_pages_and_repo_files_are_rewritten():
    out = rewrite("[CLI](docs/cli.md) and [code](src/egolearn/cli.py#L1)")
    assert "[CLI](cli.md)" in out
    assert (
        "(https://github.com/agentic-learning-ai-lab/egolearn/blob/main/src/egolearn/cli.py#L1)"
        in out
    )


def test_code_is_left_alone():
    text = "`handlers[name](batch)`\n\n```python\nhandlers[name](batch)\n```\n"
    assert rewrite(text) == text


def test_missing_targets_stay_for_mkdocs_to_report():
    assert rewrite("[broken](does-not-exist.md)") == "[broken](does-not-exist.md)"
