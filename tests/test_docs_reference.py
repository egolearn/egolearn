"""Static reference extraction and the docs-only site build."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("docs_reference", ROOT / "scripts/docs/reference.py")
assert spec is not None and spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def write(root, path, content):
    target = root / "src/egolearn" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dedent(content), encoding="utf-8")
    return target


@pytest.fixture
def source_tree(tmp_path):
    write(
        tmp_path,
        "config/registry.py",
        """\
        raise RuntimeError("source must never execute")
        class Registry:
            pass
        ENCODERS = Registry("encoder")
        GENERATIVE = Registry("generative")
        """,
    )
    write(
        tmp_path,
        "base.py",
        """\
        class Base:
            def __init__(self, rate=1e-4, *, path="a|b", flag=False):
                pass
        class Left(Base):
            pass
        class Empty(object):
            pass
        class Right(Base):
            def __init__(self, size: int = 2 * 16, /, *args, label='test', **kwargs):
                pass
        """,
    )
    write(tmp_path, "methods/__init__.py", "from ..base import Left as Parent\n")
    write(
        tmp_path,
        "methods/demo.py",
        '''\
        import egolearn.config.registry as registry
        from egolearn.config.registry import ENCODERS as E
        from . import Parent
        from ..base import Base, Empty, Right

        @E.register("base")
        class Child(Base):
            """An inherited encoder.

            Not part of the summary.
            """

        @registry.ENCODERS.register("diamond")
        class Diamond(Parent, Right):
            pass

        @E.register("empty")
        class EmptyChild(Empty):
            pass

        @E.register("factory")
        def factory(width, /, scale: float = 1e-3, *, options=("x", 'y')):
            """Build an encoder."""
            raise RuntimeError("do not call")
        ''',
    )
    return tmp_path


def test_catalog_preserves_source_defaults_and_inherited_mro(source_tree):
    index = reference.SourceIndex(source_tree)
    catalog = index.components()
    assert catalog["generative"] == []
    child, diamond, empty, factory = catalog["encoder"]
    assert child.target == "encoder:base"
    assert child.signature == '(rate = 1e-4, *, path = "a|b", flag = False)'
    assert child.constructor == "egolearn.base.Base.__init__"
    assert diamond.signature == "(size: int = 2 * 16, /, *args, label = 'test', **kwargs)"
    assert diamond.constructor == "egolearn.base.Right.__init__"
    assert empty.signature == "()"
    assert empty.constructor is None
    assert factory.signature == "(width, /, scale: float = 1e-3, *, options = (\"x\", 'y'))"
    page = reference.component_page(index, "encoder", catalog["encoder"])
    assert "Not part of the summary" not in page
    assert "No description." in page
    assert "egolearn.methods.demo" in page
    assert "Inherited constructor: `egolearn.base.Right.__init__`" in page

    # A new index reads edits rather than reusing imported objects or an old catalog.
    write(source_tree, "methods/new.py", 'from .demo import E\n@E.register("new")\ndef f(): pass\n')
    assert len(reference.SourceIndex(source_tree).components()["encoder"]) == 5


def test_config_headers_are_recursive_and_escape_markdown(tmp_path):
    write(tmp_path, "configs/group/with_header.yaml", "# A <tag> | `code`.\n# Second line.\nx: 1\n")
    write(tmp_path, "configs/plain.yml", "x: 1\n# Not a header.\n")
    page = reference.config_page(tmp_path)
    assert "group/with_header.yaml" in page
    assert "&lt;tag&gt; &#124; `code`. Second line." in page
    assert "No header comment." in page
    assert "Not a header" not in page


def test_benchmark_metadata_is_required_and_literal(source_tree):
    path = write(source_tree, "evaluation/benchmarks/demo/__init__.py", '"""Unstructured prose."""')
    with pytest.raises(ValueError, match="BENCHMARK_METADATA"):
        reference.benchmark_page(reference.SourceIndex(source_tree))
    path.write_text("BENCHMARK_METADATA = dict(name='dynamic')\n")
    with pytest.raises(ValueError, match="malformed node"):
        reference.benchmark_page(reference.SourceIndex(source_tree))
    path.write_text(
        "BENCHMARK_METADATA = {'name': 'Demo', 'source': 'https://example.org', "
        "'license': 'CC0', 'access': 'unknown', 'prepare': 'No preparer.'}\n"
    )
    with pytest.raises(ValueError, match="access tier"):
        reference.benchmark_page(reference.SourceIndex(source_tree))


def test_repository_reference_covers_registries_packages_and_api():
    pages, _ = reference.reference_pages(ROOT)
    index = reference.SourceIndex(ROOT)
    assert len([p for p in pages if p.startswith("reference/components/")]) == len(
        index.registries()
    )
    assert "VoEProbe.__init__" in pages["reference/components/probe.md"].content
    assert "No components registered." in pages["reference/components/generative.md"].content
    benchmarks = pages["reference/benchmarks.md"].content
    for path in (ROOT / "src/egolearn/evaluation/benchmarks").glob("*/__init__.py"):
        assert path.parent.name + "/__init__.py" in benchmarks
    assert "::: egolearn.pipeline.build" in pages["reference/api/pipeline.md"].content
    assert "egl eval [CONFIG] [HYDRA_ARGS ...]" in pages["reference/cli.md"].content


def test_strict_site_build_never_imports_documented_code(tmp_path):
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocstrings")
    script = """\
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'egolearn', 'torch', 'torchvision'}:
            raise AssertionError('docs imported ' + fullname)

sys.meta_path.insert(0, BlockTrainingImports())
from mkdocs.config import load_config
from mkdocs.commands.build import build
build(load_config('mkdocs.yml', site_dir=sys.argv[1], strict=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "site")],
        cwd=ROOT,
        env={**os.environ, "EGOLEARN_SITE_STAGING": "true"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    site = tmp_path / "site/reference"
    api = (site / "api/pipeline/index.html").read_text()
    assert 'id="egolearn.pipeline.build"' in api
    assert 'id="egolearn.pipeline.validate"' in api
    assert "edit/main/docs/reference/" not in api
    assert 'id="encoder-video_vit"' in (site / "components/encoder/index.html").read_text()
    assert "IntPhys 2" in (site / "benchmarks/index.html").read_text()
    assert not (ROOT / "docs/reference").exists()
