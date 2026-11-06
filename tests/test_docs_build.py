"""Deployment assembly and crawler controls in the site build wrapper."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.robotparser import RobotFileParser

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def site(tmp_path):
    scripts = tmp_path / "scripts/docs"
    scripts.mkdir(parents=True)
    for name in ("build.sh", "uv-version"):
        shutil.copyfile(ROOT / "scripts/docs" / name, scripts / name)

    landing = tmp_path / "landing"
    landing.mkdir()
    (landing / "index.html").write_text("<html><head></head><body>Landing</body></html>")
    (landing / "asset.svg").write_text("<svg/>")
    # Keep installer/rendering work out of this shell-wrapper test. Real MkDocs
    # rendering and strict link validation run in the docs CI job.
    binaries = tmp_path / "build/docs-venv/bin"
    binaries.mkdir(parents=True)
    (binaries / "python").symlink_to(sys.executable)
    renderer = binaries / "mkdocs"
    renderer.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1:] == ['build', '--strict', '-d', 'dist/docs']\n"
        "out = Path('dist/docs')\n"
        "out.mkdir(parents=True)\n"
        "(out / 'index.html').write_text(os.environ['EGOLEARN_SITE_STAGING'])\n"
        "(out / '404.html').write_text('Not found')\n"
    )
    renderer.chmod(0o755)
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('uv ' + Path('scripts/docs/uv-version').read_text().strip())\n"
        "else:\n"
        "    assert sys.argv[1:] == ['sync', '--locked', '--only-group', 'docs', "
        "'--python', '3.12']\n"
        "    assert os.environ['UV_PROJECT_ENVIRONMENT'] == 'build/docs-venv'\n"
    )
    uv.chmod(0o755)
    # Existing output must not carry staging restrictions into production.
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/robots.txt").write_text("User-agent: *\nDisallow: /\n")
    (tmp_path / "dist/old.html").write_text("stale page")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/sentinel").write_text("training environment")
    return tmp_path


@pytest.mark.parametrize(
    ("settings", "staging"),
    [
        ({}, True),
        ({"EGOLEARN_SITE_STAGING": " FALSE ", "CF_PAGES": "1", "CF_PAGES_BRANCH": "main"}, False),
        (
            {"EGOLEARN_SITE_STAGING": "false", "CF_PAGES": "1", "CF_PAGES_BRANCH": "feature"},
            True,
        ),
    ],
    ids=["default-staging", "public-production", "preview-inherits-production"],
)
def test_site_assembly_and_staging(site, settings, staging):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"EGOLEARN_SITE_STAGING", "CF_PAGES", "CF_PAGES_BRANCH"}
    }
    env.update(settings)
    env["PATH"] = str(site) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["bash", "scripts/docs/build.sh"], cwd=site, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = site / "dist"
    assert ('content="noindex, nofollow"' in (output / "index.html").read_text()) is staging
    assert (output / "docs/index.html").read_text() == str(staging).lower()
    assert (output / "robots.txt").exists() is staging
    if staging:
        robots = RobotFileParser()
        robots.parse((output / "robots.txt").read_text().splitlines())
        assert robots.can_fetch("Googlebot", "https://preview.example/")
        assert robots.can_fetch("Googlebot", "https://preview.example/docs/overview/")
    assert (output / "404.html").read_text() == (output / "docs/404.html").read_text()
    assert (output / "asset.svg").read_text() == "<svg/>"
    assert not (output / "old.html").exists()
    assert (site / ".venv/sentinel").read_text() == "training environment"
