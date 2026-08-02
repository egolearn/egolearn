from importlib.metadata import version

import egolearn


def test_installed_metadata_matches_package_version() -> None:
    # Catches a stale editable install or a mismatched pyproject/__init__ bump.
    assert version("egolearn") == egolearn.__version__
