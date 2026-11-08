import re
from importlib.metadata import version

import egolearn


def test_installed_metadata_matches_package_version() -> None:
    # Catches a stale editable install or a mismatched pyproject/__init__ bump.
    assert version("egolearn") == egolearn.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.(dev|post)\d+)?([ab]|rc\d+)?", egolearn.__version__)
