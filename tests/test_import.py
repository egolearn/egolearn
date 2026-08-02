import re

import egolearn


def test_package_imports() -> None:
    assert hasattr(egolearn, "__version__")


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.(dev|post)\d+)?([ab]|rc\d+)?", egolearn.__version__)
