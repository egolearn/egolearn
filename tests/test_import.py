import re
import subprocess
import sys
import textwrap

import egolearn


def test_package_imports() -> None:
    assert hasattr(egolearn, "__version__")


def test_evaluation_protocol_imports_without_train_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockTrainDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"PIL", "torch", "torchvision"}:
                    raise ModuleNotFoundError(
                        f"optional train dependency imported: {fullname}",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockTrainDependencies())

        import egolearn.evaluation
        from egolearn.evaluation import Probe
        from egolearn.evaluation.probe import Probe as ProbeFromModule

        assert Probe is ProbeFromModule
        """
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True)


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.(dev|post)\d+)?([ab]|rc\d+)?", egolearn.__version__)
