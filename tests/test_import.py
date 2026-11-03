import os
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def run_isolated(tmp_path):
    def run(script, *args, blocked=("torch", "torchvision", "hydra", "numpy", "PIL")):
        guard = textwrap.dedent(
            f"""
            import sys
            from importlib.abc import MetaPathFinder

            class BlockOptionalImports(MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {blocked!r}:
                        raise AssertionError(f'optional dependency imported: {{fullname}}')

            sys.meta_path.insert(0, BlockOptionalImports())
            """
        )
        return subprocess.run(
            [sys.executable, "-I", "-c", guard + textwrap.dedent(script), *args],
            cwd=tmp_path,
            env={k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return run


def test_evaluation_protocol_imports_without_train_dependencies(run_isolated):
    run_isolated(
        """
        from egolearn.evaluation import Benchmark, Probe
        from egolearn.evaluation.probe import Probe as ProbeFromModule

        assert Probe is ProbeFromModule
        assert issubclass(Probe, Benchmark)
        """,
        blocked=("PIL", "torch", "torchvision"),
    )


@pytest.mark.parametrize(
    "command,expected",
    [
        (["--help"], ["train", "eval"]),
        (["bench", "--help"], ["table", "run"]),
        (["export", "--help"], []),
        (["data", "list", "--json"], []),
        (["doctor", "--skip-cuda"], []),
        (["submit", "--help"], []),
    ],
)
def test_operational_commands_do_not_import_training_dependencies(run_isolated, command, expected):
    result = run_isolated(
        """
        from egolearn.cli import main
        raise SystemExit(main(sys.argv[1:]))
        """,
        *command,
    )
    assert all(word in result.stdout for word in expected)


def test_verify_and_table_do_not_import_training_dependencies(run_isolated, tmp_path):
    from egolearn.hub import write_checksums

    data = tmp_path / "data"
    data.mkdir()
    (data / "payload").write_text("synthetic")
    write_checksums(data)
    run_isolated(
        """
        from egolearn.cli import main
        assert main(['data', 'verify', sys.argv[1]]) == 0
        try:
            main(['bench', 'table', sys.argv[2]])
        except SystemExit as error:
            assert error.code == 1
        else:
            raise AssertionError('missing suite must fail')
        """,
        str(data),
        str(tmp_path / "missing.json"),
    )
