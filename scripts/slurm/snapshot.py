"""Archive a clean commit and create an independent working directory for a job."""

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


@contextmanager
def _job_directory(root: Path) -> Iterator[Path]:
    job = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    try:
        yield job
    except BaseException:
        # Only our own generated directory is removed; work links are not followed.
        for entry in (job, *job.rglob("*")):
            if entry.is_dir() and not entry.is_symlink():
                entry.chmod(0o700)
        shutil.rmtree(job)
        raise


def snapshot(root: Path) -> Path:
    # Reject staged, unstaged and untracked files, before creating output paths.
    # Ignored datasets, credentials and run outputs never enter git archive.
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("dirty checkout; commit the intended code before submission")
    commit = git("rev-parse", "HEAD").decode().strip()
    if commit != os.environ["EGOLEARN_CODE_COMMIT"]:
        raise ValueError("HEAD changed during submission; resubmit")
    staged = git("ls-files", "--stage")
    if staged.startswith(b"160000") or b"\n160000" in staged:
        raise ValueError("submodules are not supported by the code snapshot")
    archive = git("archive", "--format=tar", commit)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # Validate before creating directories: links could escape the archive.
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported archive entry: {member.name}")
        root.mkdir(parents=True, exist_ok=True)
        with _job_directory(root) as job:
            code = job / "code"
            code.mkdir()
            tar.extractall(code, filter="data")
            work = job / "work"
            work.mkdir()
            for entry in code.iterdir():
                (work / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
            for entry in code.rglob("*"):
                entry.chmod(stat.S_IMODE(entry.stat().st_mode) & ~0o222)
            code.chmod(0o555)
            (job / "dirty.diff").write_text("")
            (job / "run.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "commit": commit,
                        "dirty": False,
                        "dirty_diff": "dirty.diff",
                        "submitted": datetime.now(UTC).isoformat(),
                        "code": "code",
                        "work": "work",
                        "image": os.environ["EGOLEARN_IMAGE"],
                        "environment_stamp": os.environ["EGOLEARN_EXPECTED_STAMP"],
                    },
                    indent=2,
                )
                + "\n"
            )
            return job


if __name__ == "__main__":
    try:
        print(snapshot(Path(sys.argv[1])))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"snapshot: {error}")
