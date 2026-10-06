"""Torch-free, checksummed dataset and checkpoint collections on the Hub.

Checksum helpers adapted from agentic-learning-ai-lab/jepa-agent, hub.py
at a7124e81efc81eae88efdb889da0a52c33f562a8 (MIT; see LICENSE).
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)
CHECKSUMS = "checksums.sha256"
CARD = "README.md"
FETCH_RECORD = ".egolearn-fetch.json"
_CHUNK = 1 << 20
ORG = "agentic-learning-ai-lab"


def resolve(path: str | Path, *, repo_type: str = "dataset") -> Path:
    """Resolve a Hub reference or pass a local path through.

    Explicit hf://models/ and hf://datasets/ prefixes override repo_type.
    Dataset snapshots are materialized so manifest containment checks work.
    Model file references download only that file, not every checkpoint.
    """
    return resolve_with_metadata(path, repo_type=repo_type)[0]


def resolve_with_metadata(
    path: str | Path, *, repo_type: str = "dataset"
) -> tuple[Path, dict[str, str]]:
    """Resolve once and retain the original reference and immutable Hub commit."""
    if not str(path).startswith("hf://"):
        return Path(path), {}
    from huggingface_hub import hf_hub_download, snapshot_download

    match = re.fullmatch(
        r"hf://(?:(models|datasets)/)?([^/@]+/[^/@]+)@([^/@]+)(?:/(.+))?", str(path)
    )
    if match is None:
        raise ValueError("expected hf://[models/|datasets/]org/repo@revision[/file]")
    kind, repo, revision, relative = match.groups()
    repo_type = {"models": "model", "datasets": "dataset"}.get(kind, repo_type)
    if relative:
        _check_rel(relative)
    if relative and repo_type == "model":
        result = Path(
            hf_hub_download(repo_id=repo, filename=relative, repo_type=repo_type, revision=revision)
        )
        root = result
        for _ in relative.split("/"):
            root = root.parent
    else:
        root = Path(snapshot_download(repo_id=repo, repo_type=repo_type, revision=revision))
        if repo_type == "dataset":
            root = _materialize(root)
        result = root / relative if relative else root
    log.info("Resolved %s to Hub commit %s", path, root.name)
    if not result.exists():
        raise FileNotFoundError(f"{relative} is absent from {repo}@{root.name}")
    return result, {
        "reference": str(path),
        "repo": repo,
        "repo_type": repo_type,
        "revision": revision,
        "commit": root.name,
    }


def _materialize(snapshot: Path) -> Path:
    """Keep a real-file copy beside a symlink-backed Hub cache snapshot.

    The commit is immutable. Copy once, under a lock, then reuse it offline.
    Validators keep their normal symlink-escape protection for all local data.
    """
    files = sorted(p for p in snapshot.rglob("*") if p.is_file())
    if not any(p.is_symlink() for p in files):
        return snapshot
    import fcntl

    # Offline caches can be partial. A repaired/expanded snapshot gets a new copy
    # instead of permanently reusing the earlier partial materialization.
    inventory = [(p.relative_to(snapshot).as_posix(), p.stat().st_size) for p in files]
    generation = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    parent = snapshot.parent.parent / "egolearn-materialized" / generation
    target = parent / snapshot.name
    if target.is_dir():
        return target  # immutable, atomically installed copies can be shared read-only
    try:
        parent.mkdir(parents=True, exist_ok=True)
        with (parent / f".{snapshot.name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(prefix=".materialize-", dir=parent) as temporary:
                    stage = Path(temporary) / "data"
                    shutil.copytree(snapshot, stage)
                    stage.rename(target)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise PermissionError(
                "Hub dataset materialization needs a writable cache; use a writable HF_HOME "
                "or egolearn.hub fetch --dest <writable-directory>"
            ) from error
        raise
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _check_rel(rel: str) -> None:
    """A checksum entry names a POSIX path inside the collection: relative, no parent or
    empty segment, no backslash, no drive letter. The writer and the reader both apply
    this, so a list the writer produces is one the reader accepts."""
    parts = rel.split("/")
    rooted = rel.startswith("/") or re.match(r"[A-Za-z]:", rel) is not None
    if "\\" in rel or rooted or ".." in parts or "" in parts or "." in parts:
        raise ValueError(f"checksum path escapes the collection: {rel!r}")
    if any(ch in rel for ch in "\n\r\0"):  # one line per file, so a line break cannot be in a name
        raise ValueError(f"checksum path has a control character: {rel!r}")


def _check_repo_name(name: str, what: str) -> None:
    """A name the Hub accepts for a repo (what its client's own validator
    enforces): identifier characters, not starting or ending with ``.`` or
    ``-``, no ``..`` or ``--``, at most 96 characters."""
    if (
        not 1 <= len(name) <= 96
        or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
        or name[0] in ".-"
        or name[-1] in ".-"
        or ".." in name
        or "--" in name
        or name.endswith(".git")
    ):
        raise ValueError(f"{what} must be a Hub repo name (letters, digits, . _ -), got {name!r}")


def _check_name(name: str) -> None:
    """A collection name is one path segment and a Hub repo name: it names
    ``dest/<name>`` and the repo ``<org>/egolearn-<name>``, so it stays
    within the repo-name charset — no separators, whitespace or Markdown
    punctuation can be in it."""
    _check_repo_name(name, "the collection name")


def _listed_files(root: Path) -> list[Path]:
    """Every regular file under ``root``, hidden entries, the root checksum file and the
    root card excluded (the card is edited on the Hub). A file that resolves outside
    ``root`` through a symlink, a symlinked directory (which neither this walk nor the
    upload descends into), or a name the checksum format cannot carry is an error rather
    than a silent omission."""
    files = []
    inside = root.resolve()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_symlink() and p.is_dir():
            raise ValueError(f"symlinked directory in the collection: {rel.as_posix()!r}")
        hidden = any(part.startswith(".") for part in rel.parts)
        if p.is_file() and not hidden and rel not in (Path(CHECKSUMS), Path(CARD)):
            _check_rel(rel.as_posix())
            try:
                resolved = p.resolve()
            except (OSError, RuntimeError) as err:  # a symlink loop, an unreadable parent
                raise ValueError(f"unresolvable file: {rel.as_posix()!r} ({err})") from err
            if not resolved.is_relative_to(inside):
                raise ValueError(f"symlink leaves the collection: {rel.as_posix()!r}")
            files.append(rel)
    return files


def write_checksums(root: Path) -> Path:
    """Write ``checksums.sha256`` for every file under ``root``; returns its path."""
    root = Path(root)
    lines = [f"{sha256_file(root / rel)}  {rel.as_posix()}\n" for rel in _listed_files(root)]
    if not lines:
        raise ValueError(f"nothing to checksum under {root}")
    out = root / CHECKSUMS
    out.write_text("".join(lines))
    return out


def read_checksums(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` from ``root/checksums.sha256``."""
    text = (Path(root) / CHECKSUMS).read_text()
    entries = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or not rel:
            raise ValueError(f"malformed checksum line: {line!r}")
        _check_rel(rel)
        if rel in entries:  # sha256sum -c would check both lines; one digest per file
            raise ValueError(f"duplicate checksum entry: {rel!r}")
        entries[rel] = digest
    return entries


def verify_checksums(root: Path) -> list[str]:
    """Problems with the collection under ``root`` against its ``checksums.sha256``:
    an empty list, missing, altered or unlisted files, as messages. Empty means every
    listed file matches and nothing else is there. The dataset card, the fetch record
    and hidden entries (cache directories) are not listed and not errors."""
    root = Path(root)
    entries = read_checksums(root)
    if not entries:
        return [f"{CHECKSUMS} lists no files"]
    problems = []
    inside = root.resolve()
    for rel, digest in entries.items():
        path = root / rel
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as err:  # a symlink loop, an unreadable parent
            problems.append(f"unresolvable: {rel} ({getattr(err, 'strerror', None) or err})")
            continue
        if not resolved.is_relative_to(inside):  # a symlink out, or a dangling one
            raise ValueError(f"checksum path escapes the collection: {rel!r}")
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"altered: {rel}")
    for p in sorted(root.rglob("*")):
        rel_path = p.relative_to(root)
        if p.is_symlink() and p.is_dir():  # a door out of the collection no list covers
            problems.append(f"symlinked directory: {rel_path.as_posix()}")
            continue
        hidden = any(part.startswith(".") for part in rel_path.parts)
        skip = hidden or rel_path in (Path(CHECKSUMS), Path(CARD), Path(FETCH_RECORD))
        if p.is_file() and not skip and rel_path.as_posix() not in entries:
            problems.append(f"unlisted: {rel_path.as_posix()}")
    return problems


def fetch(
    name: str,
    dest: str | Path | None = None,
    *,
    revision: str = "main",
    repo: str | None = None,
    repo_type: str = "dataset",
    allow_unverified: bool = False,
) -> Path:
    """Stage and verify a pinned download before replacing dest/name.

    Each caller owns a unique staging directory. A per-collection lock serializes
    replacement; a failed rename restores the previous complete collection.
    """
    import fcntl

    from huggingface_hub import HfApi, snapshot_download

    _check_name(name)
    root = Path(dest or os.environ.get("EGOLEARN_DATA", "data"))
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    rid = repo or f"{ORG}/egolearn-{name}"
    info = HfApi().repo_info(repo_id=rid, repo_type=repo_type, revision=revision)
    if not info.sha:
        raise RuntimeError(f"no commit returned for {rid}@{revision}")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        stage = Path(temporary) / "download"
        snapshot_download(repo_id=rid, repo_type=repo_type, revision=info.sha, local_dir=stage)
        verified = (stage / CHECKSUMS).is_file()
        if verified:
            problems = verify_checksums(stage)
            if problems:
                raise ValueError("checksum verification failed: " + "; ".join(problems))
        elif not allow_unverified:
            raise ValueError(f"no {CHECKSUMS}; pass --allow-unverified to accept this collection")
        record = {
            "repo": rid,
            "repo_type": repo_type,
            "revision": revision,
            "commit": info.sha,
            "verified": verified,
            "fetched": datetime.now(UTC).isoformat(),
        }
        (stage / FETCH_RECORD).write_text(json.dumps(record, indent=2) + "\n")
        with (root / f".{name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError(f"destination must be a directory, not a symlink: {target}")
            backup = Path(temporary) / "previous"
            if target.exists():
                target.rename(backup)
            try:
                stage.rename(target)
            except BaseException:
                if backup.exists():
                    backup.rename(target)
                raise
    log.info("Fetched %s@%s to %s", rid, info.sha, target)
    return target


def push(
    src: str | Path,
    *,
    name: str,
    license_id: str,
    repo: str | None = None,
    repo_type: str = "dataset",
    public: bool = False,
) -> str:
    """Upload a private-by-default collection from a verified staging copy.

    Only non-hidden payload files, checksums and a minimal card are uploaded.
    The caller must hold distribution rights; IRB data never belongs on the Hub.
    --public authorizes a public destination but does not change existing visibility.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    _check_name(name)
    # Supported SPDX identifiers whose lowercase spelling is a Hub license ID.
    # Reject other identifiers before creating a repository or uploading data.
    licenses = {
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0",
        "cc-by-nd-4.0",
        "cc-by-nc-nd-4.0",
        "mpl-2.0",
    }
    if license_id.lower() not in licenses:
        raise ValueError("unsupported license; choose " + ", ".join(sorted(licenses)))
    src = Path(src)
    if not src.is_dir():
        raise NotADirectoryError(src)
    rid = repo or f"{ORG}/egolearn-{name}"
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="egolearn-push-") as temporary:
        stage = Path(temporary)
        for relative in _listed_files(src):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / relative, target)
        write_checksums(stage)
        # Do not copy arbitrary run README/configuration metadata into a card.
        (stage / CARD).write_text(
            f"---\nlicense: {license_id.lower()}\ntags:\n- egolearn\n---\n\n"
            f"# {name}\n\nEgoLearn {repo_type} collection.\n\n"
            f"`{CHECKSUMS}` records SHA-256 digests for every payload file.\n"
        )
        try:
            existing = api.repo_info(repo_id=rid, repo_type=repo_type)
        except RepositoryNotFoundError:
            existing = None
        if existing is None:
            api.create_repo(repo_id=rid, repo_type=repo_type, private=not public, exist_ok=True)
        # Recheck after creation too: exist_ok must not hide a racing public repo.
        current = api.repo_info(repo_id=rid, repo_type=repo_type)
        if not public and current.private is not True:
            raise ValueError(f"{rid} is public; pass --public to upload to it")
        keep = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
        remote = api.list_repo_files(repo_id=rid, repo_type=repo_type)
        # One commit replaces the payload, removing stale files that would make
        # the next fetch fail verification. Preserve Hub's .gitattributes only.
        stale = sorted(set(remote) - keep - {".gitattributes"})
        operations = [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(stage / rel))
            for rel in sorted(keep)
        ]
        result = api.create_commit(
            repo_id=rid,
            repo_type=repo_type,
            operations=[*(CommitOperationDelete(path_in_repo=p) for p in stale), *operations],
            commit_message=f"upload {name} with checksums",
            parent_commit=current.sha,
        )
    return str(result.oid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("name")
    fetch_parser.add_argument("--dest", type=Path)
    fetch_parser.add_argument("--revision", default="main")
    fetch_parser.add_argument("--allow-unverified", action="store_true")
    push_parser = commands.add_parser("push")
    push_parser.add_argument("src", type=Path)
    push_parser.add_argument("--name", required=True)
    push_parser.add_argument("--license", dest="license_id", required=True)
    push_parser.add_argument("--public", action="store_true")
    for subparser in (fetch_parser, push_parser):
        subparser.add_argument("--repo")
        subparser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    for operation in ("verify", "checksum"):
        commands.add_parser(operation).add_argument("src", type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if command == "fetch":
            print(fetch(**args))
        elif command == "push":
            print(push(**args))
        elif command == "checksum":
            print(write_checksums(args["src"]))
        else:
            problems = verify_checksums(args["src"])
            print("\n".join(problems) if problems else "ok")
            return int(bool(problems))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
