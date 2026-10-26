"""Commit immutable run records and an atomic table to Hugging Face."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path

from egolearn.hub import write_checksums
from egolearn.publication.schema import (
    build_table,
    encoded,
    merge_runs,
    portable,
    read_json,
    repo_id,
    validate_run,
)
from egolearn.utils.model_artifact import file_digest, validate_export


def _download(api, repository: str, kind: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repository, filename, repo_type=kind, revision=revision, token=api.token)
    )


def _repository(api, repository: str, kind: str, *, public: bool):
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.repo_info(repository, repo_type=kind)
    except RepositoryNotFoundError:
        api.create_repo(repository, repo_type=kind, private=not public, exist_ok=True)
        info = api.repo_info(repository, repo_type=kind)
    if info.private != (not public):
        raise ValueError(f"visibility differs for {repository}; existing visibility is preserved")
    return info


def _model(api, run: dict, directory: Path) -> str:
    from huggingface_hub import CommitOperationAdd

    repository = run["checkpoint"]["repo_id"]
    info = _repository(api, repository, "model", public=False)
    names = set(api.list_repo_files(repository, repo_type="model", revision=info.sha))
    allowed = {".gitattributes", "model.json", "model.safetensors", "checksums.sha256", "README.md"}
    if names - allowed:
        raise ValueError(f"{repository} contains unmanaged files; choose a new model repository")
    if "model.json" in names or "model.safetensors" in names:
        if not {"model.json", "model.safetensors"} <= names:
            raise ValueError(f"incomplete remote model: {repository}")
        metadata = _download(api, repository, "model", info.sha, "model.json")
        if file_digest(metadata) != run["checkpoint"]["metadata_sha256"]:
            raise ValueError(f"{repository} already holds a different export; choose a new name")
        # Confirm the actual remote weights, not only the claim in model.json.
        detailed = api.repo_info(
            repository, repo_type="model", revision=info.sha, files_metadata=True
        )
        weights = next(f for f in detailed.siblings if f.rfilename == "model.safetensors")
        if not weights.lfs or weights.lfs.sha256 != run["checkpoint"]["weights_sha256"]:
            raise ValueError(f"remote weight digest differs: {repository}")
        return info.sha
    write_checksums(directory)
    condition = run["condition"]
    # No redistribution license is inferred from the software license.
    card = (
        "---\ntags:\n- egolearn\n- safetensors\n---\n\n"
        f"# {condition['label']} — seed {run['training_seed']}\n\n"
        f"{condition['description']}\n\n"
        "## Training\n\n"
        f"Dataset: {condition['dataset']}. Architecture: {condition['architecture']}. "
        f"Training: {condition['regime']}. Checkpoint step: {run['checkpoint']['step']}.\n\n"
        "## Use and limitations\n\n"
        "Load the exported pair with `egolearn.evaluation.load_model`. "
        "Text evaluations also require the original authorized vocabulary and input data. "
        "These private research weights have no redistribution grant. "
        "Training snapshots, optimizer state, transcripts and frames are excluded.\n\n"
        "## Evaluation\n\n"
        f"Publication run ID: `{run['id']}`. "
        "The EgoLearn results repository records per-seed scores and evaluation protocols. "
        "Unavailable training or evaluation code revisions remain unspecified.\n"
    )
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(directory / name))
        for name in ("model.json", "model.safetensors", "checksums.sha256")
    ]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    return api.create_commit(
        repository,
        repo_type="model",
        operations=operations,
        commit_message="publish evaluated egolearn model",
        parent_commit=info.sha,
    ).oid


def publish_results(api, runs: list[dict], *, results_repo: str, public: bool = False) -> str:
    """Merge immutable records, retrying concurrent index changes without losing runs."""
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import HfHubHTTPError

    repo_id(results_repo)
    for run in runs:
        validate_run(run, published=True)
    for attempt in range(3):
        info = _repository(api, results_repo, "dataset", public=public)
        names = api.list_repo_files(results_repo, repo_type="dataset", revision=info.sha)
        existing = []
        if "index.json" in names:
            index = read_json(_download(api, results_repo, "dataset", info.sha, "index.json"))
            if index.get("schema_version") != 1:
                raise ValueError("unsupported remote results index")
            existing = index["runs"]
        merged = merge_runs(existing, runs)
        now = datetime.now(UTC).isoformat()
        table = build_table(merged, results_repo=results_repo, updated_at=now)
        if merged == existing:
            return info.sha
        old_ids = {r["id"] for r in existing}
        files = {f"runs/{r['id']}.json": encoded(r) for r in merged if r["id"] not in old_ids}
        files["index.json"] = encoded({"schema_version": 1, "runs": merged})
        files["benchmarks.json"] = encoded(table)
        if "README.md" not in names:
            files["README.md"] = (
                b"---\ntags:\n- egolearn\n---\n\n# EgoLearn results\n\n"
                b"Per-run evaluation records and a generated benchmark table. "
                b"Each record links scores to an immutable model revision "
                b"and checkpoint digest.\n\n"
                b"## Interpretation\n\n"
                b"Values are fractions unless a column specifies another unit. "
                b"Means and sample standard deviations require every declared training seed "
                b"and identical evaluation protocols. "
                b"Single-seed results have no standard deviation. "
                b"Shape preference is not classification accuracy.\n\n"
                b"## Provenance limits\n\n"
                b"Historical evaluations can lack code revisions. Recorded manifest hashes "
                b"do not verify underlying image content. Architectures and training budgets "
                b"can differ between conditions. See each run's protocol and provenance fields.\n\n"
                b"## Files\n\n"
                b"`runs/` contains immutable records. `index.json` contains all records. "
                b"`benchmarks.json` is the website feed. Each update commits all three together. "
                b"This repository contains no checkpoints, frames, transcripts or local paths.\n"
            )
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=content)
            for name, content in sorted(files.items())
        ]
        try:
            result = api.create_commit(
                results_repo,
                repo_type="dataset",
                operations=operations,
                commit_message="publish egolearn benchmark results",
                parent_commit=info.sha,
            )
        except HfHubHTTPError as error:
            if error.response.status_code not in (409, 412) or attempt == 2:
                raise
            continue
        remote = _download(api, results_repo, "dataset", result.oid, "benchmarks.json")
        if remote.read_bytes() != files["benchmarks.json"]:
            raise ValueError("published feed did not match the prepared table")
        return result.oid
    raise RuntimeError("results publication exhausted retries")


def publish(
    bundle: Path, *, results_repo: str, collection: str | None = None, public_results: bool = False
) -> dict:
    """Upload a prepared bundle; leave existing models and run records immutable."""
    from huggingface_hub import HfApi

    document = read_json(bundle / "bundle.json")
    if document.get("schema_version") != 1 or not document.get("runs"):
        raise ValueError("expected a prepared publication bundle")
    runs = copy.deepcopy(document["runs"])
    # Validate the entire local batch before creating any remote repositories.
    for run in runs:
        validate_run(run)
        directory = bundle / "models" / str(run["training_seed"])
        metadata = validate_export(directory, publishing=True)
        portable(metadata)
        if (
            file_digest(directory / "model.json") != run["checkpoint"]["metadata_sha256"]
            or metadata["weights_sha256"] != run["checkpoint"]["weights_sha256"]
            or metadata["provenance"]["sha256"] != run["checkpoint"]["sha256"]
        ):
            raise ValueError("prepared model no longer matches its publication record")
    preflight = copy.deepcopy(runs)
    for run in preflight:
        run["checkpoint"]["revision"] = "0" * 40
    build_table(preflight, results_repo=results_repo, updated_at="preflight")
    api = HfApi()
    if collection:
        api.get_collection(collection)
    for run in runs:
        directory = bundle / "models" / str(run["training_seed"])
        run["checkpoint"]["revision"] = _model(api, run, directory)
    revision = publish_results(api, runs, results_repo=results_repo, public=public_results)
    receipt = {
        "results_repo": results_repo,
        "revision": revision,
        "models": [run["checkpoint"] for run in runs],
        "collection": collection,
    }
    (bundle / "publication.json").write_bytes(encoded(receipt))
    if collection:
        for run in runs:
            api.add_collection_item(
                collection, run["checkpoint"]["repo_id"], "model", exists_ok=True
            )
        api.add_collection_item(collection, results_repo, "dataset", exists_ok=True)
    return receipt
