"""A table never silently averages incomplete, contaminated or non-finite runs."""

import json
import math

import pytest

from egolearn.commands.bench_table import print_table, summarize
from egolearn.hub import sha256_file
from egolearn.run_record import atomic_json


def manifest_fixture(root, values=(2.0, 4.0), dirty=False, *, seeds=None, inputs=None):
    root.mkdir()
    manifest = root / "manifest.json"
    provenance = {"commit": "a" * 40, "dirty": dirty}
    runs = [
        {"seed": seed, "directory": f"seed-{seed}", "config_sha256": "b" * 64}
        for seed in (range(len(values)) if seeds is None else seeds)
    ]
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "name": "test",
            "entry": "eval",
            "suite_sha256": "c" * 64,
            "inputs": {} if inputs is None else inputs,
            "metrics": {"score": "max"},
            "runs": runs,
            "provenance": provenance,
        },
    )
    for run, value in zip(runs, values, strict=True):
        path = root / run["directory"] / "record.json"
        atomic_json(
            path,
            {
                "entry": "eval",
                "seed": run["seed"],
                "status": "completed",
                "returncode": 0,
                "manifest_sha256": sha256_file(manifest),
                "suite_sha256": "c" * 64,
                "config_sha256": "b" * 64,
                "provenance": provenance,
                "metrics": {"score": value},
            },
        )
    return manifest


def test_mean_and_sample_sd_and_singleton(tmp_path):
    result = summarize(manifest_fixture(tmp_path / "two"))
    assert not result["problems"]
    row = result["rows"][0]
    assert row["mean"] == 3.0
    assert row["sd"] == pytest.approx(math.sqrt(2))
    single = summarize(manifest_fixture(tmp_path / "one", (5.0,)))
    assert single["rows"][0]["sd"] is None


@pytest.mark.parametrize(
    "change", ["missing", "failed", "nan", "wrong_seed", "wrong_manifest", "missing_metric"]
)
def test_bad_run_is_listed_and_prevents_partial_average(tmp_path, change):
    manifest = manifest_fixture(tmp_path / "suite")
    path = manifest.parent / "seed-1/record.json"
    record = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        if change == "failed":
            record["status"] = "failed"
        elif change == "nan":
            record["metrics"]["score"] = float("nan")
        elif change == "wrong_seed":
            record["seed"] = 9
        elif change == "wrong_manifest":
            record["manifest_sha256"] = "d" * 64
        else:
            record["metrics"] = {}
        path.write_text(json.dumps(record))
    result = summarize(manifest)
    assert result["rows"][0]["mean"] is None
    assert result["rows"][0]["n"] == 1
    assert result["problems"][0]["seed"] == 1
    assert str(tmp_path) not in json.dumps(result)


def test_dirty_code_requires_explicit_exploratory_flag(tmp_path):
    manifest = manifest_fixture(tmp_path / "suite", dirty=True)
    assert summarize(manifest)["rows"][0]["mean"] is None
    exploratory = summarize(manifest, allow_unverified=True)
    assert exploratory["rows"][0]["mean"] == 3.0
    assert exploratory["unverified_seeds"] == [0, 1]


@pytest.mark.parametrize("format", ["md", "csv", "json"])
def test_baseline_regression_is_flagged_in_each_format(tmp_path, capsys, format):
    baseline = manifest_fixture(tmp_path / "baseline", (4.0, 6.0))
    current = manifest_fixture(tmp_path / "current", (2.0, 4.0))
    assert print_table(current, format, baseline, False) == 1
    output = capsys.readouterr().out
    assert "regression" in output
    if format == "json":
        row = json.loads(output)["rows"][0]
        assert row["delta"] == -2.0 and row["regression"] is True
    assert str(tmp_path) not in output


def test_missing_baseline_run_produces_no_delta(tmp_path, capsys):
    baseline = manifest_fixture(tmp_path / "baseline")
    (baseline.parent / "seed-0/record.json").unlink()
    current = manifest_fixture(tmp_path / "current")
    assert print_table(current, "json", baseline, False) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["rows"][0]["delta"] is None
    assert result["baseline_problems"]


@pytest.mark.parametrize("seeds", [(0, 2), (0,), (0, 1, 2)])
def test_baseline_rejects_different_seed_sets(tmp_path, capsys, seeds):
    baseline = manifest_fixture(tmp_path / "baseline")
    current = manifest_fixture(tmp_path / "current", (3.0,) * len(seeds), seeds=seeds)
    with pytest.raises(ValueError, match="baseline seed sets must match"):
        print_table(current, "json", baseline, False)
    assert not capsys.readouterr().out


# Pin comparison precedes rendering; output formats are covered above.
@pytest.mark.parametrize(
    "change, allow_unverified",
    [("hash", False), ("hash", True), ("key", False), ("added", False), ("removed", False)],
)
def test_baseline_rejects_different_input_pins(tmp_path, capsys, change, allow_unverified):
    inputs = {"benchmarks.0.manifest": {"path": "/private/trials.json", "sha256": "d" * 64}}
    baseline = manifest_fixture(tmp_path / "baseline", inputs=inputs, dirty=allow_unverified)
    if change == "hash":
        inputs["benchmarks.0.manifest"]["sha256"] = "e" * 64
    elif change == "key":
        inputs["benchmarks.1.manifest"] = inputs.pop("benchmarks.0.manifest")
    elif change == "added":
        inputs["checkpoint"] = {"path": "/private/model.pth", "sha256": "f" * 64}
    else:
        inputs.clear()
    current = manifest_fixture(tmp_path / "current", inputs=inputs, dirty=allow_unverified)
    with pytest.raises(ValueError, match="baseline input pins must match"):
        print_table(current, "json", baseline, allow_unverified)
    assert not capsys.readouterr().out


def test_baseline_accepts_reordered_seeds_and_relocated_inputs(tmp_path, capsys):
    pins = {"checkpoint": "d" * 64, "benchmarks.0.manifest": "e" * 64}
    baseline = manifest_fixture(
        tmp_path / "baseline",
        inputs={
            key: {"path": str(tmp_path / key), "sha256": digest} for key, digest in pins.items()
        },
    )
    current = manifest_fixture(
        tmp_path / "current",
        seeds=(1, 0),
        inputs={
            key: {"path": str(tmp_path / "relocated" / key), "sha256": digest}
            for key, digest in reversed(pins.items())
        },
    )
    assert print_table(current, "json", baseline, False) == 0
    output = capsys.readouterr().out
    assert str(tmp_path) not in output
    result = json.loads(output)
    assert result["seeds"] == [0, 1]
    assert result["input_sha256"] == pins
    assert result["rows"][0]["delta"] == 0.0
    assert result["rows"][0]["regression"] is False


@pytest.mark.parametrize(
    "inputs", [None, [], {"checkpoint": {}}, {"checkpoint": {"sha256": "bad"}}]
)
def test_table_rejects_missing_or_invalid_input_pins(tmp_path, inputs):
    manifest = manifest_fixture(tmp_path / "suite")
    payload = json.loads(manifest.read_text())
    if inputs is None:
        del payload["inputs"]
    else:
        payload["inputs"] = inputs
    atomic_json(manifest, payload)
    with pytest.raises(ValueError, match="input"):
        summarize(manifest)
