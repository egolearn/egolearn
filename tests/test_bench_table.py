"""A table never silently averages incomplete, contaminated or non-finite runs."""

import json
import math

import pytest

from egolearn.commands.bench_table import print_table, summarize
from egolearn.hub import sha256_file
from egolearn.run_record import atomic_json


def manifest_fixture(root, values=(2.0, 4.0), dirty=False):
    root.mkdir()
    manifest = root / "manifest.json"
    provenance = {"commit": "a" * 40, "dirty": dirty}
    runs = [
        {"seed": seed, "directory": f"seed-{seed}", "config_sha256": "b" * 64}
        for seed in range(len(values))
    ]
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "name": "test",
            "entry": "eval",
            "suite_sha256": "c" * 64,
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
