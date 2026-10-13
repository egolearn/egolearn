"""Finite, complete-seed summaries of suite manifests (standard library only)."""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

from egolearn.hub import sha256_file
from egolearn.run_record import finite_metrics, read_json


def summarize(path: Path, allow_unverified: bool = False) -> dict:
    path = path.expanduser().resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported suite manifest version")
    runs = manifest["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest must contain expected runs")
    seeds = [run["seed"] for run in runs]
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("manifest contains invalid or repeated seeds")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("manifest must declare input pins (an empty mapping is allowed)")
    input_sha256 = {}
    for key, item in inputs.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValueError(f"manifest input {key!r} must declare a lowercase SHA-256")
        # Content identifies an input; its local storage path may differ between runs.
        input_sha256[key] = item["sha256"]
    directions = manifest["metrics"]
    if not directions or any(value not in {"min", "max"} for value in directions.values()):
        raise ValueError("manifest must declare metric directions")
    values: dict[str, list[float]] = {key: [] for key in directions}
    problems = []
    unverified = []
    digest = sha256_file(path)
    for run in runs:
        seed = run["seed"]
        try:
            directory = (path.parent / run["directory"]).resolve()
            if directory.parent != path.parent:
                raise ValueError("run directory escapes manifest directory")
            record = read_json(directory / "record.json")
            if type(record.get("seed")) is not int:
                raise ValueError("record seed must be an integer")
            if record.get("status") != "completed":
                raise ValueError(
                    f"run status is {record.get('status', 'missing')}; incomplete result"
                )
            for key, expected in (
                ("seed", seed),
                ("entry", manifest["entry"]),
                ("status", "completed"),
                ("returncode", 0),
                ("manifest_sha256", digest),
                ("suite_sha256", manifest["suite_sha256"]),
                ("config_sha256", run["config_sha256"]),
            ):
                if record.get(key) != expected:
                    raise ValueError(f"record {key} does not match expected {expected!r}")
            provenance = record.get("provenance", {})
            if provenance != manifest.get("provenance"):
                raise ValueError("record provenance does not match the manifest")
            verified = (
                provenance.get("dirty") is False
                and re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("commit", ""))) is not None
            )
            if not verified:
                if not allow_unverified:
                    raise ValueError(
                        "dirty/unknown code; --allow-unverified is for exploratory tables"
                    )
                unverified.append(seed)
            metrics = finite_metrics(record["metrics"])
            for key in values:
                if key in metrics:
                    values[key].append(metrics[key])
                else:
                    problems.append({"seed": seed, "reason": f"missing metric {key}"})
        except OSError:
            problems.append({"seed": seed, "reason": "missing or unreadable run record"})
        except (ValueError, KeyError, TypeError) as error:
            problems.append({"seed": seed, "reason": str(error)})
    rows = []
    for metric, samples in values.items():
        complete = len(samples) == len(runs)
        rows.append(
            {
                "metric": metric,
                "direction": directions[metric],
                "n": len(samples),
                "expected": len(runs),
                "mean": statistics.mean(samples) if complete else None,
                "sd": statistics.stdev(samples) if complete and len(samples) > 1 else None,
            }
        )
    return {
        "name": manifest["name"],
        "entry": manifest["entry"],
        "seeds": sorted(seeds),
        "input_sha256": input_sha256,
        "rows": rows,
        "problems": problems,
        "unverified_seeds": unverified,
        "provenance": manifest.get("provenance", {}),
    }


def _display(value) -> str:
    return "NA" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def print_table(path: Path, format: str, baseline: Path | None, allow_unverified: bool) -> int:
    result = summarize(path, allow_unverified)
    if baseline is not None:
        reference = summarize(baseline, allow_unverified)
        if (result["name"], result["entry"]) != (reference["name"], reference["entry"]):
            raise ValueError("baseline must describe the same suite name and entry")
        if result["seeds"] != reference["seeds"]:
            raise ValueError("baseline seed sets must match")
        if result["input_sha256"] != reference["input_sha256"]:
            raise ValueError("baseline input pins must match (config keys and SHA-256 checksums)")
        reference_rows = {row["metric"]: row for row in reference["rows"]}
        if set(reference_rows) != {row["metric"] for row in result["rows"]}:
            raise ValueError("baseline metric names must match")
        result["baseline_problems"] = reference["problems"]
        result["baseline_unverified_seeds"] = reference["unverified_seeds"]
        result["baseline_provenance"] = reference["provenance"]
        result["provenance_changed"] = [
            key
            for key in ("commit", "uv_lock_sha256", "image", "package_version")
            if result["provenance"].get(key) != reference["provenance"].get(key)
        ]
        for row in result["rows"]:
            previous = reference_rows[row["metric"]]
            if row["direction"] != previous["direction"]:
                raise ValueError("baseline metric directions must match")
            row["baseline_mean"] = previous["mean"]
            row["delta"] = (
                row["mean"] - previous["mean"]
                if row["mean"] is not None and previous["mean"] is not None
                else None
            )
            row["regression"] = (
                row["delta"] * (1 if row["direction"] == "min" else -1) > 0
                if row["delta"] is not None
                else None
            )
    if format == "json":
        print(json.dumps(result, indent=2, allow_nan=False))
    elif format == "csv":
        columns = [*result["rows"][0], "issues", "unverified_seeds"]
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(
                {
                    **row,
                    "issues": json.dumps(result["problems"] + result.get("baseline_problems", [])),
                    "unverified_seeds": json.dumps(
                        result["unverified_seeds"] + result.get("baseline_unverified_seeds", [])
                    ),
                }
            )
    else:
        columns = list(result["rows"][0])
        print("| " + " | ".join(columns) + " |")
        print("| " + " | ".join("---" for _ in columns) + " |")
        for row in result["rows"]:
            print(
                "| "
                + " | ".join(_display(row[column]).replace("|", "\\|") for column in columns)
                + " |"
            )
        for label, problems in (
            ("run", result["problems"]),
            ("baseline", result.get("baseline_problems", [])),
        ):
            for problem in problems:
                print(f"{label} seed {problem['seed']}: {problem['reason']}")
        if result["unverified_seeds"] or result.get("baseline_unverified_seeds"):
            print("Exploratory table: includes dirty or unavailable code provenance.")
        if result.get("provenance_changed"):
            print("Changed from baseline: " + ", ".join(result["provenance_changed"]))
    return int(
        bool(
            result["problems"]
            or result.get("baseline_problems")
            or any(row.get("regression") for row in result["rows"])
        )
    )
