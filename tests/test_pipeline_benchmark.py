"""The throughput report counts frame requests rather than timer scopes."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_full_pipeline_report_counts_frames_and_accepts_overrides(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/bench/bench_full_pipeline.py"),
            "--config",
            str(root / "src/egolearn/configs/toy_streaming.yaml"),
            "--steps",
            "4",
            "--override",
            f"run_dir={tmp_path / 'run'}",
            "--output-json",
            str(report),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    data = json.loads(report.read_text())
    assert data["steps"] == 4
    assert data["frame_requests"] == 16
    assert data["stages"]["data.load"]["count"] == 4
    assert data["nonfinite_steps"] == 0 and data["steps_per_s"] > 0
    assert "frame requests" in result.stdout
