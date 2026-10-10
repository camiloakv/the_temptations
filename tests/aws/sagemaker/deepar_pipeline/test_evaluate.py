"""Tests for the pipeline's evaluation step, run as a real subprocess like the container would."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[4] / "aws" / "sagemaker" / "deepar_pipeline" / "steps" / "evaluate.py"
)

PREDICTION_LENGTH = 6
FORECAST_ERROR = 3.0


def write_jsonlines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in records)


def make_inputs(root, n_predictions=2):
    """Two clients; the forecast is the truth shifted by FORECAST_ERROR, so every RMSE is 3.0."""
    actual_tails = [[10.0] * PREDICTION_LENGTH, ["NaN", 20.0, 20.0, 20.0, 20.0, 20.0]]
    test_records = [
        {"start": "2023-01-01 00:00:00", "target": [1.0] * 4 + tail, "cat": [index]}
        for index, tail in enumerate(actual_tails)
    ]
    predictions = []
    for tail in actual_tails[:n_predictions]:
        median = [(0.0 if value == "NaN" else value) + FORECAST_ERROR for value in tail]
        predictions.append({"mean": median, "quantiles": {"0.1": median, "0.5": median, "0.9": median}})

    write_jsonlines(root / "test" / "test.json", test_records)
    write_jsonlines(root / "predictions" / "train.json.out", predictions)
    (root / "meta").mkdir()
    with open(root / "meta" / "client_ids.json", "w") as handle:
        json.dump(["MT_A", "MT_B"], handle)


def run_script(root):
    cmd = [
        sys.executable, str(SCRIPT),
        "--predictions-dir", str(root / "predictions"),
        "--test-dir", str(root / "test"),
        "--meta-dir", str(root / "meta"),
        "--output-dir", str(root / "evaluation"),
        "--prediction-length", str(PREDICTION_LENGTH),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)


def test_reports_rmse_in_registry_layout(tmp_path):
    make_inputs(tmp_path)
    result = run_script(tmp_path)
    assert result.returncode == 0, result.stderr

    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["regression_metrics"]["rmse"]["value"] == FORECAST_ERROR
    assert report["n_scored_clients"] == len(["MT_A", "MT_B"])
    assert [entry["client_id"] for entry in report["per_client_errors"]] == ["MT_A", "MT_B"]


def test_row_count_mismatch_fails_loudly(tmp_path):
    make_inputs(tmp_path, n_predictions=1)
    result = run_script(tmp_path)
    assert result.returncode != 0
    assert "mismatch" in result.stderr
