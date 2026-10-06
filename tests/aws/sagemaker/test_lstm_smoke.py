"""
CI smoke test for the LSTM training script.

Deliberately runs train.py as a real subprocess (not importing its internals) -- the goal is
catching exactly the class of bugs this project hit repeatedly during manual runs: missing
required hyperparameters, metric-regex mismatches, shape errors -- cheaply, before any billed
SageMaker job ever starts.

Not a model-quality test: tiny synthetic data, 1 epoch, no accuracy assertions.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TRAIN_PY = Path(__file__).resolve().parents[3] / "aws" / "sagemaker" / "lstm_src" / "train.py"

N_SERIES = 5
SERIES_LENGTH = 400  # hours; comfortably above context+prediction (336) used below
CONTEXT_LENGTH = 48
PREDICTION_LENGTH = 48


def make_fixture(data_dir: Path):
    """Writes tiny synthetic train.json/test.json in the same DeepAR JSON Lines format
    the real pipeline uses: {"start": ..., "target": [...], "cat": [i]}.
    """
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    with open(train_dir / "train.json", "w") as f_train, open(test_dir / "test.json", "w") as f_test:
        for i in range(N_SERIES):
            full_target = [float((t % 24) + i) for t in range(SERIES_LENGTH)]  # simple synthetic signal
            record_full = {"start": "2023-01-01 00:00:00", "target": full_target, "cat": [i]}
            record_train = dict(record_full)
            record_train["target"] = full_target[:-PREDICTION_LENGTH]

            f_train.write(json.dumps(record_train) + "\n")
            f_test.write(json.dumps(record_full) + "\n")

    return train_dir, test_dir


def test_lstm_train_script_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_dir, test_dir = make_fixture(tmp_path)
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cmd = [
            sys.executable,
            str(TRAIN_PY),
            "--hidden-size", "8",
            "--num-layers", "1",
            "--embedding-dim", "4",
            "--dropout", "0.0",
            "--learning-rate", "1e-3",
            "--epochs", "1",
            "--batch-size", "4",
            "--steps-per-epoch", "2",
            "--context-length", str(CONTEXT_LENGTH),
            "--prediction-length", str(PREDICTION_LENGTH),
            "--train", str(train_dir),
            "--test", str(test_dir),
            "--model-dir", str(model_dir),
            "--s3-bucket", "unused-in-ci",
            "--s3-results-prefix", "unused-in-ci",
            "--skip-s3-upload",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        assert result.returncode == 0, (
            f"train.py exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
        assert "validation:rmse=" in result.stdout, "metric line missing - would break HPO's objective parsing"
        assert (model_dir / "model.pt").exists(), "model artifact was not saved"
        assert (model_dir / "result.json").exists(), "--skip-s3-upload should write results locally"
