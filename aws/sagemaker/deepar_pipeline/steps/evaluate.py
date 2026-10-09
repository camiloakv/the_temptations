"""
Pipeline step 4: evaluate DeepAR predictions against the held-out week.

Reads the batch-transform output (one JSON line per client, in the same order as the train file
that was sent in), compares the median forecast with the true last `prediction_length` hours from
the test file, and writes evaluation.json.

The report follows SageMaker's `regression_metrics` layout so that (a) the pipeline's condition
step can read it with JsonGet and (b) the Model Registry UI can render it as model statistics.

Same metric as the Stage 2 notebook: mean over clients of per-client RMSE on the median (0.5)
quantile, in original kWh units -- directly comparable with the LSTM and TFT numbers.

Kept compatible with Python 3.8 (what the sklearn processing image ships).
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROCESSING_INPUT = "/opt/ml/processing/input"
PROCESSING_OUTPUT = "/opt/ml/processing/evaluation"
MEDIAN_QUANTILE = "0.5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default=f"{PROCESSING_INPUT}/predictions")
    parser.add_argument("--test-dir", default=f"{PROCESSING_INPUT}/test")
    parser.add_argument("--meta-dir", default=f"{PROCESSING_INPUT}/meta")
    parser.add_argument("--output-dir", default=PROCESSING_OUTPUT)
    parser.add_argument("--prediction-length", type=int, default=168)
    return parser.parse_args()


def read_jsonlines(paths):
    records = []
    for path in paths:
        with open(path) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def per_client_rmse(test_records, predictions, client_ids, prediction_length):
    errors = []
    for position, record in enumerate(test_records):
        tail = record["target"][-prediction_length:]
        actual = np.array([np.nan if value == "NaN" else value for value in tail], dtype=float)
        forecast = np.array(predictions[position]["quantiles"][MEDIAN_QUANTILE], dtype=float)

        scored = ~np.isnan(actual)
        if scored.any():
            rmse = float(np.sqrt(np.mean((actual[scored] - forecast[scored]) ** 2)))
            errors.append({"client_id": client_ids[position], "rmse": rmse})
    return errors


def main():
    args = parse_args()

    prediction_files = sorted(Path(args.predictions_dir).glob("*.out"))
    if not prediction_files:
        raise FileNotFoundError(f"No batch-transform output (*.out) found in {args.predictions_dir}")

    predictions = read_jsonlines(prediction_files)
    test_records = read_jsonlines([Path(args.test_dir) / "test.json"])
    with open(Path(args.meta_dir) / "client_ids.json") as handle:
        client_ids = json.load(handle)

    # Alignment is purely positional, so a count mismatch must be a hard failure, never a silent
    # misalignment that produces plausible-looking but wrong metrics.
    if not len(predictions) == len(test_records) == len(client_ids):
        raise ValueError(
            f"Row count mismatch: {len(predictions)} predictions, {len(test_records)} test series, "
            f"{len(client_ids)} client ids"
        )

    errors = per_client_rmse(test_records, predictions, client_ids, args.prediction_length)
    if not errors:
        raise ValueError("No client had any non-missing actual values to score against")

    rmse_values = np.array([entry["rmse"] for entry in errors])
    report = {
        "regression_metrics": {
            "rmse": {"value": float(rmse_values.mean()), "standard_deviation": float(rmse_values.std())},
        },
        "median_rmse": float(np.median(rmse_values)),
        "n_scored_clients": len(errors),
        "per_client_errors": errors,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "evaluation.json", "w") as handle:
        json.dump(report, handle)

    print(f"Scored {len(errors)} clients; mean RMSE = {rmse_values.mean():.4f}")


if __name__ == "__main__":
    main()
