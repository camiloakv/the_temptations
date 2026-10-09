"""
Pipeline step 1: prepare DeepAR training data.

Runs inside a SageMaker Processing container (scikit-learn image; DuckDB is installed from
requirements.txt). Same logic as the Stage 2 notebook -- hourly resample of the valid clients, then
DeepAR JSON Lines with a truncated train split and a full-length test split -- as a standalone script.

Differences from the notebook, on purpose:
  * Inputs are mounted into the container by the ProcessingInput, so DuckDB reads local files. No
    httpfs extension download, no S3 credential plumbing.
  * Outputs go to per-execution S3 paths, so a pipeline run never overwrites the train/test files
    that the notebooks and the LSTM/TFT jobs share.
  * A client_ids.json mapping is written to a separate "meta" output. It must NOT sit in the "test"
    output: DeepAR reads every file in a channel as JSON Lines and would choke on it.

Kept compatible with Python 3.8 (what the sklearn processing image ships).
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

PROCESSING_INPUT = "/opt/ml/processing/input"
PROCESSING_OUTPUT = "/opt/ml/processing/output"
DEFAULT_PREDICTION_LENGTH = 168


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=f"{PROCESSING_INPUT}/raw")
    parser.add_argument("--metadata-path", default=f"{PROCESSING_INPUT}/metadata/client_metadata.parquet")
    parser.add_argument("--output-dir", default=PROCESSING_OUTPUT)
    parser.add_argument("--prediction-length", type=int, default=DEFAULT_PREDICTION_LENGTH)
    return parser.parse_args()


def load_hourly(raw_dir, metadata_path):
    """Hourly kWh per valid client, plus each client's activation timestamp."""
    con = duckdb.connect()
    raw_glob = f"{raw_dir}/year=*/*.parquet"

    hourly = con.sql(f"""
        SELECT
            r.client_id,
            time_bucket(INTERVAL '1 hour', r.timestamp) AS ts_hour,
            sum(r.kwh) AS kwh
        FROM read_parquet('{raw_glob}', hive_partitioning=1) r
        JOIN read_parquet('{metadata_path}') m USING (client_id)
        WHERE m.is_valid = true
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()

    activation = con.sql(f"""
        SELECT client_id, first_active_ts
        FROM read_parquet('{metadata_path}')
        WHERE is_valid = true
    """).df().set_index("client_id")["first_active_ts"]

    return hourly, activation


def to_deepar_target(values):
    """DeepAR wants missing values as the literal string 'NaN', not JSON null."""
    return ["NaN" if pd.isna(value) else round(float(value), 4) for value in values]


def build_records(hourly, activation, prediction_length):
    """One record per client: full-length for test, last `prediction_length` hours cut for train.

    Each series starts at its own activation hour, so the leading zeros of clients onboarded later
    never enter training. The position in the (sorted) client list is the DeepAR "cat" feature.
    """
    train_records, test_records, client_ids = [], [], []

    for position, (client_id, group) in enumerate(hourly.groupby("client_id")):
        start = activation[client_id].floor("h")
        full_range = pd.date_range(start, group["ts_hour"].max(), freq="h")
        series = group.set_index("ts_hour")["kwh"].reindex(full_range)

        record = {
            "start": start.strftime("%Y-%m-%d %H:%M:%S"),
            "target": to_deepar_target(series.values),
            "cat": [position],
        }
        train_record = dict(record)
        train_record["target"] = record["target"][:-prediction_length]

        train_records.append(train_record)
        test_records.append(record)
        client_ids.append(client_id)

    return train_records, test_records, client_ids


def write_jsonlines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main():
    args = parse_args()
    hourly, activation = load_hourly(args.raw_dir, args.metadata_path)
    train_records, test_records, client_ids = build_records(hourly, activation, args.prediction_length)

    output = Path(args.output_dir)
    write_jsonlines(output / "train" / "train.json", train_records)
    write_jsonlines(output / "test" / "test.json", test_records)

    (output / "meta").mkdir(parents=True, exist_ok=True)
    with open(output / "meta" / "client_ids.json", "w") as handle:
        json.dump(client_ids, handle)

    print(f"Prepared {len(train_records)} series; outputs written under {output}")


if __name__ == "__main__":
    main()
