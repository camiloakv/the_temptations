"""Test for the pipeline's data-prep step, using tiny synthetic parquet shaped like the Stage 0 output."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[4] / "aws" / "sagemaker" / "deepar_pipeline" / "steps" / "prepare_data.py"
)

PREDICTION_LENGTH = 24
HOURS_A = 120  # 2012-12-30 00:00 .. 2013-01-03 23:00
HOURS_B = 72  # activates 2013-01-01 00:00
QUARTERS_PER_HOUR = 4
KWH_PER_QUARTER_A = 1.0
KWH_PER_QUARTER_B = 2.0


def make_raw(raw_dir):
    index = pd.date_range("2012-12-30 00:00", "2013-01-03 23:45", freq="15min")
    frames = []
    for client_id, per_quarter, active_from in [
        ("MT_A", KWH_PER_QUARTER_A, index[0]),
        ("MT_B", KWH_PER_QUARTER_B, pd.Timestamp("2013-01-01 00:00")),
        ("MT_C", 5.0, index[0]),  # not valid: must not appear in the output
    ]:
        kwh = [per_quarter if ts >= active_from else 0.0 for ts in index]
        frames.append(pd.DataFrame({"timestamp": index, "client_id": client_id, "kwh": kwh}))
    raw = pd.concat(frames, ignore_index=True)

    for year, part in raw.groupby(raw["timestamp"].dt.year):
        partition = raw_dir / f"year={year}"
        partition.mkdir(parents=True)
        part.to_parquet(partition / "part-0.parquet", index=False)


def make_metadata(path):
    metadata = pd.DataFrame(
        {
            "client_id": ["MT_A", "MT_B", "MT_C"],
            "first_active_ts": pd.to_datetime(["2012-12-30 00:00", "2013-01-01 00:00", "2012-12-30 00:00"]),
            "is_valid": [True, True, False],
        }
    )
    path.parent.mkdir(parents=True)
    metadata.to_parquet(path, index=False)


def read_jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_prepare_data_builds_deepar_splits(tmp_path):
    make_raw(tmp_path / "raw")
    make_metadata(tmp_path / "metadata" / "client_metadata.parquet")

    cmd = [
        sys.executable, str(SCRIPT),
        "--raw-dir", str(tmp_path / "raw"),
        "--metadata-path", str(tmp_path / "metadata" / "client_metadata.parquet"),
        "--output-dir", str(tmp_path / "out"),
        "--prediction-length", str(PREDICTION_LENGTH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr

    train = read_jsonlines(tmp_path / "out" / "train" / "train.json")
    test = read_jsonlines(tmp_path / "out" / "test" / "test.json")
    client_ids = json.loads((tmp_path / "out" / "meta" / "client_ids.json").read_text())

    assert client_ids == ["MT_A", "MT_B"]  # invalid client dropped, order = DeepAR "cat" index
    assert [record["cat"] for record in test] == [[0], [1]]
    assert test[0]["start"] == "2012-12-30 00:00:00"
    assert test[1]["start"] == "2013-01-01 00:00:00"  # leading zeros before activation are dropped

    assert [len(record["target"]) for record in test] == [HOURS_A, HOURS_B]
    assert [len(record["target"]) for record in train] == [HOURS_A - PREDICTION_LENGTH, HOURS_B - PREDICTION_LENGTH]
    assert test[0]["target"][0] == QUARTERS_PER_HOUR * KWH_PER_QUARTER_A  # 15-min readings summed per hour
    assert test[1]["target"][0] == QUARTERS_PER_HOUR * KWH_PER_QUARTER_B
