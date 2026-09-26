"""
TFT (Temporal Fusion Transformer) training script, SageMaker PyTorch script mode.

Reuses the same DeepAR-format JSON Lines (start/target/cat) as the LSTM baseline. Converts to the
long-format DataFrame pytorch_forecasting expects, trains a TFT via pytorch_forecasting + lightning,
evaluates on the true held-out week, pushes results straight to S3 -- same result schema as
lstm_src/train.py so both are directly comparable.

Note: pytorch_forecasting's API has shifted across versions; verify against the installed version's
docs (requirements.txt pins pytorch-forecasting<2.0) if this errors on an unfamiliar signature.
"""

import argparse
import json
import math
import os
import time

import boto3
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--attention-head-size", type=int, default=4)
    p.add_argument("--hidden-continuous-size", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--context-length", type=int, default=168)
    p.add_argument("--prediction-length", type=int, default=168)
    p.add_argument("--max-history-hours", type=int, default=24 * 14)  # 14 days by default now
    p.add_argument("--max-series", type=int, default=30)  # subsampled by default, not just for the sanity job
    p.add_argument("--limit-train-batches", type=float, default=10)
    p.add_argument("--limit-val-batches", type=float, default=2)
    p.add_argument("--s3-bucket", type=str, required=True)
    p.add_argument("--s3-results-prefix", type=str, required=True)

    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    p.add_argument("--job-name", type=str, default=os.environ.get("SM_TRAINING_ENV", "{}"))
    return p.parse_args()


def load_jsonlines(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def to_long_df(records, max_history_hours=None):
    """DeepAR JSON records -> long-format DataFrame: series, time_idx, value, hour, dow.

    max_history_hours, if set, keeps only the most recent N hours per series -- this is the main
    lever against TimeSeriesDataSet's memory use, since it enumerates every valid encoder/decoder
    window across the full history otherwise (333 series x ~30k hours = ~10M windows).
    """
    frames = []
    for series_idx, rec in enumerate(records):
        target = np.array([np.nan if v == "NaN" else v for v in rec["target"]], dtype=np.float32)
        start = pd.Timestamp(rec["start"])
        if max_history_hours is not None and len(target) > max_history_hours:
            offset_hours = len(target) - max_history_hours
            target = target[-max_history_hours:]
            start = start + pd.Timedelta(hours=offset_hours)
        idx = pd.date_range(start, periods=len(target), freq="h")
        frames.append(
            pd.DataFrame(
                {
                    "series": str(series_idx),
                    "time_idx": np.arange(len(target), dtype=np.int32),
                    "value": target,
                    "hour": pd.Categorical(idx.hour.astype(str)),
                    "dow": pd.Categorical(idx.dayofweek.astype(str)),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["value"] = df.groupby("series")["value"].transform(lambda s: s.ffill().bfill())
    return df


def build_datasets(train_df, context_length, prediction_length):
    max_time_idx = train_df["time_idx"].max()
    training = TimeSeriesDataSet(
        train_df[train_df.time_idx <= max_time_idx - prediction_length],
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        time_varying_known_categoricals=["hour", "dow"],
        time_varying_unknown_reals=["value"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, train_df, predict=True, stop_randomization=True)
    return training, validation


def evaluate_held_out_week(model, records, context_length, prediction_length, device):
    """Same evaluation contract as the LSTM script: last context_length -> forecast -> vs. true tail.

    Builds only the small per-series tail slice needed, not a full-history long DataFrame for all
    333 series -- that would hit the same memory problem truncation fixes in to_long_df/training.
    """
    window = context_length + prediction_length
    errors = []
    model.eval()
    for series_idx, rec in enumerate(records):
        target = np.array([np.nan if v == "NaN" else v for v in rec["target"]], dtype=np.float32)
        if len(target) < window:
            continue
        tail = target[-window:]
        start = pd.Timestamp(rec["start"]) + pd.Timedelta(hours=len(target) - window)
        idx = pd.date_range(start, periods=window, freq="h")

        series_df = pd.DataFrame(
            {
                "series": str(series_idx),
                "time_idx": np.arange(window, dtype=np.int32),
                "value": pd.Series(tail).ffill().bfill().values,
                "hour": pd.Categorical(idx.hour.astype(str)),
                "dow": pd.Categorical(idx.dayofweek.astype(str)),
            }
        )
        decoder_actuals = tail[-prediction_length:]

        try:
            with torch.no_grad():
                raw_pred = model.predict(series_df, mode="prediction")
            pred = raw_pred.numpy().flatten()[:prediction_length]
            mask = ~np.isnan(decoder_actuals)
            if mask.sum() > 0:
                rmse = math.sqrt(np.mean((decoder_actuals[mask] - pred[mask]) ** 2))
                errors.append({"client_id": f"client_{series_idx}", "rmse": rmse})
        except Exception as e:  # keep evaluation resilient to a handful of edge-case series
            print(f"eval skipped for series {series_idx}: {e}")
    return errors


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_records = load_jsonlines(os.path.join(args.train, "train.json"))
    test_records = load_jsonlines(os.path.join(args.test, "test.json"))

    if args.max_series is not None:
        train_records = train_records[: args.max_series]
        test_records = test_records[: args.max_series]  # same order as train, indices still correspond

    train_df = to_long_df(train_records, max_history_hours=args.max_history_hours)
    training, validation = build_datasets(train_df, args.context_length, args.prediction_length)

    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        hidden_continuous_size=args.hidden_continuous_size,
        dropout=args.dropout,
        loss=QuantileLoss(),
        learning_rate=args.learning_rate,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    train_start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_seconds = time.time() - train_start

    errors = evaluate_held_out_week(model, test_records, args.context_length, args.prediction_length, device)
    rmse_values = [e["rmse"] for e in errors]
    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    median_rmse = float(np.median(rmse_values)) if rmse_values else float("nan")

    print(f"validation:rmse={mean_rmse:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))

    try:
        job_name = json.loads(args.job_name).get("job_name", "unknown-job")
    except (json.JSONDecodeError, AttributeError):
        job_name = "unknown-job"

    result = {
        "job_name": job_name,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("train", "test", "model_dir", "job_name")},
        "train_seconds": train_seconds,
        "mean_rmse": mean_rmse,
        "median_rmse": median_rmse,
        "n_clients_evaluated": len(errors),
        "per_client_errors": errors,
    }

    s3 = boto3.client("s3")
    key = f"{args.s3_results_prefix}/{job_name}.json"
    s3.put_object(Bucket=args.s3_bucket, Key=key, Body=json.dumps(result).encode())
    print(f"Results uploaded to s3://{args.s3_bucket}/{key}")


if __name__ == "__main__":
    main()
