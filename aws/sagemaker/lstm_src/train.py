"""
LSTM baseline training script, run inside SageMaker's PyTorch container via script mode.

Reads the same DeepAR-format JSON Lines (start/target/cat per client) prepared in Stage 2.
Trains a 2-layer LSTM with a per-client embedding, direct multi-step output.
Evaluates on the true held-out week (from the test channel) and pushes results straight to S3.
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--context-length", type=int, default=168)
    p.add_argument("--prediction-length", type=int, default=168)
    p.add_argument("--s3-bucket", type=str, required=True)
    p.add_argument("--s3-results-prefix", type=str, required=True)

    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    p.add_argument("--job-name", type=str, default=os.environ.get("SM_TRAINING_ENV", "{}"))
    return p.parse_args()


def load_jsonlines(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def time_features(start, offset, length):
    """Cyclical hour-of-day / day-of-week features for `length` hourly steps from start+offset."""
    idx = pd.date_range(pd.Timestamp(start) + pd.Timedelta(hours=offset), periods=length, freq="h")
    hour = idx.hour.values
    dow = idx.dayofweek.values
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ],
        axis=1,
    ).astype(np.float32)


class WindowDataset(Dataset):
    """Samples random (context, future) windows from the truncated train series on the fly."""

    def __init__(self, records, context_length, prediction_length, steps_per_epoch):
        self.records = records
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.steps_per_epoch = steps_per_epoch
        self.min_len = context_length + prediction_length

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, _):
        rec = self.records[np.random.randint(len(self.records))]
        target = np.array([np.nan if v == "NaN" else v for v in rec["target"]], dtype=np.float32)
        target = np.nan_to_num(target, nan=0.0)
        if len(target) <= self.min_len:
            s = 0
        else:
            s = np.random.randint(0, len(target) - self.min_len)

        ctx = target[s : s + self.context_length]
        fut = target[s + self.context_length : s + self.context_length + self.prediction_length]
        tfeat = time_features(rec["start"], s, self.context_length)
        x = np.concatenate([ctx[:, None], tfeat], axis=1)  # (context_length, 5)
        cat = rec["cat"][0]
        return x.astype(np.float32), fut.astype(np.float32), cat


class LSTMForecaster(nn.Module):
    def __init__(self, n_clients, embedding_dim, hidden_size, num_layers, dropout, prediction_length):
        super().__init__()
        self.embedding = nn.Embedding(n_clients, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=5 + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, prediction_length)

    def forward(self, x, cat):
        emb = self.embedding(cat).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, emb], dim=2)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def evaluate_held_out_week(model, test_records, context_length, prediction_length, device):
    """Real evaluation: last context_length hours -> forecast -> compare vs. true last prediction_length hours."""
    model.eval()
    errors = []
    with torch.no_grad():
        for cat_idx, rec in enumerate(test_records):
            target = np.array([np.nan if v == "NaN" else v for v in rec["target"]], dtype=np.float32)
            if len(target) < context_length + prediction_length:
                continue
            ctx = target[-(context_length + prediction_length) : -prediction_length]
            actual = target[-prediction_length:]
            if np.isnan(actual).all():
                continue

            tfeat = time_features(rec["start"], len(target) - context_length - prediction_length, context_length)
            x = np.concatenate([np.nan_to_num(ctx)[:, None], tfeat], axis=1)
            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            cat_t = torch.tensor([cat_idx], dtype=torch.long, device=device)

            pred = model(x_t, cat_t).cpu().numpy().flatten()
            mask = ~np.isnan(actual)
            if mask.sum() > 0:
                rmse = math.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))
                errors.append({"client_id": rec.get("client_id", f"client_{cat_idx}"), "rmse": rmse})
    return errors


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.train, "train.json")
    test_path = os.path.join(args.test, "test.json")
    train_records = load_jsonlines(train_path)
    test_records = load_jsonlines(test_path)

    dataset = WindowDataset(train_records, args.context_length, args.prediction_length, args.steps_per_epoch)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    model = LSTMForecaster(
        n_clients=len(train_records),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        prediction_length=args.prediction_length,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    train_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y, cat in loader:
            x, y, cat = x.to(device), y.to(device), cat.to(device)
            optimizer.zero_grad()
            pred = model(x, cat)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch}: train_loss={epoch_loss / len(loader):.4f}")
    train_seconds = time.time() - train_start

    errors = evaluate_held_out_week(model, test_records, args.context_length, args.prediction_length, device)
    rmse_values = [e["rmse"] for e in errors]
    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    median_rmse = float(np.median(rmse_values)) if rmse_values else float("nan")

    # Required by SageMaker: printed to stdout, parsed via metric_definitions regex for HPO/console.
    print(f"validation:rmse={mean_rmse:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": vars(args),
            "n_clients": len(train_records),
        },
        os.path.join(args.model_dir, "model.pt"),
    )

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
