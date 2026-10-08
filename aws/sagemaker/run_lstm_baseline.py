"""
CD entry point: launches the LSTM baseline SageMaker training job.

Same logic as 02b_train_lstm_baseline.ipynb's baseline cell, extracted into a plain script so
GitHub Actions can run it headlessly on merge/tag. Intentionally launches only the baseline
sanity job, not HPO -- CD's job is to prove the pipeline still works end to end on every merge,
not to re-run a full tuning search on every push.
"""

import os
import sys

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

BUCKET = os.environ["BUCKET"]
SAGEMAKER_ROLE = f"arn:aws:iam::{os.environ['ACCOUNT_ID']}:role/ts-forecast-demo-sagemaker-role"

DEEPAR_PREFIX = "ts-forecast-demo/deepar-v1"
LSTM_PREFIX = "ts-forecast-demo/lstm-v1"
RESULTS_PREFIX = "ts-forecast-demo/results/lstm-v1"
CODE_LOCATION = f"s3://{BUCKET}/{LSTM_PREFIX}/code"

CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 168

TAGS = [
    {"Key": "Project", "Value": "ts-forecast-demo"},
    {"Key": "Stage", "Value": "2b-lstm-v1"},
    {"Key": "Model", "Value": "lstm"},
    {"Key": "TriggeredBy", "Value": "github-actions-cd"},
]

METRIC_DEFINITIONS = [{"Name": "validation:rmse", "Regex": "validation:rmse=([0-9.]+)"}]


def main():
    session = sagemaker.Session()

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=os.path.join(os.path.dirname(__file__), "lstm_src"),
        role=SAGEMAKER_ROLE,
        instance_type="ml.c5.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        output_path=f"s3://{BUCKET}/{LSTM_PREFIX}/model-cd",
        code_location=CODE_LOCATION,
        hyperparameters={
            "hidden-size": 128,
            "num-layers": 2,
            "embedding-dim": 16,
            "dropout": 0.1,
            "learning-rate": 1e-3,
            "epochs": 30,
            "batch-size": 64,
            "steps-per-epoch": 200,
            "context-length": CONTEXT_LENGTH,
            "prediction-length": PREDICTION_LENGTH,
            "s3-bucket": BUCKET,
            "s3-results-prefix": RESULTS_PREFIX,
        },
        metric_definitions=METRIC_DEFINITIONS,
        tags=TAGS,
        sagemaker_session=session,
    )

    train_s3 = f"s3://{BUCKET}/{DEEPAR_PREFIX}/train/"
    test_s3 = f"s3://{BUCKET}/{DEEPAR_PREFIX}/test/"

    print(f"Launching training job, instance_type={estimator.instance_type}")
    estimator.fit({"train": TrainingInput(train_s3), "test": TrainingInput(test_s3)})

    job_name = estimator.latest_training_job.name
    status = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
    print(f"Job {job_name} finished with status: {status}")

    if status != "Completed":
        sys.exit(1)  # non-zero exit fails the GitHub Actions run


if __name__ == "__main__":
    main()
