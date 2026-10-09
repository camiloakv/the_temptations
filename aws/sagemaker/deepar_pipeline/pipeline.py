"""
DeepAR training pipeline:

  PrepareData -> TrainDeepAR -> CreateDeepARModel -> BatchEvaluate -> EvaluateModel -> CheckRmse
                                                                                        |-- pass: RegisterDeepAR
                                                                                        '-- fail: RmseAboveThreshold

Usage (needs BUCKET and ACCOUNT_ID in the environment, like the notebooks):
  python pipeline.py                     # create/update the pipeline definition only (free)
  python pipeline.py --start             # ... and start an execution
  python pipeline.py --start --wait      # ... and block until it finishes (exit 1 on failure)

Hyperparameters are not re-tuned here: they are read from the best-HPO entry persisted by the Stage 2
notebook (results/deepar-v1/model_card.json) and baked into the definition at upsert time.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, WaiterError
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.transformer import Transformer
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep, TransformStep

PIPELINE_NAME = "ts-forecast-demo-deepar"
MODEL_PACKAGE_GROUP = "ts-forecast-demo-deepar"

PROJECT_PREFIX = "ts-forecast-demo"
RAW_PREFIX = f"{PROJECT_PREFIX}/raw/electricity"
METADATA_KEY = f"{PROJECT_PREFIX}/curated/client_metadata.parquet"
MODEL_CARD_KEY = f"{PROJECT_PREFIX}/results/deepar-v1/model_card.json"
PIPELINE_PREFIX = f"{PROJECT_PREFIX}/pipelines/deepar"

STEPS_DIR = str(Path(__file__).resolve().parent / "steps")
PROCESSING_INPUT = "/opt/ml/processing/input"
PROCESSING_OUTPUT = "/opt/ml/processing/output"
EVALUATION_OUTPUT = "/opt/ml/processing/evaluation"

SKLEARN_VERSION = "1.2-1"
PROCESSING_INSTANCE = "ml.m5.xlarge"  # one quota covers both processing steps; they never run in parallel
TRAINING_INSTANCE = "ml.c5.xlarge"
INFERENCE_INSTANCE = "ml.m5.large"

FREQ = "H"
PREDICTION_LENGTH = 168
DEFAULT_MAX_RMSE = 150.0  # sanity gate, not a tight bar: Stage 2's DeepAR scored ~117 on this metric
APPROVAL_STATUSES = ["PendingManualApproval", "Approved", "Rejected"]
REQUIRED_HYPERPARAMETERS = ("epochs", "context_length", "num_cells", "num_layers", "learning_rate")

WAIT_DELAY_SECONDS = 30
WAIT_MAX_ATTEMPTS = 180  # 90 minutes

JSONLINES = "application/jsonlines"
JSON = "application/json"

TAGS = [
    {"Key": "Project", "Value": "ts-forecast-demo"},
    {"Key": "Stage", "Value": "3-pipeline"},
    {"Key": "Model", "Value": "deepar"},
]


def load_best_hyperparameters(s3, bucket):
    """Best-HPO hyperparameters persisted by the Stage 2 notebook, minus SageMaker-internal keys."""
    body = s3.get_object(Bucket=bucket, Key=MODEL_CARD_KEY)["Body"].read()
    best = json.loads(body)["best_hyperparameters"]

    hyperparameters = {key: str(value) for key, value in best.items() if not key.startswith("_")}
    hyperparameters["time_freq"] = FREQ
    hyperparameters["prediction_length"] = str(PREDICTION_LENGTH)

    missing = [key for key in REQUIRED_HYPERPARAMETERS if key not in hyperparameters]
    if missing:
        raise ValueError(f"{MODEL_CARD_KEY} is missing hyperparameters: {missing}")
    return hyperparameters


def execution_path(bucket, *suffix):
    """Per-execution S3 location, so runs never overwrite each other (or the shared Stage 2 data)."""
    return Join(
        on="/",
        values=["s3:/", bucket, PIPELINE_PREFIX, ExecutionVariables.PIPELINE_EXECUTION_ID, *suffix],
    )


def make_processor(session, role_arn, bucket):
    return FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version=SKLEARN_VERSION,
        role=role_arn,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        base_job_name="deepar-pipeline",
        code_location=f"s3://{bucket}/{PIPELINE_PREFIX}/code",  # our bucket, not SageMaker's default one
        tags=TAGS,
        sagemaker_session=session,
    )


def make_prepare_step(session, role_arn, bucket):
    # Caching skips this step when its arguments are unchanged. Fine here: the raw data is a static
    # historical dataset. With data that changes in place, drop the cache or add a data-version parameter.
    step_args = make_processor(session, role_arn, bucket).run(
        code="prepare_data.py",
        source_dir=STEPS_DIR,
        inputs=[
            ProcessingInput(source=f"s3://{bucket}/{RAW_PREFIX}/", destination=f"{PROCESSING_INPUT}/raw"),
            ProcessingInput(source=f"s3://{bucket}/{METADATA_KEY}", destination=f"{PROCESSING_INPUT}/metadata"),
        ],
        outputs=[
            ProcessingOutput(
                output_name=name,
                source=f"{PROCESSING_OUTPUT}/{name}",
                destination=execution_path(bucket, name),
            )
            for name in ("train", "test", "meta")
        ],
        arguments=["--prediction-length", str(PREDICTION_LENGTH)],
    )
    return ProcessingStep(
        name="PrepareData",
        step_args=step_args,
        cache_config=CacheConfig(enable_caching=True, expire_after="P7D"),
    )


def make_train_step(session, role_arn, bucket, hyperparameters, step_prepare):
    estimator = Estimator(
        image_uri=retrieve("forecasting-deepar", session.boto_region_name),
        role=role_arn,
        instance_type=TRAINING_INSTANCE,
        instance_count=1,
        output_path=f"s3://{bucket}/{PIPELINE_PREFIX}/models",
        hyperparameters=hyperparameters,
        tags=TAGS,
        sagemaker_session=session,
    )
    outputs = step_prepare.properties.ProcessingOutputConfig.Outputs
    step_args = estimator.fit(
        {
            "train": TrainingInput(s3_data=outputs["train"].S3Output.S3Uri, content_type="json"),
            "test": TrainingInput(s3_data=outputs["test"].S3Output.S3Uri, content_type="json"),
        }
    )
    return TrainingStep(name="TrainDeepAR", step_args=step_args)


def make_model(session, role_arn, step_train):
    return Model(
        image_uri=retrieve("forecasting-deepar", session.boto_region_name),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=session,
    )


def make_transform_steps(session, bucket, model, step_prepare):
    step_create_model = ModelStep(name="CreateDeepARModel", step_args=model.create(instance_type=INFERENCE_INSTANCE))
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type=INFERENCE_INSTANCE,
        instance_count=1,
        strategy="SingleRecord",
        assemble_with="Line",  # without this, records are concatenated on one line (the Stage 2 bug)
        accept=JSONLINES,
        output_path=execution_path(bucket, "transform"),
        tags=TAGS,
        sagemaker_session=session,
    )
    train_uri = step_prepare.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    step_transform = TransformStep(
        name="BatchEvaluate",
        step_args=transformer.transform(data=train_uri, content_type=JSONLINES, split_type="Line"),
    )
    return step_create_model, step_transform


def make_evaluate_step(session, role_arn, bucket, step_prepare, step_transform):
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    outputs = step_prepare.properties.ProcessingOutputConfig.Outputs
    step_args = make_processor(session, role_arn, bucket).run(
        code="evaluate.py",
        source_dir=STEPS_DIR,
        inputs=[
            ProcessingInput(
                source=step_transform.properties.TransformOutput.S3OutputPath,
                destination=f"{PROCESSING_INPUT}/predictions",
            ),
            ProcessingInput(source=outputs["test"].S3Output.S3Uri, destination=f"{PROCESSING_INPUT}/test"),
            ProcessingInput(source=outputs["meta"].S3Output.S3Uri, destination=f"{PROCESSING_INPUT}/meta"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source=EVALUATION_OUTPUT,
                destination=execution_path(bucket, "evaluation"),
            )
        ],
        arguments=["--prediction-length", str(PREDICTION_LENGTH)],
    )
    step_evaluate = ProcessingStep(name="EvaluateModel", step_args=step_args, property_files=[evaluation_report])
    return step_evaluate, evaluation_report


def make_gate_step(model, step_evaluate, evaluation_report, max_rmse, approval_status, git_sha):
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri, "evaluation.json"],
            ),
            content_type=JSON,
        )
    )
    step_register = ModelStep(
        name="RegisterDeepAR",
        step_args=model.register(
            content_types=[JSONLINES, JSON],
            response_types=[JSONLINES, JSON],
            inference_instances=[INFERENCE_INSTANCE],
            transform_instances=[INFERENCE_INSTANCE],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status=approval_status,
            model_metrics=model_metrics,
            description="DeepAR hourly electricity-load forecaster (1-week horizon)",
            # Ties this model version back to the exact code that produced it.
            customer_metadata_properties={"git_sha": git_sha, "pipeline": PIPELINE_NAME},
        ),
    )
    step_fail = FailStep(
        name="RmseAboveThreshold",
        error_message=Join(on=" ", values=["Mean RMSE is above MaxRmse =", max_rmse, "- model not registered."]),
    )
    return ConditionStep(
        name="CheckRmse",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="regression_metrics.rmse.value",
                ),
                right=max_rmse,
            )
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )


def build_pipeline(session, role_arn, bucket, hyperparameters, git_sha):
    max_rmse = ParameterFloat(name="MaxRmse", default_value=DEFAULT_MAX_RMSE)
    approval_status = ParameterString(name="ModelApprovalStatus", default_value=APPROVAL_STATUSES[0])

    step_prepare = make_prepare_step(session, role_arn, bucket)
    step_train = make_train_step(session, role_arn, bucket, hyperparameters, step_prepare)
    model = make_model(session, role_arn, step_train)
    step_create_model, step_transform = make_transform_steps(session, bucket, model, step_prepare)
    step_evaluate, evaluation_report = make_evaluate_step(session, role_arn, bucket, step_prepare, step_transform)
    step_gate = make_gate_step(model, step_evaluate, evaluation_report, max_rmse, approval_status, git_sha)

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[max_rmse, approval_status],
        steps=[step_prepare, step_train, step_create_model, step_transform, step_evaluate, step_gate],
        sagemaker_session=session,
    )


def model_package_group_exists(sm, name):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=name)
    except ClientError:
        return False
    else:
        return True


def ensure_model_package_group(sm, name):
    """The registry group must exist before a register step can write to it."""
    if not model_package_group_exists(sm, name):
        sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription="DeepAR electricity-load forecaster versions",
            Tags=TAGS,
        )


def report_steps(execution):
    for step in execution.list_steps():
        print(f"  {step['StepName']:22s} {step['StepStatus']:10s} {step.get('FailureReason', '')}")
        registered = step.get("Metadata", {}).get("RegisterModel", {}).get("Arn")
        if registered:
            print(f"    -> registered model package: {registered}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true", help="start an execution after upserting")
    parser.add_argument("--wait", action="store_true", help="wait for the execution (needs --start)")
    parser.add_argument("--max-rmse", type=float, default=DEFAULT_MAX_RMSE)
    parser.add_argument("--approval-status", choices=APPROVAL_STATUSES, default=APPROVAL_STATUSES[0])
    args = parser.parse_args()
    if args.wait and not args.start:
        parser.error("--wait requires --start")
    return args


def main():
    args = parse_args()
    bucket = os.environ["BUCKET"]
    role_arn = f"arn:aws:iam::{os.environ['ACCOUNT_ID']}:role/ts-forecast-demo-sagemaker-role"
    git_sha = os.environ.get("GITHUB_SHA", "local")

    boto_session = boto3.Session()
    session = PipelineSession(boto_session=boto_session, default_bucket=bucket)

    hyperparameters = load_best_hyperparameters(boto_session.client("s3"), bucket)
    print(f"Hyperparameters (from {MODEL_CARD_KEY}): {hyperparameters}")

    ensure_model_package_group(boto_session.client("sagemaker"), MODEL_PACKAGE_GROUP)
    pipeline = build_pipeline(session, role_arn, bucket, hyperparameters, git_sha)
    pipeline.upsert(role_arn=role_arn, description="DeepAR train/evaluate/register", tags=TAGS)
    print(f"Upserted pipeline {PIPELINE_NAME}")

    if not args.start:
        return 0

    execution = pipeline.start(parameters={"MaxRmse": args.max_rmse, "ModelApprovalStatus": args.approval_status})
    print(f"Started execution: {execution.arn}")
    if not args.wait:
        return 0

    try:
        execution.wait(delay=WAIT_DELAY_SECONDS, max_attempts=WAIT_MAX_ATTEMPTS)
    except WaiterError:
        print("Execution did not succeed:")
        report_steps(execution)
        return 1
    print("Execution succeeded:")
    report_steps(execution)
    return 0


if __name__ == "__main__":
    sys.exit(main())
