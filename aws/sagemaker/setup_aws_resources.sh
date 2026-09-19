#!/usr/bin/env bash
set -euo pipefail

# --- Config ---------------------------------------------------------------
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="ts-forecast-demo-${ACCOUNT_ID}"
ROLE_NAME="ts-forecast-demo-sagemaker-role"
POLICY_NAME="ts-forecast-demo-sagemaker-policy"

echo "Account: ${ACCOUNT_ID} | Region: ${REGION} | Bucket: ${BUCKET}"

# --- 1. S3 bucket -----------------------------------------------------------
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "Bucket ${BUCKET} already exists, skipping creation."
else
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi
fi

# Explicit best practice, even though new buckets block public access by default.
# Safe to reapply every run -- these are idempotent PUTs, not creates.
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

# --- 2. IAM execution role ---------------------------------------------------
# Substitute the placeholder account ID in the scoped policy before applying
# Resolved into the current directory (not /tmp) -- avoids a path-translation
# mismatch between Git Bash's MSYS /tmp and the native Windows aws.exe binary.
RESOLVED_POLICY="./sagemaker-execution-policy.resolved.json"
sed "s/<ACCOUNT_ID>/${ACCOUNT_ID}/g" sagemaker-execution-policy.json > "$RESOLVED_POLICY"

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "Role ${ROLE_NAME} already exists, skipping creation."
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document file://trust-policy.json \
    --description "Scoped execution role for the ts-forecast-demo SageMaker project"
fi

# put-role-policy overwrites in place -- safe to rerun, keeps the policy in sync
# with sagemaker-execution-policy.json even if the role already existed.
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document "file://${RESOLVED_POLICY}"

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)

# --- 3. Lifecycle config + notebook instance for 00_ingest_raw_data.ipynb ---
# Scoped to this one notebook instance only -- not attached to any other instance
# in the project. Publishes CPU/Memory/Disk to CloudWatch (CWAgent namespace) so a
# repeat of the OOM incident is visible instead of silent.
LIFECYCLE_CONFIG_NAME="ingest-raw-data-metrics"
NOTEBOOK_INSTANCE_NAME="ts-forecast-demo-ingest"
NOTEBOOK_INSTANCE_TYPE="ml.m5.xlarge"

if aws sagemaker describe-notebook-instance-lifecycle-config \
    --notebook-instance-lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" >/dev/null 2>&1; then
  echo "Lifecycle config ${LIFECYCLE_CONFIG_NAME} already exists, updating its on-start script."
  aws sagemaker update-notebook-instance-lifecycle-config \
    --notebook-instance-lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" \
    --on-start Content="$(base64 -w0 publish-instance-metrics-on-start.sh)"
else
  aws sagemaker create-notebook-instance-lifecycle-config \
    --notebook-instance-lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" \
    --on-start Content="$(base64 -w0 publish-instance-metrics-on-start.sh)"
fi

if aws sagemaker describe-notebook-instance \
    --notebook-instance-name "$NOTEBOOK_INSTANCE_NAME" >/dev/null 2>&1; then
  echo "Notebook instance ${NOTEBOOK_INSTANCE_NAME} already exists, skipping creation."
  echo "(If you need to change its instance type or lifecycle config, stop it first, then use update-notebook-instance.)"
else
  aws sagemaker create-notebook-instance \
    --notebook-instance-name "$NOTEBOOK_INSTANCE_NAME" \
    --instance-type "$NOTEBOOK_INSTANCE_TYPE" \
    --role-arn "$ROLE_ARN" \
    --lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" \
    --tags Key=Project,Value=ts-forecast-demo Key=Stage,Value=0-ingest
fi

echo ""
echo "Bucket:              s3://${BUCKET}"
echo "Role ARN:            ${ROLE_ARN}"
echo "Notebook instance:   ${NOTEBOOK_INSTANCE_NAME} (${NOTEBOOK_INSTANCE_TYPE}, lifecycle config: ${LIFECYCLE_CONFIG_NAME})"
echo ""
echo "It takes a few minutes to reach InService. Check status with:"
echo "  aws sagemaker describe-notebook-instance --notebook-instance-name ${NOTEBOOK_INSTANCE_NAME} --query NotebookInstanceStatus"
echo ""
echo "Use BUCKET=${BUCKET} and SAGEMAKER_ROLE=${ROLE_ARN} in 00_ingest_raw_data.ipynb."

rm -f "$RESOLVED_POLICY"
