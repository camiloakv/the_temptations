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
if [ "$REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
else
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
fi

# Explicit best practice, even though new buckets block public access by default
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

# --- 2. IAM execution role ---------------------------------------------------
# Substitute the placeholder account ID in the scoped policy before applying
sed "s/<ACCOUNT_ID>/${ACCOUNT_ID}/g" sagemaker-execution-policy.json > /tmp/sagemaker-execution-policy.resolved.json

aws iam create-role \
  --role-name "$ROLE_NAME" \
  --assume-role-policy-document file://trust-policy.json \
  --description "Scoped execution role for the ts-forecast-demo SageMaker project"

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document file:///tmp/sagemaker-execution-policy.resolved.json

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)

# --- 3. Lifecycle config + notebook instance for 00_ingest_raw_data.ipynb ---
# Scoped to this one notebook instance only -- not attached to any other instance
# in the project. Publishes CPU/Memory/Disk to CloudWatch (CWAgent namespace) so a
# repeat of the OOM incident is visible instead of silent.
LIFECYCLE_CONFIG_NAME="ingest-raw-data-metrics"
NOTEBOOK_INSTANCE_NAME="ts-forecast-demo-ingest"
NOTEBOOK_INSTANCE_TYPE="ml.m5.xlarge"

aws sagemaker create-notebook-instance-lifecycle-config \
  --notebook-instance-lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" \
  --on-start Content="$(base64 -w0 publish-instance-metrics-on-start.sh)"

aws sagemaker create-notebook-instance \
  --notebook-instance-name "$NOTEBOOK_INSTANCE_NAME" \
  --instance-type "$NOTEBOOK_INSTANCE_TYPE" \
  --role-arn "$ROLE_ARN" \
  --lifecycle-config-name "$LIFECYCLE_CONFIG_NAME" \
  --tags Key=Project,Value=ts-forecast-demo Key=Stage,Value=0-ingest

echo ""
echo "Bucket:              s3://${BUCKET}"
echo "Role ARN:            ${ROLE_ARN}"
echo "Notebook instance:   ${NOTEBOOK_INSTANCE_NAME} (${NOTEBOOK_INSTANCE_TYPE}, lifecycle config: ${LIFECYCLE_CONFIG_NAME})"
echo ""
echo "It takes a few minutes to reach InService. Check status with:"
echo "  aws sagemaker describe-notebook-instance --notebook-instance-name ${NOTEBOOK_INSTANCE_NAME} --query NotebookInstanceStatus"
echo ""
echo "Use BUCKET=${BUCKET} and SAGEMAKER_ROLE=${ROLE_ARN} in 00_ingest_raw_data.ipynb."
