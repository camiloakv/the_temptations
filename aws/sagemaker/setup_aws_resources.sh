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

echo ""
echo "Bucket:    s3://${BUCKET}"
echo "Role ARN:  ${ROLE_ARN}"
echo ""
echo "Use these as BUCKET and SAGEMAKER_ROLE in the project notebooks/scripts."
