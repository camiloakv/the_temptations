#!/bin/bash
set -e

# Publishes system-level metrics (CPU, memory, disk) from the notebook instance to CloudWatch.
# Requires: internet connectivity, and cloudwatch:PutMetricData on the execution role.
# Source: aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples

NOTEBOOK_INSTANCE_NAME=$(jq '.ResourceName' \
                      /opt/ml/metadata/resource-metadata.json --raw-output)

echo "Fetching the CloudWatch agent configuration file."
wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/publish-instance-metrics/amazon-cloudwatch-agent.json

sed -i -- "s/MyNotebookInstance/$NOTEBOOK_INSTANCE_NAME/g" amazon-cloudwatch-agent.json

echo "Starting the CloudWatch agent on the Notebook Instance."
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a \
    append-config -m ec2 -c file://$(pwd)/amazon-cloudwatch-agent.json

CURR_VERSION=$(cat /etc/os-release)
if [[ $CURR_VERSION == *$"http://aws.amazon.com/amazon-linux-ami/"* ]]; then
    restart restart-cloudwatch-agent
else
    systemctl restart amazon-cloudwatch-agent.service
fi

rm amazon-cloudwatch-agent.json
