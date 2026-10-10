# The Temptations

<p align="right"><i>
I'm doing fine<br>
Up here, on Cloud Nine<br>
<br>
The Temptations - Cloud Nine
</i></p>

![Amazon Web Services](https://img.shields.io/badge/Amazon_Web_Services-FF9900?style=for-the-badge&logo=amazonwebservices&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
![Microsoft Azure](https://img.shields.io/badge/microsoft_azure-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white)

![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)

This is a permanently ongoing project exploring the tooling of main cloud providers.


## Status

<table>
  <thead>
    <tr>
      <th>Provider</th>
      <th>Tool</th>
    </tr>
  </thead>
  <tbody>
    <!-- Amazon Web Services -------------------->
    <tr>
      <td rowspan="3">AWS</td>
      <td>✅ SageMaker</td>
    </tr>
    <tr>
      <!-- first cell occupied by "AWS" -->
      <td>⬜ Step Functions</td>
    </tr>
    <tr>
      <!-- first cell occupied by "AWS" -->
      <td>⬜ Lambda</td>
    </tr>
    <tr>
    <!-- Google Cloud Platform ------------------>
      <td>GCP</td>
      <td>⬜ -</td>
    </tr>
    <!-- Microsoft Azure ------------------------>
    <tr>
      <td>Azure</td>
      <td>⬜ -</td>
    </tr>
  </tbody>
</table>


## Setup (sketch)

<!--this line just for refference commits .-->

### AWS

1. **Download the AWS CLI**. On Windows, the easiest way to use the AWS CLI is through the Git Bash terminal.
    1. Run `irm 'https://awscli.amazonaws.com/AWSCLIV2.msi' -OutFile 'AWSCLIV2.msi'; Start-Process msiexec.exe -Wait -ArgumentList '/i AWSCLIV2.msi /qn'`.
    2. If that doesn't automatically install the CLI, manually execute the installer `AWSCLIV2.msi` downloaded.
    3. Verify the installs running `aws --version`
    4. Get an access key: AWS IAM console → Users → your user → Security credentials tab → Create access key → choose 'Command Line Interface (CLI)' as the use case.
    5. Configure credentials running `aws configure`. Paste the keys, default region (e.g. us-east-1), and default output format (e.g. json).
    6. Confirm it works running `aws sts get-caller-identity`.
2. **Set up resources**. Run `chmod +x setup_aws_resources.sh` and `./setup_aws_resources.sh`.


#### Sagemaker
