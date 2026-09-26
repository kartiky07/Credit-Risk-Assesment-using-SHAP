# Complete AWS EC2 Deployment & Evidently AI Monitoring Guide

This guide walks you through deploying your **Credit Risk Assessment & Explainability Service** to an **AWS EC2 instance** with Docker Compose, automated **GitHub Actions CI/CD**, and continuous **Evidently AI data drift monitoring**.

---

## 1. Launching Your AWS EC2 Instance

1. Log into your [AWS Management Console](https://console.aws.amazon.com/ec2/).
2. Navigate to **EC2** ➔ Click **Launch instance**.
3. Configure the following settings:
   - **Name:** `credit-risk-mlops-server`
   - **Application and OS Images:** **Ubuntu Server 24.04 LTS (HVM)**, 64-bit (x86)
   - **Instance Type:**
     - Free-tier: `t2.micro` (1 vCPU, 1 GB RAM)
     - Recommended: `t3.small` (2 vCPU, 2 GB RAM) for smoother ML inference & calibration
   - **Key Pair (login):**
     - Click **Create new key pair**.
     - Name: `credit-risk-key`
     - Key pair type: `RSA`
     - Private key file format: `.pem`
     - Download and store `credit-risk-key.pem` securely on your machine.
   - **Network settings (Firewall / Security Group):**
     Click **Edit** and add the following Inbound Security Group Rules:

     | Type | Protocol | Port Range | Source | Purpose |
     | :--- | :--- | :--- | :--- | :--- |
     | **SSH** | TCP | `22` | My IP | Secure SSH Terminal Access |
     | **HTTP** | TCP | `80` | `0.0.0.0/0` | Public Web Interface |
     | **Custom TCP** | TCP | `8000` | `0.0.0.0/0` | FastAPI Serving Endpoint |
     | **Custom TCP** | TCP | `5000` | My IP | MLflow Dashboard UI |

   - **Storage:** 20 GiB gp3 (within the 30 GiB AWS Free Tier allowance).
4. Click **Launch instance**.

---

## 2. One-Time Server Setup (Takes 2 Minutes)

On your local terminal, navigate to where you saved `credit-risk-key.pem`:

```bash
# 1. Set read-only permissions on your private key
chmod 400 credit-risk-key.pem

# 2. Connect to your EC2 instance via SSH
ssh -i credit-risk-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
```

Once inside the EC2 terminal, run the automated provisioning script:

```bash
# 3. Download and execute the provisioning script
curl -fsSL https://raw.githubusercontent.com/<YOUR-GITHUB-USERNAME>/Credit-Risk-Assesment-using-SHAP/main/scripts/setup_ec2.sh -o setup.sh
chmod +x setup.sh
./setup.sh
```

*(This automatically installs Docker CE, Docker Compose, Git, and prepares the application directory `/home/ubuntu/credit-risk-app`).*

Log out and back in once to apply Docker group permissions:
```bash
exit
ssh -i credit-risk-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
```

---

## 3. GitHub Actions Continuous Deployment (Automated Push-to-Deploy)

We configured [.github/workflows/deploy-ec2.yml](file:///Users/yadavkartik/Desktop/Credit_Default/Credit-Risk-Assesment-using-SHAP/.github/workflows/deploy-ec2.yml) to automatically test, build, push the Docker container to GitHub Container Registry, and update your EC2 server whenever you push code!

### Add EC2 Secrets to GitHub:
1. Open your GitHub Repository in your browser.
2. Go to **Settings** ➔ **Secrets and variables** ➔ **Actions**.
3. Click **New repository secret** and add the following 3 secrets:

| Secret Name | Value | Example |
| :--- | :--- | :--- |
| `EC2_HOST` | Your EC2 Public IPv4 Address or Public DNS | `54.210.14.88` |
| `EC2_USER` | The default Ubuntu username | `ubuntu` |
| `EC2_SSH_KEY` | Entire content of your `credit-risk-key.pem` file | `-----BEGIN RSA PRIVATE KEY-----...` |

*(To copy your `.pem` key content on Mac terminal, run: `cat credit-risk-key.pem | pbcopy`)*

---

## 4. Deploying Your Code

Commit and push your changes to GitHub:

```bash
git add .
git commit -m "Configure AWS EC2 automated deployment and Evidently monitoring"
git push origin main
```

Go to your repository's **Actions** tab on GitHub:
- You will see the **Deploy to AWS EC2** workflow running.
- It will execute the tests, build the multi-platform Docker container, push to `ghcr.io`, SSH into your EC2 instance, and boot the containers with zero downtime!

Once finished, open in your browser:
- **Web Interface:** `http://<YOUR-EC2-PUBLIC-IP>` (or port `:8000`)
- **Interactive Swagger Docs:** `http://<YOUR-EC2-PUBLIC-IP>:8000/docs`
- **MLflow Tracking UI:** `http://<YOUR-EC2-PUBLIC-IP>:5000`

---

## 5. Evidently AI Drift & Data Quality Monitoring

Your application now automatically records every applicant evaluation into `data/inferences.db` using asynchronous background tasks.

### 1. View the Live Interactive Drift Dashboard
Open in your browser:
```
http://<YOUR-EC2-PUBLIC-IP>:8000/monitoring/report
```
This renders an interactive Evidently AI dashboard displaying:
- **Data Drift Summary:** Kolmogorov-Smirnov & Wasserstein drift tests comparing live applicant features (`loan_percent_income`, `person_income`, `loan_int_rate`) against the baseline training distribution.
- **Data Quality & Distributions:** Quantile distributions, missing value counts, and outlier checks.

### 2. Check Drift Status API (JSON)
```bash
curl http://<YOUR-EC2-PUBLIC-IP>:8000/monitoring/status
```
Example response:
```json
{
  "dataset_drift_detected": false,
  "share_of_drifted_columns": 0.0,
  "number_of_drifted_columns": 0,
  "number_of_columns": 11,
  "reference_samples": 25217,
  "current_samples": 200,
  "drifted_features": []
}
```

### 3. Trigger On-Demand Drift Analysis
```bash
curl -X POST http://<YOUR-EC2-PUBLIC-IP>:8000/monitoring/run
```
