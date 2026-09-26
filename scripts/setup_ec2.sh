#!/bin/bash
# ==============================================================================
# AWS EC2 Provisioning Script for Credit Risk MLOps Serving Stack
# Run this once on a fresh Ubuntu 22.04 / 24.04 EC2 instance.
# Usage:
#   chmod +x scripts/setup_ec2.sh
#   ./scripts/setup_ec2.sh
# ==============================================================================

set -e

echo "=== [1/5] Updating system packages ==="
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release git

echo "=== [2/5] Installing Docker CE ==="
# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "=== [3/5] Configuring Docker permissions ==="
sudo usermod -aG docker "$USER"
sudo systemctl enable docker
sudo systemctl start docker

echo "=== [4/5] Preparing application directory ==="
APP_DIR="/home/$USER/credit-risk-app"
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/models"

echo "=== [5/5] Docker verification ==="
docker --version
docker compose version

echo "=============================================================================="
echo "Setup complete! Please log out and log back in to apply group permissions:"
echo "  exit"
echo "  ssh -i <your-key.pem> ubuntu@<ec2-public-ip>"
echo "=============================================================================="
