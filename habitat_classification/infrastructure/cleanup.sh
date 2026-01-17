#!/bin/bash
set -e

# Cleanup script for doomer-habitat Azure Function
# This will destroy all Azure resources created by Terraform

PROJECT_NAME="doomer-habitat"

echo "🗑️  Cleanup Script for ${PROJECT_NAME}"
echo "⚠️  WARNING: This will DESTROY all Azure resources!"
echo ""

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform not found. Install: brew install terraform"
    exit 1
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Install: brew install azure-cli"
    exit 1
fi

# Login check
echo "📋 Checking Azure login status..."
az account show &> /dev/null || {
    echo "⚠️  Not logged in to Azure. Running 'az login'..."
    az login
}

# Navigate to Terraform directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TERRAFORM_DIR="$SCRIPT_DIR/terraform"

if [ ! -d "$TERRAFORM_DIR" ]; then
    echo "❌ Terraform directory not found: $TERRAFORM_DIR"
    exit 1
fi

cd "$TERRAFORM_DIR"

# Check if Terraform is initialized
if [ ! -d ".terraform" ]; then
    echo "⚠️  Terraform not initialized. Running 'terraform init'..."
    terraform init
fi

# Show what will be destroyed
echo ""
echo "📋 Resources that will be destroyed:"
terraform plan -destroy

echo ""
read -p "Are you sure you want to destroy these resources? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cleanup cancelled."
    exit 0
fi

echo ""
echo "🗑️  Destroying resources..."
terraform destroy -auto-approve

echo ""
echo "✅ Cleanup complete! All resources have been destroyed."
echo "💰 You will no longer be charged for these resources."
