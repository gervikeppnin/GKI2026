#!/bin/bash
set -e

# Infrastructure setup script for doomer-habitat Azure Function
# This initializes and deploys all Azure resources using Terraform

PROJECT_NAME="doomer-habitat"

echo "🏗️  Infrastructure Setup for ${PROJECT_NAME}"
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

# Initialize Terraform
echo ""
echo "🔧 Initializing Terraform..."
terraform init

# Validate configuration
echo ""
echo "✅ Validating Terraform configuration..."
terraform validate

# Show plan
echo ""
echo "📋 Terraform Plan - Resources to be created:"
echo ""
terraform plan

# Confirm before applying
echo ""
read -p "Do you want to create these resources? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Setup cancelled."
    exit 0
fi

# Apply configuration
echo ""
echo "🚀 Creating infrastructure..."
terraform apply -auto-approve

# Show outputs
echo ""
echo "✅ Infrastructure created successfully!"
echo ""
echo "📊 Infrastructure details:"
terraform output

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Deploy your function code:"
echo "     ./infrastructure/deploy.sh"
echo ""
echo "  2. Test your function:"
echo "     curl \$(terraform -chdir=infrastructure/terraform output -raw function_app_url)/ping"
