#!/bin/bash
set -e

# Deployment script for habitat classification Azure Function
# Usage: ./deploy.sh

PROJECT_NAME="doomer-habitat"

echo "🚀 Deploying to Azure Functions"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Install: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

# Login check
echo "📋 Checking Azure login status..."
az account show &> /dev/null || {
    echo "⚠️  Not logged in to Azure. Running 'az login'..."
    az login
}

# Get function app name from Terraform output or construct it
FUNCTION_APP_NAME="func-${PROJECT_NAME}"
RESOURCE_GROUP="rg-${PROJECT_NAME}"

echo "📦 Building deployment package..."

# Create temporary directory for deployment
TEMP_DIR=$(mktemp -d)
echo "   Using temp directory: $TEMP_DIR"

# Copy necessary files
cd "$(dirname "$0")/.."  # Go to project root
cp -r api.py model.py utils.py function_app.py host.json data "$TEMP_DIR/"
cp infrastructure/requirements-azure.txt "$TEMP_DIR/requirements.txt"
cp .funcignore "$TEMP_DIR/" 2>/dev/null || true

echo "🔧 Deploying to Azure Function: $FUNCTION_APP_NAME"

# Deploy using Azure Functions Core Tools or zip deploy
cd "$TEMP_DIR"

# Option 1: Using func Azure Functions Core Tools (recommended)
if command -v func &> /dev/null; then
    func azure functionapp publish "$FUNCTION_APP_NAME" --python
else
    # Option 2: Using zip deploy (fallback)
    echo "   Azure Functions Core Tools not found, using zip deploy..."
    zip -r ../deploy.zip .
    az functionapp deployment source config-zip \
        --resource-group "$RESOURCE_GROUP" \
        --name "$FUNCTION_APP_NAME" \
        --src ../deploy.zip
fi

# Cleanup
cd - > /dev/null
rm -rf "$TEMP_DIR"

echo "✅ Deployment complete!"
echo "🌐 Function URL: https://${FUNCTION_APP_NAME}.azurewebsites.net"
echo ""
echo "Test with:"
echo "  curl https://${FUNCTION_APP_NAME}.azurewebsites.net/api/ping"
