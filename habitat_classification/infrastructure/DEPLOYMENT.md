# Azure Functions Deployment Guide

## Prerequisites

1. **Azure CLI**: `brew install azure-cli`
2. **Azure Functions Core Tools**: `brew install azure-functions-core-tools@4`
3. **Terraform**: `brew install terraform`
4. **Azure account** with appropriate permissions

## Quick Start

### One-command setup (recommended for first time)

```bash
# Create all infrastructure
chmod +x infrastructure/setup.sh
./infrastructure/setup.sh

# Deploy function code
./infrastructure/deploy.sh
```

## Terraform Deployment (manual)

### 1. Initialize infrastructure

```bash
cd infrastructure/terraform
terraform init
```

### 2. Review the plan

```bash
terraform plan
```

### 3. Deploy infrastructure

```bash
terraform apply
```

### 4. Deploy function code

```bash
cd ../..
chmod +x infrastructure/deploy.sh
./infrastructure/deploy.sh
```

## Manual Deployment (Azure Portal + CLI)

If you prefer not to use IaC:

### 1. Create resources via Azure Portal

- Create Resource Group
- Create Storage Account
- Create Function App (Python 3.11, Linux, Consumption plan)

### 2. Deploy code

```bash
# Deploy
func azure functionapp publish <your-function-app-name> --python
```

Or use the deployment script:

```bash
chmod +x infrastructure/deploy.sh
./infrastructure/deploy.sh
```

## Configuration

### Environment Variables

Set in Azure Portal or via Terraform/Bicep:

- `AUTH_TOKEN`: Your API authentication token (change default!)
- `FUNCTIONS_WORKER_RUNTIME`: `python`
- `AzureWebJobsFeatureFlags`: `EnableWorkerIndexing`

### Upgrade to Premium Plan (avoid cold starts)

In Terraform, change:

```hcl
sku_name = "P1v2"  # Instead of "Y1"
```

In Bicep, change:

```bicep
sku: {
  name: 'P1v2'  // Instead of 'Y1'
  tier: 'PremiumV2'
}
```

Then add minimum pre-warmed instances:

```hcl
# Terraform
site_config {
  pre_warmed_instance_count = 1
}
```

## Testing

````bash
# Health check (no auth):

- `AUTH_TOKEN`: Your API authentication token (change default!)
- `FUNCTIONS_WORKER_RUNTIME`: `python`
- `AzureWebJobsFeatureFlags`: `EnableWorkerIndexing`

### Upgrade to Premium Plan (avoid cold starts)

In [main.tf](infrastructure/terraform/main.tf), change:
```hcl
sku_name = "P1v2"  # Instead of "Y1"
````

Then add minimum pre-warmed instances in the `site_config` block:

```hcl
site_config {
  pre_warmed_instance_count = 1
  # ... other settings
```

## Cost Optimization

**Consumption Plan**: ~$0.20/million requests + $0.000016/GB-second

- First 1M requests free per month
- Good for: Low-moderate traffic

**Premium Plan (P1v2)**: ~$140/month + per-execution costs

- No cold starts with pre-warmed instances
- Good for: Production APIs with consistent traffic

## Cleanup

Use the automated cleanup script (recommended):

```bash
chmod +x infrastructure/cleanup.sh
./infrastructure/cleanup.sh
```

Or manually with Terraform:

```bash
cd infrastructure/terraform
terraform destroy
```

Or delete the entire resource group (fastest):

```bash
az group delete --name rg-doomer-habitat
```
