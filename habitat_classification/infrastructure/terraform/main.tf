terraform {
  required_version = ">= 1.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Variables
variable "project_name" {
  description = "Project name"
  type        = string
  default     = "doomer-habitat"
}

variable "team_name" {
  description = "Team name for resource tagging"
  type        = string
  default     = "doomers"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "westeurope"
}

variable "auth_token" {
  description = "API authentication token"
  type        = string
  sensitive   = true
  default     = "ItAllOver"
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project_name}"
  location = var.location
  
  tags = {
    Team      = var.team_name
    Project   = var.project_name
    ManagedBy = "Terraform"
  }
}

# Storage Account (required for Azure Functions)
resource "azurerm_storage_account" "function_storage" {
  name                     = "st${replace(var.project_name, "-", "")}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  tags = azurerm_resource_group.main.tags
}

# App Service Plan (Consumption plan for serverless)
resource "azurerm_service_plan" "function_plan" {
  name                = "asp-${var.project_name}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "Y1"  # Y1 = Consumption (pay-per-execution)
  
  tags = azurerm_resource_group.main.tags
}

# Application Insights (monitoring)
resource "azurerm_application_insights" "function_insights" {
  name                = "appi-${var.project_name}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  application_type    = "web"
  
  tags = azurerm_resource_group.main.tags
}

# Function App
resource "azurerm_linux_function_app" "main" {
  name                       = "func-${var.project_name}"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  service_plan_id            = azurerm_service_plan.function_plan.id
  storage_account_name       = azurerm_storage_account.function_storage.name
  storage_account_access_key = azurerm_storage_account.function_storage.primary_access_key

  site_config {
    application_stack {
      python_version = "3.11"  # Azure Functions supports 3.9, 3.10, 3.11
    }
    
    # Enable CORS if needed
    cors {
      allowed_origins = ["*"]  # Restrict this in production
    }
  }

  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME"       = "python"
    "AzureWebJobsFeatureFlags"       = "EnableWorkerIndexing"
    "APPINSIGHTS_INSTRUMENTATIONKEY" = azurerm_application_insights.function_insights.instrumentation_key
    "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.function_insights.connection_string
    
    # Custom app settings
    "AUTH_TOKEN" = var.auth_token
  }

  # Enable application insights
  https_only = true
  
  tags = azurerm_resource_group.main.tags
}

# Outputs
output "function_app_name" {
  value = azurerm_linux_function_app.main.name
}

output "function_app_url" {
  value = "https://${azurerm_linux_function_app.main.default_hostname}"
}

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}
