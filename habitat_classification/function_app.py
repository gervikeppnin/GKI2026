"""
Azure Functions entry point for habitat classification API.

This wraps the FastAPI app for Azure Functions deployment.
"""

import azure.functions as func
from api import app
import logging

# Create Azure Functions app
azure_app = func.AsgiFunctionApp(app=app, http_auth_level=func.AuthLevel.ANONYMOUS)

logging.info("Habitat Classification Function App initialized")
