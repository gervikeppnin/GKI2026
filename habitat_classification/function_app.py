"""
Azure Functions entry point for habitat classification API.

Uses native HTTP triggers instead of ASGI wrapper.
"""

import azure.functions as func
import json
import logging
from model import predict
from utils import decode_patch

app = func.FunctionApp()

AUTH_TOKEN = "abc123"

def verify_auth(req: func.HttpRequest) -> bool:
    """Check if request has valid Bearer token."""
    auth_header = req.headers.get('Authorization', '')
    return auth_header == f'Bearer {AUTH_TOKEN}'

@app.route(route="ping", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def ping(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint - no auth required."""
    return func.HttpResponse("ok", status_code=200)

@app.route(route="", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def index(req: func.HttpRequest) -> func.HttpResponse:
    """Root endpoint - requires auth."""
    if not verify_auth(req):
        return func.HttpResponse(
            json.dumps({"detail": "Invalid authentication token"}),
            status_code=401,
            mimetype="application/json"
        )
    
    return func.HttpResponse(
        json.dumps({"status": "running", "message": "Habitat Classification API"}),
        mimetype="application/json"
    )

@app.route(route="api", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def api_info(req: func.HttpRequest) -> func.HttpResponse:
    """API information endpoint - requires auth."""
    if not verify_auth(req):
        return func.HttpResponse(
            json.dumps({"detail": "Invalid authentication token"}),
            status_code=401,
            mimetype="application/json"
        )
    
    return func.HttpResponse(
        json.dumps({
            "service": "habitat-classification",
            "version": "1.0.0",
            "endpoints": {
                "/": "Health check",
                "/api": "API information",
                "/predict": "POST - Classify a patch"
            }
        }),
        mimetype="application/json"
    )

@app.route(route="predict", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def predict_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """
    Classify a satellite image patch - requires auth.
    
    Expects JSON body: {"patch": "<base64-encoded-numpy-array>"}
    Returns: {"prediction": <int>}
    """
    if not verify_auth(req):
        return func.HttpResponse(
            json.dumps({"detail": "Invalid authentication token"}),
            status_code=401,
            mimetype="application/json"
        )
    
    try:
        body = req.get_json()
        if not body or 'patch' not in body:
            return func.HttpResponse(
                json.dumps({"detail": "Missing 'patch' field in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Decode base64 to numpy array
        patch = decode_patch(body['patch'])
        
        # Get prediction from model
        prediction = predict(patch)
        
        return func.HttpResponse(
            json.dumps({"prediction": int(prediction)}),
            mimetype="application/json"
        )
    
    except ValueError as e:
        return func.HttpResponse(
            json.dumps({"detail": f"Invalid input: {str(e)}"}),
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"detail": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        )

logging.info("Habitat Classification Function App initialized")
