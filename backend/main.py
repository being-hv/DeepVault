"""
USE CASE: Core API Entry Point
This file initializes the FastAPI application, configures Cross-Origin Resource Sharing (CORS) 
so that our Next.js frontend can securely communicate with it, and hooks up the prediction router.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import predict
import uvicorn

# Initialize the FastAPI app with metadata for auto-generated Swagger documentation
app = FastAPI(
    title="Deepfake Image Detection API",
    description="API for detecting deepfake images using GANs and SGANs.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) configuration
# Allowing all origins is safe for public ML APIs and prevents CORS issues when deploying the frontend to Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)

# Include prediction endpoints under the '/api' prefix
app.include_router(predict.router, prefix="/api")

@app.get("/", tags=["Root"])
def read_root():
    """
    Welcome endpoint returning API metadata and links.
    """
    return {
        "title": "DeepVault Deepfake Image Detection API",
        "status": "healthy",
        "documentation": "/docs",
        "health_check": "/api/health"
    }

@app.get("/api/health", tags=["Health"])
def health_check():
    """
    USE CASE: API Health Check Endpoint
    Allows frontend clients or deployment platforms (like Render/Railway) to verify
    if the backend server is running and responsive.
    """
    return {"status": "healthy", "message": "Deepfake Detection API is running"}

if __name__ == "__main__":
    # Start the server using Uvicorn when this script is run directly
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
