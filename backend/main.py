"""
MHEART FastAPI Application
Main entry point for the backend API.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.api.routes import router
from backend.core.config import settings


# Create FastAPI app
app = FastAPI(
    title="MHEART API",
    description="Mental Health Emotion Analysis & Response Terminal",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(router, prefix="/api", tags=["MHEART"])


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serve basic info"""
    return """
    <html>
        <head><title>MHEART API</title></head>
        <body>
            <h1>MHEART - Mental Health Emotion Analysis & Response Terminal</h1>
            <p>Version: 1.0.0</p>
            <p>API Documentation: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """


# Health check at root
@app.get("/health")
async def root_health():
    """Root health check"""
    return {"status": "healthy", "service": "MHEART API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
