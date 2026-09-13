"""
Main FastAPI Application
Egg Yolk Color Predictor Backend Service
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import predict

app = FastAPI(
    title="Egg Yolk Color Predictor API",
    description="Backend API for predicting Egg Yolk Color Fan score (1-15) from cropped images.",
    version="1.0.0"
)

# Enable CORS Middleware to allow requests from Flutter Mobile App / Web Clients
# NOTE: allow_credentials=False is required when allow_origins=["*"].
# The CORS spec forbids wildcard origin with credentials=True.
# This API uses no cookies or session auth, so credentials are not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(predict.router)


@app.get("/", tags=["Health Check"])
async def root():
    """
    Health check endpoint. Returns ok status.
    """
    return {"status": "ok", "service": "Egg Yolk Color Predictor API"}


@app.get("/web", tags=["Web Interface"])
async def web_ui():
    """
    Serve Web App UI interface.
    """
    web_app_path = os.path.join(os.path.dirname(__file__), "web_app.html")
    return FileResponse(web_app_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


