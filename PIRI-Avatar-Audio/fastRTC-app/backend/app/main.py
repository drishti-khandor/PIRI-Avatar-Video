"""
Main entry point for the FastAPI application
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config.settings import settings, setup_platform_specific
from app.api.routes import router as api_router


# Configure platform-specific settings
setup_platform_specific()


# Initialize FastAPI
app = FastAPI(title=settings.app_name, version=settings.app_version)


# Apply CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Static files
os.makedirs(settings.static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


# Include API routes
app.include_router(api_router)

# Mount FastRTC stream
from app.api.routes.webrtc import stream
stream.mount(app)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.api.routes.avatar import viseme_controller
    
    return {
        "status": "healthy",
        "server": {
            "name": settings.app_name,
            "version": settings.app_version,
            "port": settings.port
        },
        "features": {
            "vrm_support": True,
            "forcealign_visemes": True,
            "emotion_detection": True,
            "smooth_transitions": True
        },
        "avatar": {
            "connected_clients": viseme_controller.current_state.connected_clients,
            "is_animating": viseme_controller.is_animating
        },
        "ai": {
            "azure_openai_configured": bool(settings.azure_openai_endpoint)
        }
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
