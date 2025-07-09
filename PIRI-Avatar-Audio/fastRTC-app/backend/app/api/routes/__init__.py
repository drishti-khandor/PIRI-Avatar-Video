"""
Initialize API routes
"""
from fastapi import APIRouter
from app.api.routes import avatar, webrtc


router = APIRouter()

# Include individual route modules
router.include_router(avatar.router, prefix="/api/avatar", tags=["Avatar"])
router.include_router(webrtc.router, tags=["WebRTC"])
