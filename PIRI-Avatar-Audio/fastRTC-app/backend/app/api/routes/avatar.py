"""
API routes for managing avatar interactions
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Optional

from app.core.avatar.viseme_controller import VisemeAnimationController
from app.core.audio.audio_processor import AudioProcessor
from app.models.avatar import VisemeRequest, VisemeSequence, AvatarState, VisemeData


router = APIRouter()

viseme_controller = VisemeAnimationController()
audio_processor = AudioProcessor()


@router.post("/trigger_viseme")
async def trigger_viseme(request: VisemeRequest):
    """Manually trigger a VRM viseme"""
    try:
        viseme_data = VisemeSequence(
            items=[
                VisemeData(
                    viseme=request.phoneme,
                    start_time=0.0,
                    end_time=request.duration,
                    emotion=request.emotion,
                )
            ],
            audio_duration=request.duration
        )
        
        await viseme_controller.add_viseme_sequence(viseme_data.items)
        return {"status": "success", "viseme": request.phoneme, "type": "VRM"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/avatar")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for avatar updates"""
    await viseme_controller.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Process incoming WebSocket messages
            if data.get("type") == "update_viseme":
                viseme = data.get("viseme", "sil")
                duration = data.get("duration", 0.1)
                emotion = data.get("emotion", "neutral")
                
                # Create viseme data
                viseme_data = VisemeSequence(
                    items=[
                        VisemeData(
                            viseme=viseme,
                            start_time=0.0,
                            end_time=duration,
                            emotion=emotion,
                        )
                    ],
                    audio_duration=duration,
                    emotion=emotion,
                )
                
                # Add viseme sequence to controller
                await viseme_controller.add_viseme_sequence(viseme_data.items)
            
    except WebSocketDisconnect:
        viseme_controller.disconnect(websocket)


@router.get("/avatar_status")
async def get_avatar_status() -> AvatarState:
    """Get current avatar status"""
    return viseme_controller.get_current_state()


@router.post("/reset_avatar")
async def reset_avatar():
    """Reset VRM avatar to neutral state"""
    await viseme_controller.reset_to_neutral()
    return {"status": "success", "message": "VRM avatar reset to neutral"}
