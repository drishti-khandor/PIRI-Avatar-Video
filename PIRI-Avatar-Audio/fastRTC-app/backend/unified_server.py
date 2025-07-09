"""
CORRECTED unified_server.py WITH VRM SUPPORT
This integrates the advanced VRoid viseme system for natural facial animation with VRM files
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import time
import logging
import os
import platform
import socket
from typing import List, Dict
import numpy as np
import threading

# FastRTC and AI imports
from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions, AdditionalOutputs
from fastrtc.utils import audio_to_bytes
from fastrtc import get_stt_model, get_tts_model

# Environment and OpenAI imports
from dotenv import load_dotenv
from openai import AzureOpenAI

# ForceAlign viseme extractor
from ForceAlign_viseme_integration import ForceAlignVisemeExtractor

# IMPORT THE NEW ENHANCED VISEME SYSTEM
from vroid_viseme_integration import EnhancedVRoidVisemeController, enhanced_process_audio_and_respond, detect_emotion_from_text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Platform-specific WebRTC setup
if platform.system() == 'Windows':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    os.environ['WEBRTC_IP'] = local_ip

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Enhanced 3D VRM Avatar + AI Chat Server with Advanced VRoid Visemes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =====================================================
# ENHANCED AVATAR SYSTEM with Advanced VRoid Visemes
# =====================================================

# Initialize the enhanced viseme controller
enhanced_viseme_controller = EnhancedVRoidVisemeController()

# =====================================================
# AI SYSTEM - Enhanced with Advanced Viseme Integration
# =====================================================

# Environment setup for Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# System prompt
sys_prompt = """You are a helpful AI assistant with a 3D VRM avatar. Keep responses concise and natural for speech synthesis. Express emotions appropriately."""
messages = [{"role": "system", "content": sys_prompt}]

# Initialize AI models
if not all([azure_endpoint, api_key, deployment_name]):
    logger.warning("Missing Azure OpenAI environment variables. AI features will be limited.")
    openai_client = None
else:
    openai_client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

stt_model = get_stt_model()
tts_model = get_tts_model(model="kokoro")

# Initialize ForceAlign viseme extractor
try:
    viseme_extractor = ForceAlignVisemeExtractor()
    logger.info("Using ForceAlign viseme extractor")
except Exception as e:
    logger.error(f"Failed to initialize ForceAlign viseme extractor: {e}")
    viseme_extractor = None

async def process_audio_and_respond(audio):
    """Enhanced audio processing with advanced VRoid visemes"""
    async for output in enhanced_process_audio_and_respond(audio, enhanced_viseme_controller):
        yield output

# Initialize FastRTC stream
stream = Stream(ReplyOnPause(
    process_audio_and_respond,
    algo_options=AlgoOptions(
        audio_chunk_duration=0.5,
        started_talking_threshold=0.1,
        speech_threshold=0.03
    ),
    model_options=SileroVadOptions(
        threshold=0.75,
        min_speech_duration_ms=250,
        min_silence_duration_ms=1500,
        speech_pad_ms=400,
        max_speech_duration_s=15
    )),
    modality="audio",
    mode="send-receive",
    concurrency_limit=5
)

stream.mount(app)

class PhonemeItem(BaseModel):
    phoneme: str
    start: float
    end: float

class PhonemeSeq(BaseModel):
    items: List[PhonemeItem]

class VisemeRequest(BaseModel):
    phoneme: str

# Enhanced WebSocket endpoint for VRM avatar
@app.websocket("/ws/avatar")
async def enhanced_vrm_avatar_websocket_endpoint(websocket: WebSocket):
    await enhanced_viseme_controller.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "update_viseme":
                viseme = message.get("viseme", "sil")
                await enhanced_viseme_controller.update_single_viseme(viseme)

    except WebSocketDisconnect:
        enhanced_viseme_controller.disconnect(websocket)

# Enhanced VRM endpoints
@app.post("/trigger_viseme")
async def trigger_vrm_viseme_endpoint(request: VisemeRequest):
    """Manually trigger a VRM viseme"""
    try:
        await enhanced_viseme_controller.update_single_viseme(request.phoneme)
        return {"status": "success", "viseme": request.phoneme, "type": "VRM"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reset_avatar")
async def reset_vrm_avatar_endpoint():
    """Reset VRM avatar to neutral state"""
    try:
        await enhanced_viseme_controller.reset_to_neutral()
        return {"status": "success", "message": "VRM avatar reset to neutral"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/avatar_status")
async def vrm_avatar_status_endpoint():
    """Get current VRM avatar status"""
    status = enhanced_viseme_controller.get_current_state()
    status["type"] = "VRM"
    return status

# Stream updates endpoint for AI chat
@app.get("/updates")
async def stream_updates(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            # DEBUG: Log what we're actually sending
            actual_data = output.args[0]
            logger.info(f"🔍 SSE SENDING: {actual_data}")
            logger.info(f"🔍 TYPE: {type(actual_data)}")
            logger.info(f"🔍 KEYS: {actual_data.keys() if isinstance(actual_data, dict) else 'Not a dict'}")

            yield f"data: {json.dumps(output.args[0])}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


# Enhanced health check endpoint for VRM
@app.get("/health")
async def enhanced_vrm_health_check():
    return {
        "status": "healthy",
        "enhanced_features": True,
        "model_type": "VRM",
        "avatar_connections": len(enhanced_viseme_controller.active_connections),
        "ai_enabled": openai_client is not None,
        "models": {
            "stt": "enabled",
            "tts": "kokoro",
            "viseme_extractor": type(viseme_extractor).__name__,
            "enhanced_vrm_visemes": True
        },
        "avatar_state": enhanced_viseme_controller.get_current_state(),
        "supported_formats": ["VRM", "GLB (fallback)"],
        "vroid_blend_shapes": list(enhanced_viseme_controller.viseme_mapper.vroid_blend_shapes.keys())
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Enhanced 3D VRM Avatar + AI Chat Server with Advanced VRoid Visemes")
    print("📍 Open: http://localhost:8000")
    print("📁 Place your VRM file as: static/avatar.vrm")
    print("🎯 VRM Features:")
    print("   ✅ Enhanced VRM facial animation")
    print("   ✅ Advanced phoneme-to-viseme mapping")
    print("   ✅ Smooth VRM blend shape transitions")
    print("   ✅ VRM emotional expression integration")
    print("   ✅ Real-time lip sync with VRM coarticulation")
    print("   ✅ VRM Expression Manager support")
    print("🔧 Make sure to set up your .env file with Azure OpenAI credentials")
    print("📋 Supported VRM blend shapes:")

    # Print some example VRM blend shapes
    vrm_controller = EnhancedVRoidVisemeController()
    example_shapes = list(vrm_controller.viseme_mapper.vroid_blend_shapes.keys())[:10]
    for shape in example_shapes:
        print(f"   • {shape}")
    if len(vrm_controller.viseme_mapper.vroid_blend_shapes) > 10:
        print(f"   ... and {len(vrm_controller.viseme_mapper.vroid_blend_shapes) - 10} more")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)