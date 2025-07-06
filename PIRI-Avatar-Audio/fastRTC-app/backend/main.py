"""
Clean Main Server with OVRLipsync Integration
Streamlined FastAPI server for real-time lip-sync avatar
"""

import fastapi
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import logging
import time
import os
import platform
import socket
import json
import threading

# FastRTC and AI imports
from fastrtc import ReplyOnPause, Stream, AlgoOptions, SileroVadOptions, AdditionalOutputs
from fastrtc import get_stt_model, get_tts_model

# Environment and OpenAI imports
from dotenv import load_dotenv
from openai import AzureOpenAI

# Our clean components
from vrm_controller import VRMAvatarController

# Setup
load_dotenv()
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

# Initialize FastAPI
app = fastapi.FastAPI(title="Clean Lip-Sync Avatar Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global components
vrm_controller = VRMAvatarController()

# AI Setup
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

sys_prompt = "You are a helpful AI assistant with a 3D avatar. Keep responses concise and natural for speech synthesis."
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

def process_audio_and_respond(audio):
    """
    Clean audio processing pipeline with OVRLipsync
    """
    # Speech-to-Text
    stt_time = time.time()
    logger.info("Performing STT")
    text = stt_model.stt(audio)
    if not text:
        logger.info("STT returned empty string")
        return

    logger.info(f"STT response: {text}")
    yield AdditionalOutputs({"type": "stt", "text": text})

    messages.append({"role": "user", "content": text})
    logger.info(f"STT took {time.time() - stt_time} seconds")

    # LLM Generation
    llm_time = time.time()
    try:
        if openai_client:
            response = openai_client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            full_response = response.choices[0].message.content
        else:
            full_response = "AI service not configured. Please check your environment variables."
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        full_response = "I'm having trouble processing that right now."

    logger.info(f"LLM response: {full_response}")
    logger.info(f"LLM took {time.time() - llm_time} seconds")
    yield AdditionalOutputs({"type": "llm", "text": full_response})

    # TTS with VRM Lip-Sync
    logger.info("Starting TTS with VRM lip-sync")
    chunk_index = 0

    try:
        for sample_rate, audio_chunk in tts_model.stream_tts_sync(full_response):
            # Process audio for VRM lip-sync
            def update_vrm_async():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        vrm_controller.process_audio_chunk(audio_chunk, sample_rate)
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"VRM update failed: {e}")

            # Start VRM update in background
            threading.Thread(target=update_vrm_async, daemon=True).start()

            # Yield audio for playback
            yield sample_rate, audio_chunk
            chunk_index += 1

        logger.info("Finished TTS with VRM lip-sync")

    except Exception as e:
        logger.error(f"TTS failed: {e}")

    messages.append({"role": "assistant", "content": full_response + " "})

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

# Mount FastRTC stream
stream.mount(app)

# Pydantic models
class EmotionRequest(BaseModel):
    emotion: str

class SmoothingRequest(BaseModel):
    factor: float

# API Endpoints

@app.websocket("/ws/avatar")
async def vrm_websocket_endpoint(websocket: WebSocket):
    """VRM Avatar WebSocket endpoint"""
    await vrm_controller.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
    except WebSocketDisconnect:
        vrm_controller.disconnect(websocket)

@app.post("/set_emotion")
async def set_emotion_endpoint(request: EmotionRequest):
    """Set VRM avatar emotion"""
    try:
        await vrm_controller.set_emotion(request.emotion)
        return {"status": "success", "emotion": request.emotion}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/set_smoothing")
async def set_smoothing_endpoint(request: SmoothingRequest):
    """Adjust animation smoothing factor"""
    try:
        vrm_controller.set_smoothing_factor(request.factor)
        return {"status": "success", "smoothing_factor": request.factor}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/reset_avatar")
async def reset_avatar_endpoint():
    """Reset VRM avatar to neutral state"""
    try:
        await vrm_controller.reset_to_neutral()
        return {"status": "success", "message": "Avatar reset to neutral"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/avatar_status")
async def avatar_status_endpoint():
    """Get current VRM avatar status"""
    return vrm_controller.get_status()

@app.get("/reset")
async def reset_chat():
    """Reset chat conversation"""
    global messages
    logger.info("Resetting chat")
    messages = [{"role": "system", "content": sys_prompt}]
    return {"status": "success"}

@app.get("/updates")
async def stream_updates(webrtc_id: str):
    """Stream updates for AI chat"""
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            yield f"data: {json.dumps(output.args[0])}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vrm_controller": "active",
        "avatar_connections": len(vrm_controller.active_connections),
        "ai_enabled": openai_client is not None,
        "models": {
            "stt": "enabled",
            "tts": "kokoro",
            "lipsync": "OVRLipsync"
        }
    }

if __name__ == "__main__":
    import uvicorn

    print("🚀 Clean Lip-Sync Avatar Server")
    print("📍 Open: http://localhost:8000")
    print("📁 Place your VRM file as: static/4thjuly.vrm")
    print("🎯 Features:")
    print("   ✅ OVRLipsync integration")
    print("   ✅ Smooth animation with EMA")
    print("   ✅ Real-time VRM lip-sync")
    print("   ✅ Clean architecture")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)