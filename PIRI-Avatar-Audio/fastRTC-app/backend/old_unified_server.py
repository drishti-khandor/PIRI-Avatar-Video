"""
Unified Backend Server - Combines 3D Avatar and Real-time AI Chat
Merges fixed_camera_avatar.py and main.py into a single FastAPI server
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

# Viseme extractor
from viseme_extractor import VisemeExtractor, AdvancedVisemeExtractor

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
app = FastAPI(title="Unified 3D Avatar + AI Chat Server")

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
# AVATAR SYSTEM - From fixed_camera_avatar.py
# =====================================================

class VisemeController:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.current_visemes = {
            'A': 0.0, 'E': 0.0, 'I': 0.0, 'O': 0.0, 'U': 0.0, 'M': 0.0, 'REST': 1.0
        }

        # Viseme mapping for avatar
        # self.VISEME_MAP = {
        #     'A': ['aa', 'Fcl_MTH_A'],
        #     'E': ['ee', 'Fcl_MTH_E'],
        #     'I': ['ih', 'Fcl_MTH_I'],
        #     'O': ['oh', 'Fcl_MTH_O'],
        #     'U': ['ou', 'Fcl_MTH_U'],
        #     'M': ['pp', 'Fcl_MTH_Close'],
        #     'REST': ['neutral', 'Fcl_MTH_Neutral']
        # }
        self.VISEME_MAP = {
            'A': ['Fcl_MTH_A'],  # Remove 'aa'
            'E': ['Fcl_MTH_E'],  # Remove 'ee'
            'I': ['Fcl_MTH_I'],  # Remove 'ih'
            'O': ['Fcl_MTH_O'],  # Remove 'oh'
            'U': ['Fcl_MTH_U'],  # Remove 'ou'
            'M': ['Fcl_MTH_Close'],  # Remove 'pp'
            'REST': ['Fcl_MTH_Neutral']  # Remove 'neutral'
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ Avatar client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"❌ Avatar client disconnected. Total: {len(self.active_connections)}")

    async def broadcast_viseme_update(self):
        if not self.active_connections:
            logger.info("❌ No avatar connections to broadcast to")  # ADD THIS
            return

        blend_shapes = {}
        for viseme, weight in self.current_visemes.items():
            blend_shape_names = self.VISEME_MAP.get(viseme, [])
            for blend_name in blend_shape_names:
                blend_shapes[blend_name] = weight

        message = {
            "type": "viseme_update",
            "visemes": self.current_visemes,
            "blend_shapes": blend_shapes,
            "timestamp": time.time()
        }

        logger.info(f"📡 Broadcasting to {len(self.active_connections)} connections: {blend_shapes}")  # ADD THIS

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)

    async def update_viseme(self, viseme: str, value: float):
        if viseme in self.current_visemes:
            self.current_visemes[viseme] = max(0.0, min(1.0, value))
            await self.broadcast_viseme_update()

    async def update_from_ai_visemes(self, ai_visemes: List[Dict]):
        """Convert AI-generated visemes to avatar visemes"""
        # Reset all visemes
        for v in self.current_visemes:
            self.current_visemes[v] = 0.0
        self.current_visemes['REST'] = 1.0

        # Map AI visemes (numbers) to avatar visemes (letters)
        viseme_mapping = {
            '0': 'REST',  # Silence
            '1': 'A',  # Open vowels
            '2': 'A',  # Open back vowels
            '3': 'E',  # Diphthongs
            '4': 'E',  # R-colored vowels
            '5': 'I',  # Close front vowels
            '6': 'U',  # Close back vowels
            '7': 'M',  # Bilabials
            '8': 'E',  # Labiodentals
            '9': 'E',  # Dental fricatives
            '10': 'REST',  # Alveolars
            '11': 'E',  # Sibilants
            '12': 'O',  # Post-alveolars
            '13': 'REST',  # Velars
            '14': 'O',  # Approximants
        }

        # Apply strongest viseme
        if ai_visemes:
            # Get the most recent/dominant viseme
            current_viseme = ai_visemes[0]
            ai_viseme_id = str(current_viseme.get('viseme', '0'))
            avatar_viseme = viseme_mapping.get(ai_viseme_id, 'REST')
            logger.info(f"🎭 Converting AI viseme '{ai_viseme_id}' to avatar viseme '{avatar_viseme}'")  # ADD THIS

            # Reset all and set the current one
            for v in self.current_visemes:
                self.current_visemes[v] = 0.0
            self.current_visemes[avatar_viseme] = 1.0
        logger.info(f"🎨 Final avatar visemes: {self.current_visemes}")  # ADD THIS
        await self.broadcast_viseme_update()


# Global viseme controller
viseme_controller = VisemeController()

# =====================================================
# AI SYSTEM - From main.py
# =====================================================

# Environment setup for Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# System prompt
sys_prompt = """You are a helpful AI assistant with a 3D avatar. Keep responses concise and natural for speech synthesis."""
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

# Initialize viseme extractor
try:
    viseme_extractor = AdvancedVisemeExtractor()
    logger.info("Using advanced viseme extractor")
except:
    viseme_extractor = VisemeExtractor()
    logger.info("Using basic viseme extractor")


def process_audio_and_respond(audio):
    """Main AI processing function - handles STT, LLM, TTS, and visemes"""

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

    # TTS with Viseme Extraction
    logger.info("Starting TTS streaming with viseme extraction.")
    chunk_index = 0
    accumulated_time = 0.0

    try:
        for sample_rate, audio_chunk in tts_model.stream_tts_sync(full_response):
            # Calculate timing for this chunk
            chunk_duration = len(audio_chunk) / sample_rate
            chunk_start_time = accumulated_time
            accumulated_time += chunk_duration

            # Extract visemes from this audio chunk
            try:
                visemes = viseme_extractor.extract_visemes_from_chunk(audio_chunk, sample_rate)

                # Adjust viseme timing to global timeline
                for viseme in visemes:
                    viseme.start_time += chunk_start_time
                    viseme.end_time += chunk_start_time

                # Send viseme data to frontend
                viseme_data = {
                    "type": "visemes",
                    "chunk_index": chunk_index,
                    "visemes": [
                        {
                            "viseme": str(v.viseme),
                            "start_time": float(v.start_time),
                            "end_time": float(v.end_time),
                            "confidence": float(v.confidence)
                        }
                        for v in visemes
                    ],
                    "chunk_duration": float(chunk_duration),
                    "chunk_start_time": float(chunk_start_time)
                }

                logging.info(f"Sent visemes for chunk {chunk_index}: {[(v.viseme, v.start_time, v.end_time, v.confidence) for v in visemes]}")
                yield AdditionalOutputs(viseme_data)
                logging.info(f"Sent visemes for chunk {chunk_index}: {[v.viseme for v in visemes]}")
                # ADD THIS LINE:
                # await viseme_controller.update_from_ai_visemes(viseme_data["visemes"])
                # try:
                #     loop = asyncio.get_event_loop()
                #     if loop.is_running():
                #         # If loop is running, schedule the coroutine
                #         asyncio.run_coroutine_threadsafe(
                #             viseme_controller.update_from_ai_visemes(viseme_data["visemes"]),
                #             loop
                #         )
                #     else:
                #         # If no loop running, run it directly
                #         loop.run_until_complete(
                #             viseme_controller.update_from_ai_visemes(viseme_data["visemes"])
                #         )
                # except Exception as e:
                #     logging.error(f"Failed to update avatar visemes: {e}")

                def update_avatar_async():
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            viseme_controller.update_from_ai_visemes(viseme_data["visemes"])
                        )
                    finally:
                        loop.close()

                # Start in background thread
                threading.Thread(target=update_avatar_async, daemon=True).start()

                yield sample_rate, audio_chunk

                chunk_index += 1

            except Exception as e:
                logging.error(f"Viseme extraction failed for chunk {chunk_index}: {e}")
                # Send default silence viseme
                yield AdditionalOutputs({
                    "type": "visemes",
                    "chunk_index": chunk_index,
                    "visemes": [{"viseme": "0", "start_time": chunk_start_time, "end_time": accumulated_time,
                                 "confidence": 1.0}],
                    "chunk_duration": chunk_duration,
                    "chunk_start_time": chunk_start_time
                })

            # Yield the audio chunk for playback
            # yield sample_rate, audio_chunk
            #
            # chunk_index += 1

        logging.info("Finished TTS streaming with visemes.")

    except Exception as e:
        logging.error(f"TTS failed: {e}")

    messages.append({"role": "assistant", "content": full_response + " "})


    # try:
    #     for audio_chunk in tts_model.stream_tts_sync(full_response):
    #         yield audio_chunk
    #         # Calculate timing for this chunk
    #     #     chunk_duration = len(audio_chunk) / sample_rate
    #     #     chunk_start_time = accumulated_time
    #     #     accumulated_time += chunk_duration
    #     #
    #     #     # Extract visemes from this audio chunk
    #     #     try:
    #     #         visemes = viseme_extractor.extract_visemes_from_chunk(audio_chunk, sample_rate)
    #     #
    #     #         # Adjust viseme timing to global timeline
    #     #         for viseme in visemes:
    #     #             viseme.start_time += chunk_start_time
    #     #             viseme.end_time += chunk_start_time
    #     #
    #     #         # Send viseme data to frontend AND update avatar
    #     #         viseme_data = {
    #     #             "type": "visemes",
    #     #             "chunk_index": chunk_index,
    #     #             "visemes": [
    #     #                 {
    #     #                     "viseme": str(v.viseme),
    #     #                     "start_time": float(v.start_time),
    #     #                     "end_time": float(v.end_time),
    #     #                     "confidence": float(v.confidence)
    #     #                 }
    #     #                 for v in visemes
    #     #             ],
    #     #             "chunk_duration": float(chunk_duration),
    #     #             "chunk_start_time": float(chunk_start_time)
    #     #         }
    #     #
    #     #         # Update avatar in real-time
    #     #         asyncio.create_task(
    #     #             viseme_controller.update_from_ai_visemes(viseme_data["visemes"])
    #     #         )
    #     #
    #     #         logger.info(f"Sent visemes for chunk {chunk_index}: {[v.viseme for v in visemes]}")
    #     #         yield AdditionalOutputs(viseme_data)
    #     #
    #     #     except Exception as e:
    #     #         logger.error(f"Viseme extraction failed for chunk {chunk_index}: {e}")
    #     #         # Send default silence viseme
    #     #         yield AdditionalOutputs({
    #     #             "type": "visemes",
    #     #             "chunk_index": chunk_index,
    #     #             "visemes": [{"viseme": "0", "start_time": chunk_start_time, "end_time": accumulated_time,
    #     #                          "confidence": 1.0}],
    #     #             "chunk_duration": chunk_duration,
    #     #             "chunk_start_time": chunk_start_time
    #     #         })
    #     #
    #     #     # Yield the audio chunk for playback
    #     #     yield sample_rate, audio_chunk
    #     #     chunk_index += 1
    #     #
    #     logger.info("Finished TTS streaming with visemes.")
    #
    # except Exception as e:
    #     logger.error(f"TTS failed: {e}")
    #
    # messages.append({"role": "assistant", "content": full_response + " "})


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

# Mount FastRTC stream to app
stream.mount(app)


# =====================================================
# PYDANTIC MODELS
# =====================================================

class PhonemeItem(BaseModel):
    phoneme: str
    start: float
    end: float


class PhonemeSeq(BaseModel):
    items: List[PhonemeItem]


# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def get_unified_interface():
    """Unified HTML interface combining avatar and chat"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Unified 3D Avatar + AI Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
            height: 100vh;
        }

        .main-container {
            display: flex;
            height: 100vh;
            gap: 2px;
        }

        /* Avatar Panel */
        .avatar-panel {
            flex: 1;
            background: #1a1a1a;
            border-radius: 10px 0 0 10px;
            position: relative;
            overflow: hidden;
        }

        .avatar-header {
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            text-align: center;
            border-bottom: 2px solid #667eea;
        }

        .avatar-canvas-container {
            flex: 1;
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            height: calc(100vh - 80px);
        }

        #avatarCanvas {
            width: 100%;
            height: 100%;
            max-width: 100%;
            max-height: 100%;
        }

        /* AI Chat Panel */
        .ai-panel {
            flex: 1;
            background: #2d2d2d;
            border-radius: 0 10px 10px 0;
            display: flex;
            flex-direction: column;
        }

        .ai-header {
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            text-align: center;
            border-bottom: 2px solid #764ba2;
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 15px;
        }

        .messages-area {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            overflow-y: auto;
            backdrop-filter: blur(10px);
        }

        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            max-width: 85%;
        }

        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
            text-align: right;
        }

        .message.ai {
            background: rgba(255, 255, 255, 0.9);
            color: #333;
        }

        .voice-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }

        .voice-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            background: #667eea;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            transition: all 0.3s ease;
        }

        .voice-btn:hover {
            background: #764ba2;
            transform: scale(1.1);
        }

        .voice-btn.recording {
            background: #ff4757;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        .status-display {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
        }

        .status-dot.disconnected {
            background: #ef4444;
        }

        .viseme-debug {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 8px;
            font-size: 12px;
            min-width: 100px;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
            }

            .avatar-panel, .ai-panel {
                border-radius: 0;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Avatar Panel -->
        <div class="avatar-panel">
            <div class="avatar-header">
                <h2>🤖 3D Avatar</h2>
                <p>Real-time Facial Animation</p>
            </div>

            <div class="avatar-canvas-container">
                <canvas id="avatarCanvas"></canvas>

                <!-- Viseme Debug Display -->
                <div class="viseme-debug" id="visemeDebug">
                    <div>Current Viseme:</div>
                    <div id="currentViseme">REST</div>
                </div>
            </div>
        </div>

        <!-- AI Chat Panel -->
        <div class="ai-panel">
            <div class="ai-header">
                <h2>🧠 AI Assistant</h2>
                <p>Real-time Speech & Chat</p>
            </div>

            <div class="chat-container">
                <div class="messages-area" id="messagesArea">
                    <div class="message ai">
                        <strong>AI:</strong> Hello! I'm ready to chat. Press and hold the microphone button to speak with me. I'll respond with both voice and synchronized facial animations.
                    </div>
                </div>

                <div class="voice-controls">
                    <button class="voice-btn" id="voiceBtn">🎤</button>
                    <div class="status-display">
                        <div class="status-item">
                            <span>Avatar:</span>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div class="status-dot" id="avatarStatus"></div>
                                <span>Ready</span>
                            </div>
                        </div>
                        <div class="status-item">
                            <span>AI:</span>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div class="status-dot" id="aiStatus"></div>
                                <span>Ready</span>
                            </div>
                        </div>
                        <div class="status-item">
                            <span>Audio:</span>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div class="status-dot" id="audioStatus"></div>
                                <span>Ready</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

        // Global variables
        let scene, camera, renderer, gltfLoader;
        let avatar = null;
        let avatarWebSocket = null;
        let webrtcClient = null;
        let isRecording = false;
        let webrtcId = Math.random().toString(36).substring(7);

        const BLEND_SHAPE_MAP = {
            'A': ['aa', 'Fcl_MTH_A'],
            'E': ['ee', 'Fcl_MTH_E'], 
            'I': ['ih', 'Fcl_MTH_I'],
            'O': ['oh', 'Fcl_MTH_O'],
            'U': ['ou', 'Fcl_MTH_U'],
            'M': ['pp', 'Fcl_MTH_Close'],
            'REST': ['neutral', 'Fcl_MTH_Neutral']
        };

        // Initialize Avatar WebSocket
        async function initAvatarWebSocket() {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                avatarWebSocket = new WebSocket(`${protocol}//${window.location.host}/ws/avatar`);

                avatarWebSocket.onopen = () => {
                    updateStatus('avatarStatus', true);
                    console.log('✅ Avatar WebSocket connected');
                };

                avatarWebSocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'viseme_update') {
                        applyBlendShapes(data.blend_shapes);
                        updateVisemeDebug(data.visemes);
                    }
                };

                avatarWebSocket.onclose = () => {
                    updateStatus('avatarStatus', false);
                    setTimeout(initAvatarWebSocket, 3000);
                };

            } catch (error) {
                console.error('Avatar WebSocket error:', error);
            }
        }

        // Initialize Three.js for Avatar
        async function initThreeJS() {
            try {
                const viewport = document.getElementById('avatarCanvas').parentElement;
                const aspect = viewport.clientWidth / viewport.clientHeight || 1;

                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);

                # camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 1000);
                # camera.position.set(0, 0, 1);
                # camera.lookAt(0, 0, 0);
                
                // FIXED CAMERA - Uses your Blender setup
                camera = new THREE.PerspectiveCamera(5, aspect, 0.1, 1000);
                // Don't override camera position - use default/Blender position
                camera.position.set(0, 0, 35); // Minimal distance for face focus
                camera.lookAt(0, 0, 5); // Look at center

                renderer = new THREE.WebGLRenderer({ 
                    canvas: document.getElementById('avatarCanvas'),
                    antialias: true,
                    preserveDrawingBuffer: true,
                    powerPreference: "high-performance"
                });

                renderer.physicallyCorrectLights = true;
                renderer.toneMapping = THREE.ACESFilmicToneMapping;
                renderer.toneMappingExposure = 1.0;
                renderer.setSize(viewport.clientWidth, viewport.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.outputColorSpace = THREE.SRGBColorSpace;

                // Lighting
                const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
                scene.add(hemi);

                const key = new THREE.DirectionalLight(0xffffff, 1.2);
                key.position.set(0, 1, 2);
                scene.add(key);

                const rim = new THREE.DirectionalLight(0xffffff, 0.6);
                rim.position.set(0, 1, -2);
                scene.add(rim);

                gltfLoader = new GLTFLoader();

                window.addEventListener('resize', onWindowResize);
                onWindowResize();

                console.log('✅ Three.js initialized');
                return true;

            } catch (error) {
                console.error('Three.js error:', error);
                return false;
            }
        }

        // Load Avatar GLB
        async function loadAvatar() {
            // const avatarPaths = ['/static/test5.glb', '/static/avatar.glb'];
            # const avatarPaths = ['/static/joined1111.glb'];
            const avatarPaths = ['/static/joined2.glb'];
            # const avatarPaths = ['/static/fixedaf.glb'];

            for (const path of avatarPaths) {
                try {
                    console.log(`Loading: ${path}`);

                    const gltf = await new Promise((resolve, reject) => {
                        gltfLoader.load(path, resolve, undefined, reject);
                    });

                    if (avatar) scene.remove(avatar);

                    avatar = gltf.scene;
                    scene.add(avatar);

                    // Find and log morph targets
                    avatar.traverse((child) => {
                        if (child.isMesh && child.morphTargetInfluences) {
                            child.userData.morphTargets = child.morphTargetDictionary;
                            console.log('🎭 Morph targets found:', Object.keys(child.morphTargetDictionary || {}));
                        }
                    });

                    console.log(`✅ Avatar loaded: ${path}`);
                    return true;

                } catch (error) {
                    console.log(`Failed: ${path}`);
                }
            }

            console.log('❌ No avatar found');
            return false;
        }

        // Apply blend shapes to avatar
        function applyBlendShapes(blendShapeValues) {
            if (!avatar) return;

            avatar.traverse((child) => {
                if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
                    // Reset all
                    for (let i = 0; i < child.morphTargetInfluences.length; i++) {
                        child.morphTargetInfluences[i] = 0;
                    }

                    // Apply new values
                    for (const [shapeName, value] of Object.entries(blendShapeValues)) {
                        const index = child.morphTargetDictionary[shapeName];
                        if (index !== undefined) {
                            child.morphTargetInfluences[index] = value;
                        }
                    }
                }
            });
        }

        // Update viseme debug display
        function updateVisemeDebug(visemes) {
            const currentViseme = Object.entries(visemes)
                .find(([key, value]) => value > 0.5)?.[0] || 'REST';
            document.getElementById('currentViseme').textContent = currentViseme;
        }

        // WebRTC Client for AI Chat
        class SimpleWebRTCClient {
            constructor() {
                this.peerConnection = null;
                this.mediaStream = null;
                this.dataChannel = null;
            }

            async connect() {
                try {
                    this.peerConnection = new RTCPeerConnection();

                    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

                    this.mediaStream.getTracks().forEach(track => {
                        this.peerConnection.addTrack(track, this.mediaStream);
                    });

                    this.peerConnection.addEventListener('track', (event) => {
                        const audio = new Audio();
                        audio.srcObject = event.streams[0];
                        audio.autoplay = true;
                    });

                    this.dataChannel = this.peerConnection.createDataChannel('text');

                    // Create and send offer
                    const offer = await this.peerConnection.createOffer();
                    await this.peerConnection.setLocalDescription(offer);

                    const response = await fetch('/webrtc/offer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            sdp: offer.sdp,
                            type: offer.type,
                            webrtc_id: webrtcId
                        })
                    });

                    const serverResponse = await response.json();
                    await this.peerConnection.setRemoteDescription(serverResponse);

                    updateStatus('aiStatus', true);
                    updateStatus('audioStatus', true);

                } catch (error) {
                    console.error('WebRTC connection error:', error);
                    this.disconnect();
                    throw error;
                }
            }

            disconnect() {
                if (this.mediaStream) {
                    this.mediaStream.getTracks().forEach(track => track.stop());
                    this.mediaStream = null;
                }

                if (this.peerConnection) {
                    this.peerConnection.close();
                    this.peerConnection = null;
                }

                this.dataChannel = null;
                updateStatus('aiStatus', false);
                updateStatus('audioStatus', false);
            }
        }

        // Initialize SSE for AI messages
        function initSSE() {
            const eventSource = new EventSource(`/updates?webrtc_id=${webrtcId}`);

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === "stt" || data.type === "llm") {
                        addMessage(data.type === "stt" ? "user" : "ai", data.text);
                    } else if (data.type === "visemes") {
                        console.log("Received visemes from AI:", data);
                        // Visemes are automatically sent to avatar via the backend
                    }
                } catch (err) {
                    console.error("SSE parse error", err);
                }
            };

            eventSource.onerror = (err) => {
                console.error("SSE error", err);
            };

            return eventSource;
        }

        // UI Helper Functions
        function addMessage(sender, text) {
            const messagesArea = document.getElementById('messagesArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'AI'}:</strong> ${text}`;
            messagesArea.appendChild(messageDiv);
            messagesArea.scrollTop = messagesArea.scrollHeight;
        }

        function updateStatus(statusId, isConnected) {
            const statusDot = document.getElementById(statusId);
            if (isConnected) {
                statusDot.classList.remove('disconnected');
            } else {
                statusDot.classList.add('disconnected');
            }
        }

        function onWindowResize() {
            const viewport = document.getElementById('avatarCanvas').parentElement;
            camera.aspect = viewport.clientWidth / viewport.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewport.clientWidth, viewport.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }

        // Voice Button Handling
        document.getElementById('voiceBtn').addEventListener('mousedown', async () => {
            if (!isRecording) {
                try {
                    if (!webrtcClient) {
                        webrtcClient = new SimpleWebRTCClient();
                    }
                    logger.debug({webrtcClient}, "Creating WebRTC client");
                    await webrtcClient.connect();
                    isRecording = true;
                    document.getElementById('voiceBtn').classList.add('recording');
                    document.getElementById('voiceBtn').textContent = '🔴';
                } catch (error) {
                    console.error('Failed to start recording:', error);
                    addMessage('ai', 'Failed to start recording. Please check your microphone permissions.');
                }
            }
        });

        document.getElementById('voiceBtn').addEventListener('mouseup', () => {
            if (isRecording && webrtcClient) {
                webrtcClient.disconnect();
                isRecording = false;
                document.getElementById('voiceBtn').classList.remove('recording');
                document.getElementById('voiceBtn').textContent = '🎤';
            }
        });

        // Prevent context menu on voice button
        document.getElementById('voiceBtn').addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });

        // Initialize the application
        async function init() {
            console.log('🚀 Starting Unified Avatar + AI Chat...');

            // Initialize avatar system
            await initAvatarWebSocket();
            const threeReady = await initThreeJS();

            if (threeReady) {
                await loadAvatar();
                animate();
            }

            // Initialize AI chat system
            initSSE();

            console.log('✅ Application initialized');
        }

        // Start the application
        window.addEventListener('load', init);
    </script>
</body>
</html>
    '''

@app.websocket("/ws")
async def fastrtc_websocket_endpoint(websocket: WebSocket):
    """FastRTC WebSocket endpoint - handles WebRTC signaling"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # FastRTC handles its own message processing
            # This endpoint just needs to accept connections
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"FastRTC WebSocket error: {e}")

# Avatar WebSocket endpoint
@app.websocket("/ws/avatar")
async def avatar_websocket_endpoint(websocket: WebSocket):
    await viseme_controller.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "update_viseme":
                viseme = message.get("viseme")
                value = message.get("value", 0.0)
                await viseme_controller.update_viseme(viseme, value)

    except WebSocketDisconnect:
        viseme_controller.disconnect(websocket)


# Phoneme playback endpoint (from original avatar app)
@app.post("/play_phonemes")
async def play_phonemes_endpoint(seq: PhonemeSeq):
    """Play phoneme sequences on the avatar"""
    # Convert phonemes to viseme format for avatar
    # This is for direct phoneme control if needed
    aligned = [(i.phoneme, i.start, i.end) for i in seq.items]

    # You can implement phoneme-to-viseme conversion here
    # For now, we'll use the AI-generated visemes

    return {"status": "started", "frames": len(aligned)}


# Reset chat endpoint
@app.get("/reset")
async def reset():
    global messages
    logger.info("Resetting chat")
    messages = [{"role": "system", "content": sys_prompt}]
    return {"status": "success"}


# Stream updates endpoint for AI chat
@app.get("/updates")
async def stream_updates(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            yield f"data: {json.dumps(output.args[0])}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


# Viseme info endpoint
@app.get("/viseme-info")
async def get_viseme_info():
    """Return viseme mapping information for frontend"""
    return {
        "viseme_mapping": viseme_extractor.phoneme_to_viseme,
        "avatar_visemes": list(viseme_controller.current_visemes.keys()),
        "viseme_descriptions": {
            "0": "Silence",
            "1": "Open vowels (AE, AH, EH, UH)",
            "2": "Open back vowels (AA, AO, AW, OW)",
            "3": "Diphthongs (AY, EY, OY)",
            "4": "R-colored vowels (ER, AX, IX)",
            "5": "Close front vowels (IH, IY)",
            "6": "Close back vowels (UW, UH)",
            "7": "Bilabials (B, P, M)",
            "8": "Labiodentals (F, V)",
            "9": "Dental fricatives (TH, DH)",
            "10": "Alveolars (T, D, N, L)",
            "11": "Sibilants (S, Z)",
            "12": "Post-alveolars (SH, ZH, CH, JH)",
            "13": "Velars (K, G, NG)",
            "14": "Approximants (R, W, Y, HH)"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "avatar_connections": len(viseme_controller.active_connections),
        "ai_enabled": openai_client is not None,
        "models": {
            "stt": "enabled",
            "tts": "kokoro",
            "viseme_extractor": type(viseme_extractor).__name__
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Unified 3D Avatar + AI Chat Server")
    print("📍 Open: http://localhost:8000")
    print("📁 Place your GLB as: static/test5.glb or static/avatar.glb")
    print("🎯 Features: 3D Avatar + Real-time AI Chat + Synchronized Visemes")
    print("🔧 Make sure to set up your .env file with Azure OpenAI credentials")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)