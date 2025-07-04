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

# Viseme extractor
from viseme_extractor import VisemeExtractor, AdvancedVisemeExtractor

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

# Initialize viseme extractor
try:
    viseme_extractor = AdvancedVisemeExtractor()
    logger.info("Using advanced viseme extractor")
except:
    viseme_extractor = VisemeExtractor()
    logger.info("Using basic viseme extractor")

def process_audio_and_respond(audio):
    """Enhanced audio processing with advanced VRoid visemes"""
    return enhanced_process_audio_and_respond(audio, enhanced_viseme_controller)

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

class EmotionRequest(BaseModel):
    emotion: str

class VisemeRequest(BaseModel):
    phoneme: str
    emotion: str = "neutral"

# =====================================================
# ENHANCED API ENDPOINTS
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def get_unified_interface():
    """Enhanced unified interface with VRM avatar support"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced 3D VRM Avatar + AI Chat with Advanced Visemes</title>
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
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            min-width: 200px;
        }

        .emotion-controls {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 8px;
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }

        .emotion-btn {
            padding: 5px 10px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 10px;
            transition: all 0.3s ease;
        }

        .emotion-btn:hover {
            background: #764ba2;
        }

        .emotion-btn.active {
            background: #4ade80;
        }

        .vrm-status {
            position: absolute;
            bottom: 80px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 8px;
            font-size: 10px;
            color: #4ade80;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- VRM Avatar Panel -->
        <div class="avatar-panel">
            <div class="avatar-header">
                <h2>🤖 Enhanced VRM Avatar</h2>
                <p>Advanced VRoid Facial Animation with Natural Lip Sync</p>
            </div>

            <div class="avatar-canvas-container">
                <canvas id="avatarCanvas"></canvas>

                <!-- Enhanced Viseme Debug Display -->
                <div class="viseme-debug" id="visemeDebug">
                    <div><strong>VRM Avatar State:</strong></div>
                    <div>Emotion: <span id="currentEmotion">neutral</span></div>
                    <div>Blend Shapes: <span id="activeBlendShapes">0</span></div>
                    <div>Status: <span id="avatarConnectionStatus">Connecting...</span></div>
                    <div>Dominant: <span id="dominantViseme">sil</span></div>
                </div>

                <!-- VRM Status Indicator -->
                <div class="vrm-status" id="vrmStatus">
                    🎭 VRM Model Ready
                </div>

                <!-- Emotion Controls -->
                <div class="emotion-controls">
                    <div style="color: white; font-size: 10px; width: 100%; text-align: center; margin-bottom: 5px;">VRM Emotions:</div>
                    <button class="emotion-btn active" onclick="setEmotion('neutral')">😐 Neutral</button>
                    <button class="emotion-btn" onclick="setEmotion('happy')">😊 Happy</button>
                    <button class="emotion-btn" onclick="setEmotion('sad')">😢 Sad</button>
                    <button class="emotion-btn" onclick="setEmotion('surprised')">😲 Surprised</button>
                    <button class="emotion-btn" onclick="setEmotion('angry')">😠 Angry</button>
                </div>
            </div>
        </div>

        <!-- AI Chat Panel -->
        <div class="ai-panel">
            <div class="ai-header">
                <h2>🧠 AI Assistant</h2>
                <p>Real-time Speech & Chat with VRM Emotional Expression</p>
            </div>

            <div class="chat-container">
                <div class="messages-area" id="messagesArea">
                    <div class="message ai">
                        <strong>AI:</strong> Hello! I'm ready to chat with enhanced VRM facial expressions and natural lip sync. Place your VRM file as /static/avatar.vrm to get started!
                    </div>
                </div>

                <div class="voice-controls">
                    <button class="voice-btn" id="voiceBtn">🎤</button>
                    <div class="status-display">
                        <div class="status-item">
                            <span>Enhanced VRM Avatar:</span>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div class="status-dot" id="avatarStatus"></div>
                                <span>Ready</span>
                            </div>
                        </div>
                        <div class="status-item">
                            <span>AI + VRM Emotions:</span>
                            <div style="display: flex; align-items: center; gap: 5px;">
                                <div class="status-dot" id="aiStatus"></div>
                                <span>Ready</span>
                            </div>
                        </div>
                        <div class="status-item">
                            <span>Advanced VRM Visemes:</span>
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
            "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/",
            "@pixiv/three-vrm": "https://unpkg.com/@pixiv/three-vrm@2.0.6/lib/three-vrm.module.js"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
        import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

        // Global variables
        let scene, camera, renderer, gltfLoader;
        let avatar = null;
        let vrm = null;
        let avatarWebSocket = null;
        let webrtcClient = null;
        let isRecording = false;
        let webrtcId = Math.random().toString(36).substring(7);

        // Enhanced VRoid blend shape mapping for VRM files
        const ENHANCED_VRM_BLEND_SHAPES = {
            'Fcl_MTH_A': 'mouth_open_wide',
            'Fcl_MTH_E': 'mouth_smile', 
            'Fcl_MTH_I': 'mouth_narrow',
            'Fcl_MTH_O': 'mouth_round',
            'Fcl_MTH_U': 'mouth_pucker',
            'Fcl_MTH_Close': 'mouth_closed',
            'Fcl_MTH_Neutral': 'mouth_neutral',
            'Fcl_MTH_Small': 'mouth_small',
            'Fcl_MTH_Large': 'mouth_large',
            'Fcl_MTH_Fun': 'mouth_fun',
            'Fcl_MTH_Down': 'mouth_down',
            'Fcl_ALL_Joy': 'expression_joy',
            'Fcl_ALL_Sorrow': 'expression_sad',
            'Fcl_ALL_Surprised': 'expression_surprised',
            'Fcl_ALL_Angry': 'expression_angry',
            'Fcl_ALL_Neutral': 'expression_neutral',
            'Fcl_EYE_Natural': 'eye_natural',
            'Fcl_EYE_Joy': 'eye_joy',
            'Fcl_EYE_Sorrow': 'eye_sad',
            'Fcl_EYE_Surprised': 'eye_surprised',
            'Fcl_EYE_Angry': 'eye_angry',
            'Fcl_BRW_Angry': 'brow_angry'
        };

        // Initialize Avatar WebSocket with VRM support
        async function initAvatarWebSocket() {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                avatarWebSocket = new WebSocket(`${protocol}//${window.location.host}/ws/avatar`);

                avatarWebSocket.onopen = () => {
                    updateStatus('avatarStatus', true);
                    console.log('✅ Enhanced VRM Avatar WebSocket connected');
                    document.getElementById('avatarConnectionStatus').textContent = 'Connected';
                };

                avatarWebSocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'viseme_update') {
                        applyEnhancedVRMBlendShapes(data.blend_shapes);
                        updateEnhancedVisemeDebug(data);
                    }
                };

                avatarWebSocket.onclose = () => {
                    updateStatus('avatarStatus', false);
                    document.getElementById('avatarConnectionStatus').textContent = 'Disconnected';
                    setTimeout(initAvatarWebSocket, 3000);
                };

            } catch (error) {
                console.error('VRM Avatar WebSocket error:', error);
            }
        }

        // Initialize Three.js for VRM Avatar
        async function initThreeJS() {
            try {
                const viewport = document.getElementById('avatarCanvas').parentElement;
                const aspect = viewport.clientWidth / viewport.clientHeight || 1;

                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);

                // Camera optimized for VRM face focus
                camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
                camera.position.set(0, 0, 2);
                camera.lookAt(0, 0, 0);

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

                // Enhanced lighting for VRM models
                const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
                scene.add(hemi);

                const key = new THREE.DirectionalLight(0xffffff, 1.2);
                key.position.set(0, 1, 2);
                scene.add(key);

                const rim = new THREE.DirectionalLight(0xffffff, 0.6);
                rim.position.set(0, 1, -2);
                scene.add(rim);

                // Initialize GLTF loader with VRM plugin
                gltfLoader = new GLTFLoader();
                gltfLoader.register((parser) => new VRMLoaderPlugin(parser));

                window.addEventListener('resize', onWindowResize);
                onWindowResize();

                console.log('✅ Enhanced Three.js with VRM support initialized');
                return true;

            } catch (error) {
                console.error('Three.js VRM error:', error);
                return false;
            }
        }
        
        async function loadVRMModel(path) {
            return new Promise((resolve, reject) => {
                gltfLoader.load(
                    path,
                    (gltf) => {
                        // Convert GLTF to VRM
                        VRMUtils.removeUnnecessaryVertices(gltf.scene);
                        VRMUtils.removeUnnecessaryJoints(gltf.scene);
        
                        const vrmInstance = VRM.from(gltf);
                        vrmInstance.then((loadedVrm) => {
                            vrm = loadedVrm;           // Save to global `vrm` reference
                            scene.add(vrm.scene);      // ✅ THIS is the critical part
                            console.log("✅ VRM model added to scene");
                            resolve(loadedVrm);
                        });
                    },
                    undefined,
                    (error) => {
                        console.error("❌ Failed to load VRM model", error);
                        reject(error);
                    }
                );
            });
        }


        // Load VRM Avatar with enhanced support
        async function loadVRMAvatar() {
            # const vrmPaths = ['/static/avatar.vrm', '/static/joined1111.vrm', '/static/test.vrm'];
            const vrmPaths = ['/static/4thjuly.vrm'];

            for (const path of vrmPaths) {
                try {
                    console.log(`Loading VRM avatar: ${path}`);

                    const gltf = await new Promise((resolve, reject) => {
                        gltfLoader.load(path, resolve, undefined, reject);
                    });

                    vrm = gltf.userData.vrm;
                    if (!vrm) {
                        console.log(`No VRM data found in ${path}`);
                        continue;
                    }

                    if (avatar) scene.remove(avatar);

                    avatar = vrm.scene;
                    scene.add(avatar);

                    // VRM-specific optimizations
                    VRMUtils.removeUnnecessaryVertices(vrm.scene);
                    VRMUtils.removeUnnecessaryJoints(vrm.scene);

                    // Enhanced VRM blend shape detection
                    vrm.scene.traverse((child) => {
                        if (child.isMesh && child.morphTargetInfluences) {
                            child.userData.morphTargets = child.morphTargetDictionary;
                            child.userData.enhancedMapping = {};
                            
                            if (child.morphTargetDictionary) {
                                const morphNames = Object.keys(child.morphTargetDictionary);
                                console.log('🎭 VRM morph targets found:', morphNames);
                                
                                // Map VRM blend shapes
                                morphNames.forEach(name => {
                                    if (ENHANCED_VRM_BLEND_SHAPES[name]) {
                                        child.userData.enhancedMapping[name] = child.morphTargetDictionary[name];
                                    }
                                });
                                
                                console.log('🎯 VRM mapping created:', child.userData.enhancedMapping);
                            }
                        }
                    });

                    // Check VRM expression manager
                    if (vrm.expressionManager) {
                        console.log('✅ VRM Expression Manager found');
                        console.log('VRM expressions:', Object.keys(vrm.expressionManager.expressionMap || {}));
                    }

                    document.getElementById('vrmStatus').innerHTML = '🎭 VRM Model Loaded';
                    console.log(`✅ VRM avatar loaded: ${path}`);
                    return true;

                } catch (error) {
                    console.log(`Failed to load VRM: ${path}`);
                }
            }

            console.log('❌ No VRM avatar found. Please place a .vrm file in /static/');
            document.getElementById('vrmStatus').innerHTML = '❌ No VRM Found';
            return false;
        }

        // Apply enhanced VRM blend shapes
        function applyEnhancedVRMBlendShapes(blendShapeValues) {
            if (!vrm) return;

            // Use VRM expression manager if available
            if (vrm.expressionManager) {
                for (const [shapeName, value] of Object.entries(blendShapeValues)) {
                    if (vrm.expressionManager.expressionMap[shapeName]) {
                        vrm.expressionManager.setValue(shapeName, Math.max(0, Math.min(1, value)));
                    }
                }
                vrm.expressionManager.update();
            } else {
                // Fallback to direct morph target manipulation
                avatar.traverse((child) => {
                    if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
                        // Smooth decay
                        for (let i = 0; i < child.morphTargetInfluences.length; i++) {
                            child.morphTargetInfluences[i] *= 0.9;
                        }

                        // Apply new values
                        for (const [shapeName, value] of Object.entries(blendShapeValues)) {
                            const index = child.morphTargetDictionary[shapeName];
                            if (index !== undefined && value > 0.001) {
                                const currentValue = child.morphTargetInfluences[index] || 0;
                                child.morphTargetInfluences[index] = currentValue + (value - currentValue) * 0.3;
                            }
                        }
                    }
                });
            }
        }

        // Update enhanced viseme debug display
        function updateEnhancedVisemeDebug(data) {
            const blendShapes = data.blend_shapes || {};
            const emotion = data.emotion || 'neutral';
            
            document.getElementById('currentEmotion').textContent = emotion;
            document.getElementById('activeBlendShapes').textContent = Object.keys(blendShapes).length;
            
            // Find dominant blend shape
            let dominantShape = 'sil';
            let maxWeight = 0;
            
            for (const [shapeName, weight] of Object.entries(blendShapes)) {
                if (weight > maxWeight) {
                    maxWeight = weight;
                    dominantShape = shapeName;
                }
            }
            
            if (maxWeight > 0.1) {
                document.getElementById('dominantViseme').textContent = dominantShape;
            }
        }

        // Enhanced emotion setting for VRM
        async function setEmotion(emotion) {
            try {
                // Update UI
                document.querySelectorAll('.emotion-btn').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
                
                // Send to server
                const response = await fetch('/set_emotion', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ emotion: emotion })
                });
                
                if (response.ok) {
                    console.log(`✅ VRM emotion set to: ${emotion}`);
                    document.getElementById('currentEmotion').textContent = emotion;
                } else {
                    console.error('Failed to set VRM emotion');
                }
            } catch (error) {
                console.error('Error setting VRM emotion:', error);
            }
        }

        // WebRTC Client for AI Chat (same as before)
        class EnhancedWebRTCClient {
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
                    console.error('Enhanced WebRTC connection error:', error);
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

        // Initialize enhanced SSE for AI messages
        function initEnhancedSSE() {
            const eventSource = new EventSource(`/updates?webrtc_id=${webrtcId}`);

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === "stt" || data.type === "llm") {
                        addMessage(data.type === "stt" ? "user" : "ai", data.text);
                    } else if (data.type === "visemes") {
                        console.log("Received enhanced VRM visemes from AI:", data);
                        // Enhanced visemes are automatically processed by the backend
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
            
            // Update VRM if available
            if (vrm) {
                vrm.update(0.016); // ~60fps delta time
            }
            
            renderer.render(scene, camera);
        }

        // Enhanced Voice Button Handling
        document.getElementById('voiceBtn').addEventListener('mousedown', async () => {
            if (!isRecording) {
                try {
                    if (!webrtcClient) {
                        webrtcClient = new EnhancedWebRTCClient();
                    }
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

        // Make setEmotion globally available
        window.setEmotion = setEmotion;

        // Initialize the enhanced VRM application
        async function init() {
            console.log('🚀 Starting Enhanced VRM Avatar + AI Chat with Advanced VRoid Visemes...');

            // Initialize enhanced avatar system
            await initAvatarWebSocket();
            const threeReady = await initThreeJS();
            await loadVRMModel('/static/4thjuly.vrm');  // Path to your VRM file

            if (threeReady) {
                const vrmLoaded = await loadVRMAvatar();
                if (vrmLoaded) {
                    animate();
                    console.log('✅ VRM avatar system ready');
                } else {
                    console.warn('⚠️ VRM avatar not loaded, but system continues');
                    animate(); // Still animate scene even without VRM
                }
            }

            // Initialize enhanced AI chat system
            initEnhancedSSE();

            console.log('✅ Enhanced VRM application initialized');
        }

        // Start the enhanced VRM application
        window.addEventListener('load', init);
    </script>
</body>
</html>
    '''

# Enhanced WebSocket endpoint for VRM avatar
@app.websocket("/ws/avatar")
async def enhanced_vrm_avatar_websocket_endpoint(websocket: WebSocket):
    await enhanced_viseme_controller.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "update_viseme":
                phoneme = message.get("phoneme", "sil")
                emotion = message.get("emotion", "neutral")
                await enhanced_viseme_controller.update_single_viseme(phoneme, emotion)

    except WebSocketDisconnect:
        enhanced_viseme_controller.disconnect(websocket)

# Enhanced phoneme playback endpoint for VRM
@app.post("/play_phonemes")
async def enhanced_vrm_play_phonemes_endpoint(seq: PhonemeSeq):
    """Play phoneme sequences with enhanced VRM visemes"""
    phoneme_sequence = [(i.phoneme, i.start, i.end) for i in seq.items]

    try:
        await enhanced_viseme_controller.play_phoneme_sequence(phoneme_sequence)
        return {"status": "success", "frames": len(phoneme_sequence), "type": "VRM"}
    except Exception as e:
        logger.error(f"Failed to play VRM phonemes: {e}")
        return {"status": "error", "message": str(e)}

# Enhanced VRM endpoints
@app.post("/set_emotion")
async def set_vrm_emotion_endpoint(request: EmotionRequest):
    """Set VRM avatar emotion"""
    try:
        await enhanced_viseme_controller.set_emotion(request.emotion)
        return {"status": "success", "emotion": request.emotion, "type": "VRM"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/trigger_viseme")
async def trigger_vrm_viseme_endpoint(request: VisemeRequest):
    """Manually trigger a VRM viseme"""
    try:
        await enhanced_viseme_controller.update_single_viseme(request.phoneme, request.emotion)
        return {"status": "success", "phoneme": request.phoneme, "emotion": request.emotion, "type": "VRM"}
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

# Enhanced VRM viseme info endpoint
@app.get("/enhanced_vrm_viseme_info")
async def get_enhanced_vrm_viseme_info():
    """Return enhanced VRM viseme mapping information"""
    return {
        "enhanced_vrm_mapping": enhanced_viseme_controller.viseme_mapper.vroid_blend_shapes,
        "phoneme_mapping": enhanced_viseme_controller.viseme_mapper.phoneme_to_viseme,
        "available_emotions": enhanced_viseme_controller.get_available_emotions(),
        "transition_types": ["linear", "smooth", "cubic", "anticipate"],
        "current_state": enhanced_viseme_controller.get_current_state(),
        "model_type": "VRM",
        "vroid_blend_shapes": list(enhanced_viseme_controller.viseme_mapper.vroid_blend_shapes.keys())
    }


@app.post("/test_avatar_broadcast")
async def test_avatar_broadcast():
    """Test endpoint to manually trigger avatar movement"""
    try:
        test_blend_shapes = {
            "Fcl_MTH_A": 0.8,
            "Fcl_MTH_Close": 0.2,
            "Fcl_EYE_Natural": 0.9,
            "Fcl_ALL_Joy": 0.3
        }

        await enhanced_viseme_controller._broadcast_blend_shapes(test_blend_shapes)
        return {"status": "success", "message": "Test broadcast sent"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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