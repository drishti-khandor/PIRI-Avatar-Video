"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Types
interface ChatMessage {
    type: "stt" | "llm";
    text: string;
    timestamp?: number;
}

interface VRMWebSocketData {
    type: string;
    blend_shapes?: Record<string, number>;
    emotion?: string;
    is_speaking?: boolean;
    timestamp?: number;
}

interface WebRTCClientOptions {
    onConnected?: () => void;
    onDisconnected?: () => void;
    onMessage?: (message: any) => void;
    onAudioStream?: (stream: MediaStream) => void;
    onAudioLevel?: (level: number) => void;
    webrtcId?: string;
}

// Clean WebRTC Client
class CleanWebRTCClient {
    private peerConnection: RTCPeerConnection | null = null;
    private mediaStream: MediaStream | null = null;
    private dataChannel: RTCDataChannel | null = null;
    private options: WebRTCClientOptions;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private dataArray: Uint8Array | null = null;
    private animationFrameId: number | null = null;
    private webrtcId: string | undefined;

    constructor(options: WebRTCClientOptions = {}) {
        this.options = options;
        this.webrtcId = options.webrtcId;
    }

    async connect() {
        try {
            this.peerConnection = new RTCPeerConnection();

            const constraints: MediaStreamConstraints = { audio: true };
            this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

            this.setupAudioAnalysis();

            this.mediaStream.getTracks().forEach(track => {
                if (this.peerConnection) {
                    this.peerConnection.addTrack(track, this.mediaStream!);
                }
            });

            this.peerConnection.addEventListener('track', (event) => {
                if (this.options.onAudioStream) {
                    this.options.onAudioStream(event.streams[0]);
                }
            });

            this.dataChannel = this.peerConnection.createDataChannel('text');

            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            const response = await fetch('http://localhost:8000/webrtc/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    webrtc_id: this.webrtcId
                })
            });

            const serverResponse = await response.json();
            await this.peerConnection.setRemoteDescription(serverResponse);

            if (this.options.onConnected) {
                this.options.onConnected();
            }
        } catch (error) {
            console.error('WebRTC connection error:', error);
            this.disconnect();
            throw error;
        }
    }

    private setupAudioAnalysis() {
        if (!this.mediaStream) return;

        try {
            this.audioContext = new AudioContext();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.analyser);

            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);

            this.startAnalysis();
        } catch (error) {
            console.error('Audio analysis setup error:', error);
        }
    }

    private startAnalysis() {
        if (!this.analyser || !this.dataArray || !this.options.onAudioLevel) return;

        const analyze = () => {
            this.analyser!.getByteFrequencyData(this.dataArray!);

            let sum = 0;
            for (let i = 0; i < this.dataArray!.length; i++) {
                sum += this.dataArray![i];
            }
            const average = sum / this.dataArray!.length / 255;

            this.options.onAudioLevel!(average);
            this.animationFrameId = requestAnimationFrame(analyze);
        };

        this.animationFrameId = requestAnimationFrame(analyze);
    }

    disconnect() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }

        if (this.options.onDisconnected) {
            this.options.onDisconnected();
        }
    }
}

// Main Component
export function CleanVRMAvatar() {
    // State
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [avatarStatus, setAvatarStatus] = useState<'connected' | 'disconnected'>('disconnected');
    const [currentEmotion, setCurrentEmotion] = useState<string>('neutral');
    const [audioLevel, setAudioLevel] = useState(0);
    const [vrmLoaded, setVrmLoaded] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [blendShapeCount, setBlendShapeCount] = useState(0);

    // Refs
    const webrtcClientRef = useRef<CleanWebRTCClient | null>(null);
    const audioRef = useRef<HTMLAudioElement>(null);
    const avatarWebSocketRef = useRef<WebSocket | null>(null);
    const chatBottomRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneRef = useRef<any>(null);
    const vrmRef = useRef<any>(null);
    const webrtcId = useRef(Math.random().toString(36).substring(7));

    // Auto-scroll chat
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    // Initialize VRM Avatar WebSocket
    const initVRMWebSocket = useCallback(() => {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//localhost:8000/ws/avatar`);

            ws.onopen = () => {
                setAvatarStatus('connected');
                console.log('✅ VRM WebSocket connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data: VRMWebSocketData = JSON.parse(event.data);
                    console.log('📨 Received WebSocket message:', data.type);
                    
                    if (data.type === 'vrm_update') {
                        if (data.blend_shapes) {
                            console.log('🎭 Received blend shapes:', data.blend_shapes);
                            applyVRMBlendShapes(data.blend_shapes);
                            setBlendShapeCount(Object.keys(data.blend_shapes).length);
                        }
                        if (data.emotion) {
                            setCurrentEmotion(data.emotion);
                        }
                        if (data.is_speaking !== undefined) {
                            setIsSpeaking(data.is_speaking);
                        }
                    }
                } catch (error) {
                    console.error('VRM WebSocket message error:', error);
                }
            };

            ws.onclose = () => {
                setAvatarStatus('disconnected');
                console.log('❌ VRM WebSocket disconnected');
                setTimeout(initVRMWebSocket, 3000);
            };

            avatarWebSocketRef.current = ws;

        } catch (error) {
            console.error('VRM WebSocket initialization failed:', error);
            setTimeout(initVRMWebSocket, 3000);
        }
    }, []);

    // Initialize Three.js with VRM
    const initThreeJSWithVRM = useCallback(async () => {
        if (!canvasRef.current) return false;

        try {
            const THREE = await import('three');
            const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');

            let VRMLoaderPlugin, VRMUtils;
            let isVRMSupported = false;

            try {
                const vrmModule = await import('@pixiv/three-vrm');
                VRMLoaderPlugin = vrmModule.VRMLoaderPlugin;
                VRMUtils = vrmModule.VRMUtils;
                isVRMSupported = true;
                console.log('✅ VRM support loaded');
            } catch (vrmError) {
                console.warn('⚠️ VRM support not available:', vrmError);
                isVRMSupported = false;
            }

            const canvas = canvasRef.current;
            const container = canvas.parentElement;
            if (!container) return false;

            const aspect = container.clientWidth / container.clientHeight || 1;

            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);

            // Camera
            const camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
            camera.position.set(0, 1.5, 2);
            camera.lookAt(0, 1.5, 0);

            // Renderer
            const renderer = new THREE.WebGLRenderer({
                canvas,
                antialias: true,
                preserveDrawingBuffer: true,
                powerPreference: "high-performance"
            });

            renderer.physicallyCorrectLights = true;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.0;
            renderer.setSize(container.clientWidth, container.clientHeight);
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

            // GLTF loader
            const gltfLoader = new GLTFLoader();
            if (isVRMSupported && VRMLoaderPlugin) {
                gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
            }

            sceneRef.current = { scene, camera, renderer };

            // Load VRM
            try {
                console.log('Loading VRM: /static/4thjuly.vrm');
                const gltf = await new Promise<any>((resolve, reject) => {
                    gltfLoader.load('http://localhost:8000/static/4thjuly.vrm', resolve, undefined, reject);
                });

                const vrm = gltf.userData?.vrm;
                if (vrm) {
                    vrmRef.current = vrm;
                    scene.add(vrm.scene);

                    if (VRMUtils) {
                        VRMUtils.removeUnnecessaryVertices(vrm.scene);
                        VRMUtils.removeUnnecessaryJoints(vrm.scene);
                    }
                    
                    // Debug VRM structure
                    console.log('🎭 VRM loaded, checking structure...');
                    console.log('VRM object:', vrm);
                    console.log('Has expressionManager:', !!vrm.expressionManager);
                    
                    if (vrm.expressionManager) {
                        console.log('Available VRM expressions:', Object.keys(vrm.expressionManager.expressionMap || {}));
                    }
                    
                    // Check for morph targets as fallback
                    vrm.scene.traverse((child: any) => {
                        if (child.isMesh && child.morphTargetDictionary) {
                            console.log(`Mesh "${child.name}" morph targets:`, Object.keys(child.morphTargetDictionary));
                        }
                    });

                    // Camera positioning
                    const box = new THREE.Box3().setFromObject(vrm.scene);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());

                    const faceTarget = center.clone();
                    faceTarget.y += size.y * 0.35;

                    camera.position.set(faceTarget.x, faceTarget.y, faceTarget.z + size.z * 2);
                    camera.lookAt(faceTarget);

                    console.log('✅ VRM loaded successfully');
                    setVrmLoaded(true);
                } else {
                    console.error('❌ No VRM data found');
                }

            } catch (error) {
                console.error('❌ Failed to load VRM:', error);
            }

            // Animation loop
            const animate = () => {
                requestAnimationFrame(animate);

                if (vrmRef.current && vrmRef.current.update) {
                    vrmRef.current.update(0.016);
                }

                renderer.render(scene, camera);
            };
            animate();

            // Resize handler
            const handleResize = () => {
                if (!container || !sceneRef.current) return;
                const { camera, renderer } = sceneRef.current;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            };

            window.addEventListener('resize', handleResize);

            return true;

        } catch (error) {
            console.error('Three.js initialization error:', error);
            return false;
        }
    }, []);

    // Apply VRM blend shapes
    const applyVRMBlendShapes = useCallback((blendShapes: Record<string, number>) => {
        if (!vrmRef.current) return;

        console.log('🎭 Applying blend shapes:', blendShapes);
        
        // Check for significant blend shapes
        const significantShapes = Object.entries(blendShapes).filter(([_, value]) => value > 0.01);
        if (significantShapes.length > 0) {
            console.log('📊 Significant shapes:', significantShapes);
        }

        if (vrmRef.current.expressionManager) {
            console.log('✅ Using VRM Expression Manager');
            console.log('Available expressions:', Object.keys(vrmRef.current.expressionManager.expressionMap || {}));
            
            for (const [shapeName, value] of Object.entries(blendShapes)) {
                if (vrmRef.current.expressionManager.expressionMap[shapeName]) {
                    vrmRef.current.expressionManager.setValue(shapeName, Math.max(0, Math.min(1, value)));
                    console.log(`✅ Set VRM expression: ${shapeName} = ${value}`);
                } else {
                    console.log(`❌ VRM expression not found: ${shapeName}`);
                }
            }
            vrmRef.current.expressionManager.update();
            console.log('✅ VRM Expression Manager updated');
        } else {
            console.log('⚠️ Using fallback morph targets');
            // Fallback to morph targets
            const targetObject = vrmRef.current.scene || vrmRef.current;
            targetObject.traverse((child: any) => {
                if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
                    console.log(`🎯 Applying to mesh: ${child.name}`);
                    console.log(`Available morph targets:`, Object.keys(child.morphTargetDictionary));
                    
                    for (const [shapeName, value] of Object.entries(blendShapes)) {
                        const index = child.morphTargetDictionary[shapeName];
                        if (index !== undefined) {
                            child.morphTargetInfluences[index] = Math.max(0, Math.min(1, value));
                            console.log(`✅ Set morph target: ${shapeName}[${index}] = ${value}`);
                        } else {
                            console.log(`❌ Morph target not found: ${shapeName}`);
                        }
                    }
                }
            });
        }
    }, []);

    // Initialize AI Chat SSE
    const initAIChatSSE = useCallback(() => {
        const eventSource = new EventSource(`http://localhost:8000/updates?webrtc_id=${webrtcId.current}`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "stt" || data.type === "llm") {
                    setChatMessages(prev => [...prev, {
                        type: data.type,
                        text: data.text,
                        timestamp: Date.now()
                    }]);
                }
            } catch (err) {
                console.error("SSE parse error", err);
            }
        };

        return eventSource;
    }, []);

    // Set emotion
    const setEmotion = useCallback(async (emotion: string) => {
        try {
            const response = await fetch('http://localhost:8000/set_emotion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ emotion })
            });

            if (response.ok) {
                console.log(`✅ Emotion set to: ${emotion}`);
            }
        } catch (error) {
            console.error('Error setting emotion:', error);
        }
    }, []);

    // Toggle recording
    const toggleRecording = useCallback(async () => {
        if (isRecording) {
            if (webrtcClientRef.current) {
                webrtcClientRef.current.disconnect();
                setIsRecording(false);
                setIsConnected(false);
            }
        } else {
            try {
                if (!webrtcClientRef.current) {
                    webrtcClientRef.current = new CleanWebRTCClient({
                        webrtcId: webrtcId.current,
                        onAudioLevel: setAudioLevel,
                        onConnected: () => setIsConnected(true),
                        onDisconnected: () => setIsConnected(false),
                        onAudioStream: (stream: MediaStream) => {
                            if (!audioRef.current) {
                                const audio = document.createElement('audio');
                                audio.autoplay = true;
                                audio.volume = 1.0;
                                document.body.appendChild(audio);
                                audioRef.current = audio;
                            }

                            audioRef.current.srcObject = stream;
                            audioRef.current.play().catch(console.error);
                        }
                    });
                }

                await webrtcClientRef.current.connect();
                setIsRecording(true);
                setIsConnected(true);

            } catch (error) {
                console.error('Failed to start recording:', error);
            }
        }
    }, [isRecording]);

    // Initialize everything
    useEffect(() => {
        const init = async () => {
            console.log('🚀 Initializing Clean VRM Avatar...');

            initVRMWebSocket();
            await initThreeJSWithVRM();
            const eventSource = initAIChatSSE();

            console.log('✅ Initialization complete');

            return () => {
                eventSource.close();
                if (avatarWebSocketRef.current) {
                    avatarWebSocketRef.current.close();
                }
                if (webrtcClientRef.current) {
                    webrtcClientRef.current.disconnect();
                }
            };
        };

        init();
    }, [initVRMWebSocket, initThreeJSWithVRM, initAIChatSSE]);

    return (
        <div className="relative w-full h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-hidden">
            {/* Background Effects */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-1/4 -right-1/4 w-1/2 h-1/2 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"></div>
                <div className="absolute -bottom-1/4 -left-1/4 w-1/2 h-1/2 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"></div>
            </div>

            {/* Main Container */}
            <div className="relative z-10 h-full flex">
                {/* VRM Avatar Panel */}
                <div className="flex-1 bg-black/20 backdrop-blur-sm border-r border-white/10">
                    {/* Header */}
                    <div className="bg-black/30 backdrop-blur-sm p-4 border-b border-white/10">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            🤖 Clean VRM Avatar
                            <span className={`inline-block w-3 h-3 rounded-full ${
                                avatarStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                            }`}></span>
                        </h2>
                        <p className="text-sm text-gray-300">OVRLipsync + Smooth Animation</p>
                    </div>

                    {/* Canvas Container */}
                    <div className="relative h-[calc(100vh-80px)]">
                        <canvas ref={canvasRef} className="w-full h-full" />

                        {/* Status Display */}
                        <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm p-3 rounded-lg text-white">
                            <div className="text-xs text-gray-300 mb-2">VRM Status:</div>
                            <div className="space-y-1 text-xs">
                                <div>Model: <span className={vrmLoaded ? "text-green-400" : "text-yellow-400"}>
                                    {vrmLoaded ? "Loaded" : "Loading..."}
                                </span></div>
                                <div>Emotion: <span className="text-blue-400">{currentEmotion}</span></div>
                                <div>Speaking: <span className={isSpeaking ? "text-green-400" : "text-gray-400"}>
                                    {isSpeaking ? "Yes" : "No"}
                                </span></div>
                                <div>Shapes: <span className="text-yellow-400">{blendShapeCount}</span></div>
                            </div>
                        </div>

                        {/* Emotion Controls */}
                        <div className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-sm p-3 rounded-lg">
                            <div className="text-xs text-white mb-2 text-center">Emotions</div>
                            <div className="grid grid-cols-2 gap-1">
                                {['neutral', 'happy', 'sad', 'surprised', 'angry'].map((emotion) => (
                                    <button
                                        key={emotion}
                                        onClick={() => setEmotion(emotion)}
                                        className={`px-2 py-1 text-xs rounded transition-colors ${
                                            currentEmotion === emotion
                                                ? 'bg-blue-500 text-white'
                                                : 'bg-gray-600 text-gray-200 hover:bg-gray-500'
                                        }`}
                                    >
                                        {emotion}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Loading State */}
                        {!vrmLoaded && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="text-center text-white">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                                    <p>Loading VRM Avatar...</p>
                                    <p className="text-sm text-gray-300 mt-2">Place 4thjuly.vrm in /static/</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* AI Chat Panel */}
                <div className="flex-1 bg-black/20 backdrop-blur-sm flex flex-col">
                    {/* Header */}
                    <div className="bg-black/30 backdrop-blur-sm p-4 border-b border-white/10">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            🧠 AI Assistant
                            <span className={`inline-block w-3 h-3 rounded-full ${
                                isConnected ? 'bg-green-400' : 'bg-red-400'
                            }`}></span>
                        </h2>
                        <p className="text-sm text-gray-300">Real-time Speech with VRM Lip-Sync</p>
                    </div>

                    {/* Chat Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                        <AnimatePresence>
                            {chatMessages.length === 0 ? (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="h-full flex flex-col items-center justify-center text-center text-gray-400"
                                >
                                    <div className="w-16 h-16 mb-4 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center">
                                        <svg className="w-8 h-8 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                                        </svg>
                                    </div>
                                    <p className="text-lg font-medium text-gray-300">Start a conversation</p>
                                    <p className="text-sm mt-1">Click the microphone to speak with clean VRM lip-sync</p>
                                </motion.div>
                            ) : (
                                chatMessages.map((msg, idx) => (
                                    <motion.div
                                        key={idx}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className={`flex ${msg.type === "stt" ? "justify-end" : "justify-start"} mb-3`}
                                    >
                                        <div
                                            className={`rounded-2xl px-4 py-3 max-w-[80%] shadow-sm backdrop-blur-sm ${
                                                msg.type === "stt"
                                                    ? "bg-blue-500/80 text-white rounded-br-none"
                                                    : "bg-white/80 text-gray-900 rounded-bl-none border border-gray-100/20"
                                            }`}
                                        >
                                            <p className="text-sm leading-relaxed">{msg.text}</p>
                                        </div>
                                    </motion.div>
                                ))
                            )}
                        </AnimatePresence>
                        <div ref={chatBottomRef} />
                    </div>

                    {/* Audio Level */}
                    <div className="px-4 mb-2">
                        <div className="h-1 w-full bg-white/20 rounded-full overflow-hidden">
                            <motion.div
                                className={`h-full rounded-full transition-colors ${
                                    audioLevel > 0.3 ? 'bg-green-400' : audioLevel > 0.1 ? 'bg-yellow-400' : 'bg-blue-400'
                                }`}
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min(audioLevel * 100, 100)}%` }}
                                transition={{ type: "spring", damping: 15 }}
                            />
                        </div>
                    </div>

                    {/* Voice Controls */}
                    <div className="p-4">
                        <div className="flex items-center justify-center gap-4">
                            <button
                                className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all duration-300 ${
                                    isRecording
                                        ? "bg-red-500 hover:bg-red-600 animate-pulse shadow-lg shadow-red-500/50"
                                        : "bg-blue-500 hover:bg-blue-600 hover:scale-110 shadow-lg shadow-blue-500/50"
                                } text-white`}
                                onClick={toggleRecording}
                            >
                                {isRecording ? "🔴" : "🎤"}
                            </button>

                            <div className="flex flex-col gap-1 text-xs text-gray-300">
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${
                                        avatarStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                                    }`}></span>
                                    <span>VRM Avatar</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${
                                        isConnected ? 'bg-green-400' : 'bg-red-400'
                                    }`}></span>
                                    <span>AI Chat</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${
                                        vrmLoaded ? 'bg-green-400' : 'bg-yellow-400'
                                    }`}></span>
                                    <span>OVRLipsync</span>
                                </div>
                            </div>
                        </div>

                        <div className="mt-3 text-center text-xs text-gray-400">
                            {isRecording ? (
                                <div className="space-y-1">
                                    <div className="flex items-center justify-center gap-2">
                                        <div className="animate-pulse w-2 h-2 bg-red-400 rounded-full"></div>
                                        <span>🎙️ Recording with OVRLipsync...</span>
                                    </div>
                                    <div>Click again to stop and process</div>
                                </div>
                            ) : (
                                <div className="space-y-1">
                                    <div>Click microphone to start clean VRM conversation</div>
                                    <div className="text-blue-400">✨ OVRLipsync • Smooth Animation • Perfect Sync</div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Status Bar */}
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black/50 backdrop-blur-sm px-4 py-2 rounded-lg text-white text-sm">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span>🎭</span>
                        <span>VRM:</span>
                        <span className={vrmLoaded ? "text-green-400" : "text-yellow-400"}>
                            {vrmLoaded ? "Active" : "Loading..."}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span>🎵</span>
                        <span>OVRLipsync:</span>
                        <span className="text-green-400">Ready</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span>😊</span>
                        <span>Emotion:</span>
                        <span className="text-blue-400">{currentEmotion}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default CleanVRMAvatar;