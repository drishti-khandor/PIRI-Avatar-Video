"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Types
interface VisemeData {
    viseme: string;
    start_time: number;
    end_time: number;
    confidence: number;
}

interface ChatMessage {
    type: "stt" | "llm";
    text: string;
    timestamp?: number;
}

interface AvatarWebSocketData {
    type: string;
    visemes?: Record<string, number>;
    blend_shapes?: Record<string, number>;
    timestamp?: number;
}

interface WebRTCClientOptions {
    onConnected?: () => void;
    onDisconnected?: () => void;
    onMessage?: (message: any) => void;
    onAudioStream?: (stream: MediaStream) => void;
    onAudioLevel?: (level: number) => void;
    audioInputDeviceId?: string;
    audioOutputDeviceId?: string;
    webrtcId?: string;


}

    class UnifiedWebRTCClient {
    private peerConnection: RTCPeerConnection | null = null;
    private mediaStream: MediaStream | null = null;
    private dataChannel: RTCDataChannel | null = null;
    private options: WebRTCClientOptions;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private dataArray: Uint8Array | null = null;
    private animationFrameId: number | null = null;
    private currentInputDeviceId: string | undefined = undefined;
    private currentOutputDeviceId: string | undefined = undefined;
    private webrtcId: string | undefined;

    constructor(options: WebRTCClientOptions = {}) {
        this.options = options;
        this.currentInputDeviceId = options.audioInputDeviceId;
        this.currentOutputDeviceId = options.audioOutputDeviceId;
        this.webrtcId = options.webrtcId;
    }

    // Method to change audio input device
    setAudioInputDevice(deviceId: string) {
        this.currentInputDeviceId = deviceId;

        // If we're already connected, reconnect with the new device
        if (this.peerConnection) {
            this.disconnect();
            this.connect();
        }
    }

    // Method to change audio output device
    setAudioOutputDevice(deviceId: string) {
        this.currentOutputDeviceId = deviceId;

        // Apply to any current audio elements
        if (this.options.onAudioStream) {
            // The onAudioStream callback should handle setting the output device
            // We'll pass the updated device ID through the options
            this.options.audioOutputDeviceId = deviceId;
        }
    }

    async connect() {
        try {
            this.peerConnection = new RTCPeerConnection();

            // Get user media with specific device if specified
            try {
                const constraints: MediaStreamConstraints = {
                    audio: this.currentInputDeviceId
                        ? { deviceId: { exact: this.currentInputDeviceId } }
                        : true
                };

                this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (mediaError: any) {
                console.error('Media error:', mediaError);
                if (mediaError.name === 'NotAllowedError') {
                    throw new Error('Microphone access denied. Please allow microphone access and try again.');
                } else if (mediaError.name === 'NotFoundError') {
                    throw new Error('No microphone detected. Please connect a microphone and try again.');
                } else {
                    throw mediaError;
                }
            }

            this.setupAudioAnalysis();

            this.mediaStream.getTracks().forEach(track => {
                if (this.peerConnection) {
                    this.peerConnection.addTrack(track, this.mediaStream!);
                }
            });

            this.peerConnection.addEventListener('track', (event) => {
                if (this.options.onAudioStream) {
                    const stream = event.streams[0];

                    // If we have an audio output device specified and the browser supports setSinkId
                    if (this.currentOutputDeviceId && 'setSinkId' in HTMLAudioElement.prototype) {
                        // We'll let the callback handle this, as we need access to the audio element
                        this.options.audioOutputDeviceId = this.currentOutputDeviceId;
                    }

                    this.options.onAudioStream(stream);
                }
            });

            this.dataChannel = this.peerConnection.createDataChannel('text');

            this.dataChannel.addEventListener('message', (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log('Received message:', message);

                    if (this.options.onMessage) {
                        this.options.onMessage(message);
                    }
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            });

            // Create and send offer
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            // Use same-origin request to avoid CORS preflight
            const response = await fetch('http://localhost:8001/webrtc/offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                mode: 'cors', // Explicitly set CORS mode
                credentials: 'same-origin',
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    webrtc_id: this.webrtcId
                })
            });
            // const serverResponse = await response.json();
            console.log('Response status:', response.status);
            const responseText = await response.text();
            console.log('Raw response:', responseText);
             let serverResponse: any;
            try {
                serverResponse = JSON.parse(responseText);
                console.log('Parsed response:', serverResponse);
            } catch (e) {
                console.error('Failed to parse response as JSON:', e);
                throw new Error(`Invalid response format: ${responseText}`);
            }

            if (this.peerConnection) {
                await this.peerConnection.setRemoteDescription(serverResponse);
            }

            if (this.options.onConnected) {
                this.options.onConnected();
            }
        } catch (error) {
            console.error('Error connecting:', error);
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
            console.error('Error setting up audio analysis:', error);
        }
    }

    private startAnalysis() {
        if (!this.analyser || !this.dataArray || !this.options.onAudioLevel) return;

        // Add throttling to prevent too many updates
        let lastUpdateTime = 0;
        const throttleInterval = 100; // Only update every 100ms

        const analyze = () => {
            this.analyser!.getByteFrequencyData(this.dataArray!);

            const currentTime = Date.now();
            // Only update if enough time has passed since last update
            if (currentTime - lastUpdateTime > throttleInterval) {
                // Calculate average volume level (0-1)
                let sum = 0;
                for (let i = 0; i < this.dataArray!.length; i++) {
                    sum += this.dataArray![i];
                }
                const average = sum / this.dataArray!.length / 255;

                this.options.onAudioLevel!(average);
                lastUpdateTime = currentTime;
            }

            this.animationFrameId = requestAnimationFrame(analyze);
        };

        this.animationFrameId = requestAnimationFrame(analyze);
    }

    private stopAnalysis() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.analyser = null;
        this.dataArray = null;
    }

    disconnect() {
        this.stopAnalysis();

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }

        this.dataChannel = null;

        if (this.options.onDisconnected) {
            this.options.onDisconnected();
        }
    }
}

// Main component
export function UnifiedAvatarChat() {
    // State management
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [avatarStatus, setAvatarStatus] = useState<'connected' | 'disconnected'>('disconnected');
    const [currentVisemes, setCurrentVisemes] = useState<Record<string, number>>({});
    const [audioLevel, setAudioLevel] = useState(0);

    // Refs
    const webrtcClientRef = useRef<UnifiedWebRTCClient | null>(null);
    const audioRef = useRef<HTMLAudioElement>(null);
    const outputDeviceIdRef = useRef<string | undefined>(undefined);
    const avatarWebSocketRef = useRef<WebSocket | null>(null);
    const chatBottomRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneRef = useRef<any>(null);
    const avatarRef = useRef<any>(null);
    const rendererRef = useRef<any>(null);
    const webrtcId = useRef(Math.random().toString(36).substring(7));

    // Auto-scroll chat to bottom
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    // Initialize Avatar WebSocket
    const initAvatarWebSocket = useCallback(() => {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//localhost:8001/ws/avatar`);

            ws.onopen = () => {
                setAvatarStatus('connected');
                console.log('✅ Avatar WebSocket connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data: AvatarWebSocketData = JSON.parse(event.data);
                    if (data.type === 'viseme_update' && data.visemes) {
                        setCurrentVisemes(data.visemes);
                        // Apply to 3D avatar if loaded
                        if (avatarRef.current && data.blend_shapes) {
                            applyBlendShapesToAvatar(data.blend_shapes);
                        }
                    }
                } catch (error) {
                    console.error('Avatar WebSocket message error:', error);
                }
            };

            ws.onclose = () => {
                setAvatarStatus('disconnected');
                console.log('❌ Avatar WebSocket disconnected');
                // Reconnect after 3 seconds
                setTimeout(initAvatarWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('Avatar WebSocket error:', error);
            };

            avatarWebSocketRef.current = ws;

        } catch (error) {
            console.error('Failed to initialize Avatar WebSocket:', error);
            setTimeout(initAvatarWebSocket, 3000);
        }
    }, []);

    // Initialize Three.js Avatar Scene
    const initThreeJS = useCallback(async () => {
        if (!canvasRef.current) return false;

        try {
            // Dynamic import of Three.js
            const THREE = await import('three');
            // const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
            const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');

            const canvas = canvasRef.current;
            const container = canvas.parentElement;
            if (!container) return false;

            const aspect = container.clientWidth / container.clientHeight || 1;

            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);

            // Camera setup
            const camera = new THREE.PerspectiveCamera(25, aspect, 0.1, 1000);
            camera.position.set(0, 1.5, 2);
            // camera.position.set(0, 1.5, 20);
            camera.lookAt(0, 1.5, -2);
            // camera.lookAt(0, 1.5, 0);

            // Renderer setup
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
            const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
            scene.add(hemisphereLight);

            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
            directionalLight1.position.set(0, 1, 2);
            scene.add(directionalLight1);

            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
            directionalLight2.position.set(0, 1, -2);
            scene.add(directionalLight2);

            // Store references
            sceneRef.current = { scene, camera, renderer };
            rendererRef.current = renderer;

            // Load avatar
            const loader = new GLTFLoader();
            //const avatarPaths = ['/static/test5.glb', '/static/avatar.glb'];
            // const avatarPaths = ['/static/joined1111.glb'];
            const avatarPaths = ['/static/joined2.glb'];
            // const avatarPaths = ['/static/fixedaf.glb'];

            for (const path of avatarPaths) {
                try {
                    console.log(`Loading avatar: ${path}`);
                    const gltf = await new Promise<any>((resolve, reject) => {
                        loader.load(
                            `http://localhost:8001${path}`,
                            resolve,
                            undefined,
                            reject
                        );
                    });

                    if (avatarRef.current) {
                        scene.remove(avatarRef.current);
                    }

                    avatarRef.current = gltf.scene;
                    scene.add(gltf.scene);

                    // ✅ Fix: Only run after avatar is set
                    const box = new THREE.Box3().setFromObject(gltf.scene);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());

                    // camera.position.copy(center.clone().add(new THREE.Vector3(0, size.y * 0.5, size.z * 2)));
                    // camera.lookAt(center);

                    // Move camera up to face level (1.5m is typical human eye height)
                    const faceTarget = center.clone();
                    faceTarget.y += size.y * 0.35;  // adjust this if face is too high/low

                    camera.position.set(faceTarget.x, faceTarget.y, faceTarget.z + size.z * 2);
                    camera.lookAt(faceTarget);

                    console.log('📦 Avatar bounds:', box);
                    console.log('📦 Bounding Box Min:', box.min);
                    console.log('📦 Bounding Box Max:', box.max);
                    console.log('📏 Size:', size);

                    // Also fix materials
                    gltf.scene.traverse((child) => {

                        if (child.isMesh) {
                            console.log('🧩 Mesh found:', child.name);
                            console.log('    ➤ Vertices:', child.geometry?.attributes?.position?.count);
                            console.log('    ➤ Morph targets:', child.morphTargetDictionary && Object.keys(child.morphTargetDictionary));
                        }

                        if (child.isMesh && child.material) {
                            child.material.transparent = false;
                            child.material.opacity = 1.0;
                            child.material.depthWrite = true;
                            child.material.side = THREE.FrontSide;

                            console.log('🎨 Material:', child.material.name, {
                                color: child.material.color?.getHexString(),
                                emissive: child.material.emissive?.getHexString(),
                                opacity: child.material.opacity,
                                transparent: child.material.transparent,
                            });

                            // Debug: force emissive white for visibility
                            child.material.emissive?.set(0xffffff);
                        }
                    });

                    // Optional extra light
                    const light = new THREE.PointLight(0xffffff, 10);
                    light.position.set(0, 2, 2);
                    scene.add(light);


                    // Find morph targets
                    gltf.scene.traverse((child: any) => {
                        if (child.isMesh && child.morphTargetInfluences) {
                            child.userData.morphTargets = child.morphTargetDictionary;
                            console.log('🎭 Morph targets found:', Object.keys(child.morphTargetDictionary || {}));
                        }
                    });

                    console.log(`✅ Avatar loaded: ${path}`);
                    break;

                } catch (error) {
                    console.log(`Failed to load: ${path}`);
                }
            }



            // Start render loop
            const animate = () => {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            };
            animate();

            // Handle window resize
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



    // Apply blend shapes to avatar
    const applyBlendShapesToAvatar = useCallback((blendShapes: Record<string, number>) => {
        if (!avatarRef.current) return;

        avatarRef.current.traverse((child: any) => {
            if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
                // Reset all morph targets
                for (let i = 0; i < child.morphTargetInfluences.length; i++) {
                    child.morphTargetInfluences[i] = 0;
                }

                // Apply new blend shape values
                for (const [shapeName, value] of Object.entries(blendShapes)) {
                    const index = child.morphTargetDictionary[shapeName];
                    if (index !== undefined) {
                        child.morphTargetInfluences[index] = value;
                    }
                }
            }
        });
    }, []);

    // Initialize AI Chat SSE
    const initAIChatSSE = useCallback(() => {
        const eventSource = new EventSource(`http://localhost:8001/updates?webrtc_id=${webrtcId.current}`);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === "stt" || data.type === "llm") {
                    setChatMessages(prev => [...prev, {
                        type: data.type,
                        text: data.text,
                        timestamp: Date.now()
                    }]);
                } else if (data.type === "visemes") {
                    console.log("Received AI visemes:", data);
                    // Visemes are automatically applied via the backend WebSocket
                }
            } catch (err) {
                console.error("SSE parse error", err);
            }
        };

        eventSource.onerror = (err) => {
            console.error("SSE error", err);
        };

        return eventSource;
    }, []);

    // Voice recording handlers
    const startRecording = useCallback(async () => {
        try {
            if (!webrtcClientRef.current) {
                webrtcClientRef.current = new UnifiedWebRTCClient({
                    webrtcId: webrtcId.current,
                    onAudioLevel: (level) => setAudioLevel(level),
                    onConnected: () => setIsConnected(true),
                    onDisconnected: () => setIsConnected(false)
                });
            }

            await webrtcClientRef.current.connect();
            setIsRecording(true);
            setIsConnected(true);

        } catch (error) {
            console.error('Failed to start recording:', error);
            setChatMessages(prev => [...prev, {
                type: "llm",
                text: "Failed to start recording. Please check your microphone permissions.",
                timestamp: Date.now()
            }]);
        }
    }, []);

    const stopRecording = useCallback(() => {
        if (webrtcClientRef.current) {
            webrtcClientRef.current.disconnect();
            setIsRecording(false);
            setIsConnected(false);
        }
    }, []);

    // Voice recording toggle handler
    const toggleRecording = useCallback(async () => {
        if (isRecording) {
            // Stop recording
            if (webrtcClientRef.current) {
                webrtcClientRef.current.disconnect();
                setIsRecording(false);
                setIsConnected(false);
            }
        } else {
            // Start recording
            try {
                if (!webrtcClientRef.current) {
                    // webrtcClientRef.current = new UnifiedWebRTCClient({
                    //     webrtcId: webrtcId.current,
                    //     onAudioLevel: (level) => setAudioLevel(level),
                    //     onConnected: () => setIsConnected(true),
                    //     onDisconnected: () => setIsConnected(false)
                    // });

                    webrtcClientRef.current = new UnifiedWebRTCClient({
                        webrtcId: webrtcId.current,
                        onAudioLevel: (level) => setAudioLevel(level),
                        onConnected: () => setIsConnected(true),
                        onDisconnected: () => setIsConnected(false),
                        // ADD THIS:
                        onAudioStream: (stream: MediaStream) => {
                            console.log('🎵 Received audio stream:', stream);

                            if (!audioRef.current) {
                                // Create audio element if it doesn't exist
                                const audio = document.createElement('audio');
                                audio.autoplay = true;
                                audio.volume = 1.0;
                                document.body.appendChild(audio);
                                audioRef.current = audio;
                            }

                            audioRef.current.srcObject = stream;

                            // Handle output device if specified
                            if (outputDeviceIdRef.current && 'setSinkId' in HTMLAudioElement.prototype) {
                                (audioRef.current as any).setSinkId(outputDeviceIdRef.current)
                                    .catch((err: any) => console.error('Error setting audio output device:', err));
                            }

                            // Force play (handle autoplay restrictions)
                            audioRef.current.play().then(() => {
                                console.log('🔊 Audio playing successfully');
                            }).catch(error => {
                                console.error('❌ Audio autoplay blocked:', error);
                                // You might want to show a user interaction to enable audio
                            });
                        }
                    });
                }

                await webrtcClientRef.current.connect();
                setIsRecording(true);
                setIsConnected(true);

            } catch (error) {
                console.error('Failed to start recording:', error);
                setChatMessages(prev => [...prev, {
                    type: "llm",
                    text: "Failed to start recording. Please check your microphone permissions.",
                    timestamp: Date.now()
                }]);
            }
        }
    }, [isRecording]);

    // Initialize everything
    useEffect(() => {
        const init = async () => {
            console.log('🚀 Initializing Unified Avatar Chat...');

            // Initialize avatar system
            initAvatarWebSocket();

            // Initialize Three.js
            const threeReady = await initThreeJS();
            if (!threeReady) {
                console.error('Failed to initialize Three.js');
            }

            // Initialize AI chat
            const eventSource = initAIChatSSE();

            console.log('✅ Initialization complete');

            // Cleanup function
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
    }, [initAvatarWebSocket, initThreeJS, initAIChatSSE]);

    // Get dominant viseme for display
    const getDominantViseme = useCallback(() => {
        const entries = Object.entries(currentVisemes);
        if (entries.length === 0) return 'REST';

        const dominant = entries.reduce((max, current) =>
            current[1] > max[1] ? current : max
        );

        return dominant[1] > 0.5 ? dominant[0] : 'REST';
    }, [currentVisemes]);

    // Handle device change
    const handleDeviceChange = useCallback((deviceId: string, type: 'input' | 'output') => {
        if (!webrtcClientRef.current) return;

        if (type === 'input') {
            webrtcClientRef.current.setAudioInputDevice(deviceId);
        } else if (type === 'output') {
            webrtcClientRef.current.setAudioOutputDevice(deviceId);
            outputDeviceIdRef.current = deviceId;

            if (audioRef.current && audioRef.current.srcObject && 'setSinkId' in HTMLAudioElement.prototype) {
                try {
                    (audioRef.current as any).setSinkId(deviceId)
                        .catch((err: any) => {
                            console.error('Error setting audio output device:', err);
                        });
                } catch (err) {
                    console.error('Error applying setSinkId:', err);
                }
            }
        }
    }, []);

    return (
        <div className="relative w-full h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 overflow-hidden">
            {/* Background Effects */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-1/4 -right-1/4 w-1/2 h-1/2 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"></div>
                <div className="absolute -bottom-1/4 -left-1/4 w-1/2 h-1/2 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20"></div>
            </div>

            {/* Main Container */}
            <div className="relative z-10 h-full flex">
                {/* Avatar Panel */}
                <div className="flex-1 bg-black/20 backdrop-blur-sm border-r border-white/10">
                    {/* Avatar Header */}
                    <div className="bg-black/30 backdrop-blur-sm p-4 border-b border-white/10">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            🤖 3D Avatar
                            <span className={`inline-block w-3 h-3 rounded-full ${
                                avatarStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                            }`}></span>
                        </h2>
                        <p className="text-sm text-gray-300">Real-time Facial Animation</p>
                    </div>

                    {/* Avatar Canvas Container */}
                    <div className="relative h-[calc(100vh-80px)]">
                        <canvas
                            ref={canvasRef}
                            className="w-full h-full"
                        />

                        {/* Viseme Debug Display */}
                        <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm p-3 rounded-lg text-white">
                            <div className="text-xs text-gray-300 mb-1">Current Viseme:</div>
                            <div className="text-lg font-mono font-bold text-green-400">
                                {getDominantViseme()}
                            </div>
                        </div>

                        {/* Loading State */}
                        {!avatarRef.current && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="text-center text-white">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                                    <p>Loading 3D Avatar...</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* AI Chat Panel */}
                <div className="flex-1 bg-black/20 backdrop-blur-sm flex flex-col">
                    {/* Chat Header */}
                    <div className="bg-black/30 backdrop-blur-sm p-4 border-b border-white/10">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            🧠 AI Assistant
                            <span className={`inline-block w-3 h-3 rounded-full ${
                                isConnected ? 'bg-green-400' : 'bg-red-400'
                            }`}></span>
                        </h2>
                        <p className="text-sm text-gray-300">Real-time Speech & Chat</p>
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
                                    <p className="text-sm mt-1">Press and hold the microphone to speak</p>
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

                    {/* Audio Level Indicator */}
                    <div className="px-4 mb-2">
                        <div className="h-1 w-full bg-white/20 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-blue-400 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min(audioLevel * 100, 100)}%` }}
                                transition={{ type: "spring", damping: 15 }}
                            />
                        </div>
                    </div>

                    {/* Voice Controls */}
                    <div className="p-4">
                        <div className="flex items-center justify-center gap-4">
                            {/* Voice Button */}
                            {/*<button*/}
                            {/*    className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all duration-300 ${*/}
                            {/*        isRecording*/}
                            {/*            ? "bg-red-500 hover:bg-red-600 animate-pulse"*/}
                            {/*            : "bg-blue-500 hover:bg-blue-600 hover:scale-110"*/}
                            {/*    } text-white shadow-lg`}*/}
                            {/*    onMouseDown={startRecording}*/}
                            {/*    onMouseUp={stopRecording}*/}
                            {/*    onMouseLeave={stopRecording}*/}
                            {/*    onTouchStart={startRecording}*/}
                            {/*    onTouchEnd={stopRecording}*/}
                            {/*    disabled={false}*/}
                            {/*>*/}
                            {/*    {isRecording ? "🔴" : "🎤"}*/}
                            {/*</button>*/}

                            {/* Voice Toggle Button */}
                            <button
                                className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all duration-300 ${
                                    isRecording
                                        ? "bg-red-500 hover:bg-red-600 animate-pulse"
                                        : "bg-blue-500 hover:bg-blue-600 hover:scale-110"
                                } text-white shadow-lg`}
                                onClick={toggleRecording}
                                disabled={false}
                            >
                                {isRecording ? "🔴" : "🎤"}
                            </button>

                            {/* Status Display */}
                            <div className="flex flex-col gap-1 text-xs text-gray-300">
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${
                                        avatarStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                                    }`}></span>
                                    <span>Avatar</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${
                                        isConnected ? 'bg-green-400' : 'bg-red-400'
                                    }`}></span>
                                    <span>AI Chat</span>
                                </div>
                            </div>
                        </div>

                        {/* Instructions */}
                        <div className="mt-3 text-center text-xs text-gray-400">
                            {isRecording ? (
                                "🎙️ Recording... Release to send"
                            ) : (
                                "Press and hold microphone to speak"
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Export for use in your Next.js app
export default UnifiedAvatarChat;