"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Types for VRM-enhanced system
interface EnhancedVisemeData {
    viseme: string;
    start_time: number;
    end_time: number;
    confidence: number;
    phoneme?: string;
    emotion?: string;
}

interface ChatMessage {
    type: "stt" | "llm";
    text: string;
    timestamp?: number;
}

interface VRMWebSocketData {
    type: string;
    blend_shapes?: Record<string, number>;
    timestamp?: number;
    emotion?: string;
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

// Enhanced WebRTC Client for VRM system (using working port and endpoints)
class EnhancedVRMWebRTCClient {
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

    setAudioInputDevice(deviceId: string) {
        this.currentInputDeviceId = deviceId;
        if (this.peerConnection) {
            this.disconnect();
            this.connect();
        }
    }

    setAudioOutputDevice(deviceId: string) {
        this.currentOutputDeviceId = deviceId;
        this.options.audioOutputDeviceId = deviceId;
    }

    async connect() {
        try {
            this.peerConnection = new RTCPeerConnection();

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
                    if (this.currentOutputDeviceId && 'setSinkId' in HTMLAudioElement.prototype) {
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

            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            // Fixed: Use port 8000 and relative URL to match your unified_server.py
            const response = await fetch('http://localhost:8001/webrtc/offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                mode: 'cors',
                credentials: 'same-origin',
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    webrtc_id: this.webrtcId
                })
            });

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

        let lastUpdateTime = 0;
        const throttleInterval = 100;

        const analyze = () => {
            this.analyser!.getByteFrequencyData(this.dataArray!);

            const currentTime = Date.now();
            if (currentTime - lastUpdateTime > throttleInterval) {
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

// Main Enhanced VRM Avatar Chat Component
export function EnhancedVRMAvatarChat() {
    // State management
    const [isConnected, setIsConnected] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [avatarStatus, setAvatarStatus] = useState<'connected' | 'disconnected'>('disconnected');
    const [currentBlendShapes, setCurrentBlendShapes] = useState<Record<string, number>>({});
    const [currentEmotion, setCurrentEmotion] = useState<string>('neutral');
    const [audioLevel, setAudioLevel] = useState(0);
    const [vrmLoaded, setVrmLoaded] = useState(false);
    const [dominantViseme, setDominantViseme] = useState<string>('sil');

    // Refs
    const webrtcClientRef = useRef<EnhancedVRMWebRTCClient | null>(null);
    const audioRef = useRef<HTMLAudioElement>(null);
    const outputDeviceIdRef = useRef<string | undefined>(undefined);
    const avatarWebSocketRef = useRef<WebSocket | null>(null);
    const chatBottomRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const sceneRef = useRef<any>(null);
    const vrmRef = useRef<any>(null);
    const rendererRef = useRef<any>(null);
    const webrtcId = useRef(Math.random().toString(36).substring(7));

    // Auto-scroll chat to bottom
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    // Initialize Enhanced VRM Avatar WebSocket (Fixed URL)
    const initVRMAvatarWebSocket = useCallback(() => {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//localhost:8001/ws/avatar`);

            ws.onopen = () => {
                setAvatarStatus('connected');
                console.log('✅ Enhanced VRM Avatar WebSocket connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data: VRMWebSocketData = JSON.parse(event.data);
                    if (data.type === 'viseme_update') {
                        if (data.blend_shapes) {
                            setCurrentBlendShapes(data.blend_shapes);
                            applyVRMBlendShapes(data.blend_shapes);

                            // Update dominant viseme
                            const dominant = findDominantBlendShape(data.blend_shapes);
                            setDominantViseme(dominant);
                        }
                        if (data.emotion) {
                            setCurrentEmotion(data.emotion);
                        }
                    }
                } catch (error) {
                    console.error('VRM Avatar WebSocket message error:', error);
                }
            };

            ws.onclose = () => {
                setAvatarStatus('disconnected');
                console.log('❌ VRM Avatar WebSocket disconnected');
                setTimeout(initVRMAvatarWebSocket, 3000);
            };

            ws.onerror = (error) => {
                console.error('VRM Avatar WebSocket error:', error);
            };

            avatarWebSocketRef.current = ws;

        } catch (error) {
            console.error('Failed to initialize VRM Avatar WebSocket:', error);
            setTimeout(initVRMAvatarWebSocket, 3000);
        }
    }, []);

    // Initialize Three.js with VRM Support (Fixed paths and imports)
    const initThreeJSWithVRM = useCallback(async () => {
        if (!canvasRef.current) return false;

        try {
            // Dynamic imports for Three.js and VRM
            const THREE = await import('three');
            const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');

            // Try to import VRM - if it fails, fallback to GLB
            let VRMLoaderPlugin, VRMUtils;
            let isVRMSupported = false;

            try {
                const vrmModule = await import('@pixiv/three-vrm');
                VRMLoaderPlugin = vrmModule.VRMLoaderPlugin;
                VRMUtils = vrmModule.VRMUtils;
                isVRMSupported = true;
                console.log('✅ VRM support loaded');
            } catch (vrmError) {
                console.warn('⚠️ VRM support not available, falling back to GLB:', vrmError);
                isVRMSupported = false;
            }

            const canvas = canvasRef.current;
            const container = canvas.parentElement;
            if (!container) return false;

            const aspect = container.clientWidth / container.clientHeight || 1;

            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);

            // Camera setup (optimized for face focus)
            const camera = new THREE.PerspectiveCamera(35, aspect, 0.1, 1000);
            camera.position.set(0, 1.5, 2);
            camera.lookAt(0, 1.5, 0);

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

            // Enhanced lighting for VRM models
            const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
            scene.add(hemi);

            const key = new THREE.DirectionalLight(0xffffff, 1.2);
            key.position.set(0, 1, 2);
            scene.add(key);

            const rim = new THREE.DirectionalLight(0xffffff, 0.6);
            rim.position.set(0, 1, -2);
            scene.add(rim);

            // Initialize GLTF loader
            const gltfLoader = new GLTFLoader();

            // Add VRM plugin if available
            if (isVRMSupported && VRMLoaderPlugin) {
                gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
            }

            // Store references
            sceneRef.current = { scene, camera, renderer };
            rendererRef.current = renderer;

            // Load avatar - try VRM first, then GLB fallback
            // const avatarPaths = [
            //     '/static/avatar.vrm',
            //     '/static/joined1111.vrm',
            //     '/static/test.vrm',
            //     // Fallback to GLB files
            //     '/static/joined2.glb',
            //     '/static/joined1111.glb',
            //     '/static/avatar.glb'
            // ];
            const avatarPaths = ['/static/4thjuly.vrm'];

            for (const path of avatarPaths) {
                try {
                    console.log(`Loading avatar: ${path}`);

                    const gltf = await new Promise<any>((resolve, reject) => {
                        gltfLoader.load(`http://localhost:8001${path}`, resolve, undefined, reject);
                    });

                    // Check if this is a VRM file
                    const vrm = gltf.userData?.vrm;
                    const isVRM = !!vrm;

                    if (vrmRef.current) {
                        scene.remove(vrmRef.current.scene || vrmRef.current);
                    }

                    if (isVRM) {
                        // Handle VRM
                        vrmRef.current = vrm;
                        scene.add(vrm.scene);

                        // VRM-specific optimizations
                        if (VRMUtils) {
                            VRMUtils.removeUnnecessaryVertices(vrm.scene);
                            VRMUtils.removeUnnecessaryJoints(vrm.scene);
                        }

                        // Log VRM blend shapes for debugging
                        if (vrm.expressionManager) {
                            console.log('✅ VRM Expression Manager found');
                            console.log('VRM expressions:', Object.keys(vrm.expressionManager.expressionMap || {}));
                        }

                        // Enhanced VRM blend shape detection
                        vrm.scene.traverse((child: any) => {
                            if (child.isMesh && child.morphTargetInfluences) {
                                child.userData.morphTargets = child.morphTargetDictionary;
                                if (child.morphTargetDictionary) {
                                    const morphNames = Object.keys(child.morphTargetDictionary);
                                    console.log('🎭 VRM morph targets found:', morphNames);
                                }
                            }
                        });

                        // Camera positioning for VRM
                        const box = new THREE.Box3().setFromObject(vrm.scene);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());

                        const faceTarget = center.clone();
                        faceTarget.y += size.y * 0.35;

                        camera.position.set(faceTarget.x, faceTarget.y, faceTarget.z + size.z * 2);
                        camera.lookAt(faceTarget);

                        console.log(`✅ VRM avatar loaded: ${path}`);
                    } else {
                        // Handle GLB
                        vrmRef.current = gltf.scene;
                        scene.add(gltf.scene);

                        // Camera positioning for GLB
                        const box = new THREE.Box3().setFromObject(gltf.scene);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());

                        const faceTarget = center.clone();
                        faceTarget.y += size.y * 0.35;

                        camera.position.set(faceTarget.x, faceTarget.y, faceTarget.z + size.z * 2);
                        camera.lookAt(faceTarget);

                        // Find morph targets for GLB
                        gltf.scene.traverse((child: any) => {
                            if (child.isMesh && child.morphTargetInfluences) {
                                child.userData.morphTargets = child.morphTargetDictionary;
                                if (child.morphTargetDictionary) {
                                    console.log('🎭 GLB morph targets found:', Object.keys(child.morphTargetDictionary));
                                }
                            }
                        });

                        console.log(`✅ GLB avatar loaded: ${path}`);
                    }

                    setVrmLoaded(true);
                    break;

                } catch (error) {
                    console.log(`Failed to load: ${path}`);
                }
            }

            if (!vrmRef.current) {
                console.log('❌ No avatar found. Please place a .vrm or .glb file in /static/');
                setVrmLoaded(false);
            }

            // Start render loop
            const animate = () => {
                requestAnimationFrame(animate);

                // Update VRM if available
                if (vrmRef.current && vrmRef.current.update) {
                    vrmRef.current.update(0.016); // ~60fps delta time
                }

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

    // Add this VRM expression mapping
        const VRM_EXPRESSION_MAPPING = {
            'Fcl_MTH_A': 'aa',
            'Fcl_MTH_E': 'ee',
            'Fcl_MTH_I': 'ih',
            'Fcl_MTH_O': 'ou',
            'Fcl_MTH_U': 'ou',
            'Fcl_MTH_Small': 'ih',
            'Fcl_MTH_Large': 'aa',
            'Fcl_MTH_Close': 'sil',
            'Fcl_MTH_Neutral': 'neutral',
            'Fcl_ALL_Joy': 'happy',
            'Fcl_ALL_Sorrow': 'sad',
            'Fcl_ALL_Surprised': 'surprised',
            'Fcl_ALL_Angry': 'angry',
            'Fcl_EYE_Natural': 'relaxed',
            'Fcl_EYE_Joy': 'happy',
            'Fcl_EYE_Sorrow': 'sad'
        };

    // Apply VRM/GLB blend shapes
    const applyVRMBlendShapes = useCallback((blendShapes: Record<string, number>) => {
        console.log()
        if (!vrmRef.current) return;
        else console.log('workingggggggg')
        // Check if this is a VRM with expression manager
        // if (vrmRef.current.expressionManager) {
        //     for (const [shapeName, value] of Object.entries(blendShapes)) {
        //         if (vrmRef.current.expressionManager.expressionMap[shapeName]) {
        //             vrmRef.current.expressionManager.setValue(shapeName, Math.max(0, Math.min(1, value)));
        //             console.log(`Setting VRM expression: ${shapeName} to ${value}`);
        //         }
        //
        //     }
        //     vrmRef.current.expressionManager.update();
        // }


        // Replace the expression manager section with:
        if (vrmRef.current.expressionManager) {
            console.log('Available VRM expressions:', Object.keys(vrmRef.current.expressionManager.expressionMap || {}));

            for (const [shapeName, value] of Object.entries(blendShapes)) {
                // Try direct mapping first
                if (vrmRef.current.expressionManager.expressionMap[shapeName]) {
                    vrmRef.current.expressionManager.setValue(shapeName, Math.max(0, Math.min(1, value)));
                    console.log(`✅ Direct VRM expression: ${shapeName} = ${value}`);
                }
                // Try mapped name
                else if (VRM_EXPRESSION_MAPPING[shapeName]) {
                    const mappedName = VRM_EXPRESSION_MAPPING[shapeName];
                    if (vrmRef.current.expressionManager.expressionMap[mappedName]) {
                        vrmRef.current.expressionManager.setValue(mappedName, Math.max(0, Math.min(1, value)));
                        console.log(`✅ Mapped VRM expression: ${shapeName} -> ${mappedName} = ${value}`);
                    }
                }
                else {
                    console.log(`❌ No VRM expression found for: ${shapeName}`);
                }
            }
            vrmRef.current.expressionManager.update();
        }

        else {
            // Fallback to direct morph target manipulation (GLB or VRM without expression manager)
            const targetObject = vrmRef.current.scene || vrmRef.current;

            targetObject.traverse((child: any) => {
                if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
                    // Smooth decay
                    for (let i = 0; i < child.morphTargetInfluences.length; i++) {
                        child.morphTargetInfluences[i] *= 0.9;
                    }
                    console.log('Applying blend shapes to mesh:', child.name);

                    // Apply new values
                    for (const [shapeName, value] of Object.entries(blendShapes)) {
                        const index = child.morphTargetDictionary[shapeName];
                        if (index !== undefined && value > 0.001) {
                            const currentValue = child.morphTargetInfluences[index] || 0;
                            child.morphTargetInfluences[index] = currentValue + (value - currentValue) * 0.3;
                        }
                        console.log(`Setting morph target: ${shapeName} to ${child.morphTargetInfluences[index]}`);
                    }
                }
            });
        }
    }, []);

    // Find dominant blend shape for display
    const findDominantBlendShape = useCallback((blendShapes: Record<string, number>): string => {
        let dominantShape = 'sil';
        let maxWeight = 0;

        for (const [shapeName, weight] of Object.entries(blendShapes)) {
            if (weight > maxWeight) {
                maxWeight = weight;
                dominantShape = shapeName;
            }
        }

        return maxWeight > 0.1 ? dominantShape : 'sil';
    }, []);

    // Initialize Enhanced AI Chat SSE (Fixed URL)
    const initEnhancedAIChatSSE = useCallback(() => {
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
                    console.log("Received enhanced VRM visemes from AI:", data);
                    // Enhanced visemes are automatically processed by the backend
                }
            } catch (err) {
                console.error("Enhanced SSE parse error", err);
            }
        };

        eventSource.onerror = (err) => {
            console.error("Enhanced SSE error", err);
        };

        return eventSource;
    }, []);

    // Enhanced emotion setting for VRM
    const setVRMEmotion = useCallback(async (emotion: string) => {
        try {
            const response = await fetch('http://localhost:8001/set_emotion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ emotion: emotion })
            });

            if (response.ok) {
                setCurrentEmotion(emotion);
                console.log(`✅ VRM emotion set to: ${emotion}`);
            } else {
                console.error('Failed to set VRM emotion');
            }
        } catch (error) {
            console.error('Error setting VRM emotion:', error);
        }
    }, []);

    // Enhanced voice recording toggle handler
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
                    webrtcClientRef.current = new EnhancedVRMWebRTCClient({
                        webrtcId: webrtcId.current,
                        onAudioLevel: (level) => setAudioLevel(level),
                        onConnected: () => setIsConnected(true),
                        onDisconnected: () => setIsConnected(false),
                        onAudioStream: (stream: MediaStream) => {
                            console.log('🎵 Received audio stream:', stream);

                            if (!audioRef.current) {
                                const audio = document.createElement('audio');
                                audio.autoplay = true;
                                audio.volume = 1.0;
                                document.body.appendChild(audio);
                                audioRef.current = audio;
                            }

                            audioRef.current.srcObject = stream;

                            if (outputDeviceIdRef.current && 'setSinkId' in HTMLAudioElement.prototype) {
                                (audioRef.current as any).setSinkId(outputDeviceIdRef.current)
                                    .catch((err: any) => console.error('Error setting audio output device:', err));
                            }

                            audioRef.current.play().then(() => {
                                console.log('🔊 Audio playing successfully');
                            }).catch(error => {
                                console.error('❌ Audio autoplay blocked:', error);
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

    // Reset VRM avatar
    const resetVRMAvatar = useCallback(async () => {
        try {
            const response = await fetch('/reset_avatar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                console.log('✅ VRM avatar reset to neutral');
                setCurrentEmotion('neutral');
                setDominantViseme('sil');
            }
        } catch (error) {
            console.error('Error resetting VRM avatar:', error);
        }
    }, []);

    // Initialize everything
    useEffect(() => {
        const init = async () => {
            console.log('🚀 Initializing Enhanced VRM Avatar Chat...');

            // Initialize VRM avatar system
            initVRMAvatarWebSocket();

            // Initialize Three.js with VRM support
            const threeReady = await initThreeJSWithVRM();
            if (!threeReady) {
                console.error('Failed to initialize Three.js with VRM support');
            }

            // Initialize AI chat
            const eventSource = initEnhancedAIChatSSE();

            console.log('✅ Enhanced VRM initialization complete');

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
    }, [initVRMAvatarWebSocket, initThreeJSWithVRM, initEnhancedAIChatSSE]);



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
                {/* Enhanced VRM Avatar Panel */}
                <div className="flex-1 bg-black/20 backdrop-blur-sm border-r border-white/10">
                    {/* Avatar Header */}
                    <div className="bg-black/30 backdrop-blur-sm p-4 border-b border-white/10">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            🤖 Enhanced VRM Avatar
                            <span className={`inline-block w-3 h-3 rounded-full ${
                                avatarStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'
                            }`}></span>
                        </h2>
                        <p className="text-sm text-gray-300">Advanced VRoid Facial Animation</p>
                    </div>

                    {/* VRM Avatar Canvas Container */}
                    <div className="relative h-[calc(100vh-80px)]">
                        <canvas
                            ref={canvasRef}
                            className="w-full h-full"
                        />

                        {/* Enhanced Viseme Debug Display */}
                        <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm p-3 rounded-lg text-white">
                            <div className="text-xs text-gray-300 mb-2">VRM Status:</div>
                            <div className="space-y-1 text-xs">
                                <div>Emotion: <span className="text-blue-400">{currentEmotion}</span></div>
                                <div>Viseme: <span className="text-green-400">{dominantViseme}</span></div>
                                <div>Avatar: <span className={vrmLoaded ? "text-green-400" : "text-red-400"}>
                                    {vrmLoaded ? "Loaded" : "Loading..."}
                                </span></div>
                                <div>Shapes: <span className="text-yellow-400">{Object.keys(currentBlendShapes).length}</span></div>
                            </div>
                        </div>

                        {/* Enhanced Emotion Controls */}
                        <div className="absolute bottom-4 right-4 bg-black/50 backdrop-blur-sm p-3 rounded-lg">
                            <div className="text-xs text-white mb-2 text-center">VRM Emotions</div>
                            <div className="grid grid-cols-2 gap-1">
                                {['neutral', 'happy', 'sad', 'surprised', 'angry'].map((emotion) => (
                                    <button
                                        key={emotion}
                                        onClick={() => setVRMEmotion(emotion)}
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

                        {/* VRM Loading State */}
                        {!vrmLoaded && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="text-center text-white">
                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                                    <p>Loading Enhanced VRM Avatar...</p>
                                    <p className="text-sm text-gray-300 mt-2">Place your .vrm file in /static/</p>
                                </div>
                            </div>
                        )}

                        {/* VRM Reset Button */}
                        <div className="absolute bottom-4 left-4">
                            <button
                                onClick={resetVRMAvatar}
                                className="px-3 py-2 bg-black/50 backdrop-blur-sm text-white text-xs rounded-lg hover:bg-black/70 transition-colors"
                            >
                                🔄 Reset VRM
                            </button>
                        </div>
                    </div>
                </div>

                {/* Enhanced AI Chat Panel */}
                <div className="flex-1 bg-black/20 backdrop-blur-sm flex flex-col">
                    {/* Chat Header */}
                    <div className="bg-black/30 backdrop-blur-sm p-4 border-b border-white/10">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            🧠 AI Assistant + VRM
                            <span className={`inline-block w-3 h-3 rounded-full ${
                                isConnected ? 'bg-green-400' : 'bg-red-400'
                            }`}></span>
                        </h2>
                        <p className="text-sm text-gray-300">Real-time Speech & Chat with VRM Emotions</p>
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
                                    <p className="text-lg font-medium text-gray-300">Start a VRM conversation</p>
                                    <p className="text-sm mt-1">Click the microphone to speak with enhanced VRM facial animation</p>
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
                                            {msg.type === "llm" && (
                                                <div className="text-xs text-gray-600 mt-1 flex items-center gap-1">
                                                    <span>🎭</span>
                                                    <span>VRM: {currentEmotion}</span>
                                                </div>
                                            )}
                                        </div>
                                    </motion.div>
                                ))
                            )}
                        </AnimatePresence>
                        <div ref={chatBottomRef} />
                    </div>

                    {/* Enhanced Audio Level Indicator */}
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
                        <div className="text-xs text-gray-400 mt-1 text-center">
                            {isRecording ? `Audio Level: ${Math.round(audioLevel * 100)}%` : 'Audio Level Monitor'}
                        </div>
                    </div>

                    {/* Enhanced Voice Controls */}
                    <div className="p-4">
                        <div className="flex items-center justify-center gap-4">
                            {/* Enhanced Voice Toggle Button */}
                            <button
                                className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl transition-all duration-300 ${
                                    isRecording
                                        ? "bg-red-500 hover:bg-red-600 animate-pulse shadow-lg shadow-red-500/50"
                                        : "bg-blue-500 hover:bg-blue-600 hover:scale-110 shadow-lg shadow-blue-500/50"
                                } text-white`}
                                onClick={toggleRecording}
                                disabled={false}
                            >
                                {isRecording ? "🔴" : "🎤"}
                            </button>

                            {/* Enhanced Status Display */}
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
                                    <span>AI + TTS</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${
                                        vrmLoaded ? 'bg-green-400' : 'bg-yellow-400'
                                    }`}></span>
                                    <span>VRM Model</span>
                                </div>
                            </div>
                        </div>

                        {/* Enhanced Instructions */}
                        <div className="mt-3 text-center text-xs text-gray-400">
                            {isRecording ? (
                                <div className="space-y-1">
                                    <div className="flex items-center justify-center gap-2">
                                        <div className="animate-pulse w-2 h-2 bg-red-400 rounded-full"></div>
                                        <span>🎙️ Recording with VRM facial animation...</span>
                                    </div>
                                    <div>Click again to stop and process</div>
                                </div>
                            ) : (
                                <div className="space-y-1">
                                    <div>Click microphone to start enhanced VRM conversation</div>
                                    <div className="text-blue-400">✨ Advanced VRoid visemes • Emotional expressions • Real-time lip sync</div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Enhanced System Status Footer */}
                    <div className="border-t border-white/10 p-2 bg-black/20">
                        <div className="flex justify-between items-center text-xs text-gray-400">
                            <div className="flex items-center gap-3">
                                <span>VRM Shapes: {Object.keys(currentBlendShapes).length}</span>
                                <span>Emotion: {currentEmotion}</span>
                                <span>Viseme: {dominantViseme}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span>Enhanced VRoid System</span>
                                <div className={`w-2 h-2 rounded-full ${
                                    vrmLoaded && avatarStatus === 'connected' ? 'bg-green-400' : 'bg-yellow-400'
                                }`}></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Enhanced Floating VRM Info Panel */}
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black/50 backdrop-blur-sm px-4 py-2 rounded-lg text-white text-sm">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span>🎭</span>
                        <span>VRM Status:</span>
                        <span className={vrmLoaded ? "text-green-400" : "text-yellow-400"}>
                            {vrmLoaded ? "Active" : "Loading..."}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span>😊</span>
                        <span>Emotion:</span>
                        <span className="text-blue-400">{currentEmotion}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span>👄</span>
                        <span>Viseme:</span>
                        <span className="text-green-400">{dominantViseme}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Export for use in your Next.js app
export default EnhancedVRMAvatarChat;