"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

// -----------------------
//  Types
// -----------------------
interface ChatMessage {
  type: "stt" | "llm";
  text: string;
  timestamp?: number;
}

interface VRMWebSocketData {
  type: string;
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

// -----------------------
//  WebRTC Client (unchanged except hard‑coded URLs made relative)
// -----------------------
class PureWebRTCClient {
  private peerConnection: RTCPeerConnection | null = null;
  private mediaStream: MediaStream | null = null;
  private dataChannel: RTCDataChannel | null = null;
  private options: WebRTCClientOptions;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private animationFrameId: number | null = null;
  private currentInputDeviceId?: string;
  private currentOutputDeviceId?: string;
  private webrtcId?: string;

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

      // --- getUserMedia ---
      const constraints: MediaStreamConstraints = {
        audio: this.currentInputDeviceId ? { deviceId: { exact: this.currentInputDeviceId } } : true,
      };
      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

      this.setupAudioAnalysis();
      this.mediaStream.getTracks().forEach((t) => this.peerConnection!.addTrack(t, this.mediaStream!));

      this.peerConnection.addEventListener("track", (ev) => {
        if (this.options.onAudioStream) this.options.onAudioStream(ev.streams[0]);
      });

      this.dataChannel = this.peerConnection.createDataChannel("text");
      this.dataChannel.addEventListener("message", (e) => {
        if (this.options.onMessage) this.options.onMessage(JSON.parse(e.data));
      });

      const offer = await this.peerConnection.createOffer();
      await this.peerConnection.setLocalDescription(offer);

      const resp = await fetch(`/webrtc/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type, webrtc_id: this.webrtcId }),
      });
      const answer = await resp.json();
      await this.peerConnection.setRemoteDescription(answer);

      this.options.onConnected?.();
    } catch (err) {
      console.error(err);
      this.disconnect();
      throw err;
    }
  }

  private setupAudioAnalysis() {
    if (!this.mediaStream) return;
    this.audioContext = new AudioContext();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    const src = this.audioContext.createMediaStreamSource(this.mediaStream);
    src.connect(this.analyser);
    const bufLen = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(bufLen);

    const tick = () => {
      this.analyser!.getByteFrequencyData(this.dataArray!);
      const avg = this.dataArray!.reduce((a, b) => a + b) / (bufLen * 255);
      this.options.onAudioLevel?.(avg);
      this.animationFrameId = requestAnimationFrame(tick);
    };
    this.animationFrameId = requestAnimationFrame(tick);
  }

  disconnect() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    if (this.audioContext) this.audioContext.close();
    this.mediaStream?.getTracks().forEach((t) => t.stop());
    this.peerConnection?.close();
    this.mediaStream = null;
    this.peerConnection = null;
    this.options.onDisconnected?.();
  }
}

// -----------------------
//  Main Component
// -----------------------
export function PureLipSyncAvatar() {
  // ---- state ----
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [blendShapes, setBlendShapes] = useState<Record<string, number>>({});
  const [avatarWS, setAvatarWS] = useState<"connected" | "disconnected">("disconnected");
  const [vrmReady, setVrmReady] = useState(false);

  // ---- refs ----
  const wsRef = useRef<WebSocket | null>(null);
  const webrtcRef = useRef<PureWebRTCClient | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const webrtcId = useRef(Math.random().toString(36).slice(2));
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const vrmRef = useRef<any>(null);
  const sceneRef = useRef<any>(null);

  // ---------------- WebSocket avatar ----------------
  const initAvatarWS = useCallback(() => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//localhost:8001/ws/avatar`);
    //const ws = new WebSocket(`${proto}//${location.host}/ws/avatar`);

    ws.onopen = () => setAvatarWS("connected");

    ws.onmessage = (evt) => {
      try {
        const data: VRMWebSocketData = JSON.parse(evt.data);
        if (data.type === "viseme_update" && data.blend_shapes) {
          setBlendShapes(data.blend_shapes);
          applyBlendShapes(data.blend_shapes);
        }
      } catch (e) {
        console.error(e);
      }
    };

    ws.onclose = () => {
      setAvatarWS("disconnected");
      setTimeout(initAvatarWS, 3000);
    };

    wsRef.current = ws;
  }, []);

  // ---------------- Three.js + VRM (no emotion) ---------------
  const initThree = useCallback(async () => {
    if (!canvasRef.current) return;
    const THREE = await import("three");
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const { VRM, VRMLoaderPlugin, VRMUtils } = await import("@pixiv/three-vrm");

    // scene, cam, renderer
    const container = canvasRef.current.parentElement!;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 1.5, 2);
    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.8));

    // load model
    const loader = new GLTFLoader();
    loader.register((p) => new VRMLoaderPlugin(p));
    const gltf = await loader.loadAsync("/static/4thjuly.vrm");
    const vrm = gltf.userData.vrm as any;
    VRMUtils.removeUnnecessaryVertices(vrm.scene);
    VRMUtils.removeUnnecessaryJoints(vrm.scene);
    scene.add(vrm.scene);
    vrmRef.current = vrm;
    setVrmReady(true);

    // render loop
    const animate = () => {
      requestAnimationFrame(animate);
      vrm.update(0.016);
      renderer.render(scene, camera);
    };
    animate();
    sceneRef.current = { scene, camera, renderer };
  }, []);

  // --------------- apply blend shapes (no emotion) -------------
  const applyBlendShapes = useCallback((bs: Record<string, number>) => {
    if (!vrmRef.current) return;
    if (vrmRef.current.expressionManager) {
      Object.entries(bs).forEach(([k, v]) => {
        if (vrmRef.current.expressionManager.expressionMap[k]) {
          vrmRef.current.expressionManager.setValue(k, v);
        }
      });
      vrmRef.current.expressionManager.update();
    } else {
      const target = vrmRef.current.scene;
      target.traverse((child: any) => {
        if (child.isMesh && child.morphTargetInfluences && child.morphTargetDictionary) {
          for (const [name, val] of Object.entries(bs)) {
            const idx = child.morphTargetDictionary[name];
            if (idx !== undefined) child.morphTargetInfluences[idx] = val;
          }
        }
      });
    }
  }, []);

  // ---------------- SSE -------------------
  useEffect(() => {
    const es = new EventSource(`/updates?webrtc_id=${webrtcId.current}`);
    es.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.type === "stt" || data.type === "llm") {
        setChat((c) => [...c, { type: data.type, text: data.text, timestamp: Date.now() }]);
      }
    };
    return () => es.close();
  }, []);

  // ---------------- init mounts ----------------
  useEffect(() => {
    initAvatarWS();
    initThree();
  }, [initAvatarWS, initThree]);

  // --------------- recording control --------------
  const toggleRec = async () => {
    if (isRecording) {
      webrtcRef.current?.disconnect();
      setIsRecording(false);
    } else {
      if (!webrtcRef.current) {
        webrtcRef.current = new PureWebRTCClient({
          webrtcId: webrtcId.current,
          onAudioLevel: setAudioLevel,
          onDisconnected: () => setIsRecording(false),
          onAudioStream: (s) => {
            if (!audioRef.current) {
              audioRef.current = new Audio();
              audioRef.current.autoplay = true;
            }
            audioRef.current.srcObject = s;
          },
        });
      }
      await webrtcRef.current.connect();
      setIsRecording(true);
    }
  };

  // --------------- dominant viseme memo -------------
  const dominant = useMemo(() => {
    let max = 0;
    let key = "sil";
    Object.entries(blendShapes).forEach(([k, v]) => {
      if (v > max) {
        max = v;
        key = k;
      }
    });
    return max > 0.1 ? key : "sil";
  }, [blendShapes]);

  // ---------------- JSX ------------------
  return (
    <div className="w-full h-screen bg-black text-white flex">
      {/* Avatar */}
      <div className="flex-1 relative border-r border-gray-700">
        <canvas ref={canvasRef} className="w-full h-full" />
        <div className="absolute top-2 left-2 text-xs bg-black/50 p-2 rounded">
          <div>WS: {avatarWS}</div>
          <div>VRM: {vrmReady ? "ready" : "loading"}</div>
          <div>Viseme: {dominant}</div>
        </div>
      </div>

      {/* Chat & Controls */}
      <div className="w-96 flex flex-col">
        <div className="flex-1 overflow-y-auto p-3 space-y-2 bg-gray-900/50">
          <AnimatePresence>
            {chat.map((m, i) => (
              <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={m.type === "stt" ? "text-right" : "text-left"}>
                <span className="inline-block bg-gray-700 rounded px-2 py-1 text-sm">
                  {m.text}
                </span>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
        <button onClick={toggleRec} className={`h-16 ${isRecording ? "bg-red-600" : "bg-blue-600"}`}>{isRecording ? "Stop" : "Talk"}</button>
        <div className="h-2 bg-gray-700">
          <div style={{ width: `${Math.min(audioLevel * 100, 100)}%` }} className="h-full bg-green-500" />
        </div>
      </div>
    </div>
  );
}

export default PureLipSyncAvatar;